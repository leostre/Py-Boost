import warnings
from inspect import signature

import cupy as cp

from py_boost.gpu.sketch_boost import SketchBoost
from py_boost.gpu.tree import DepthwiseTreeBuilder
from py_boost.gpu.utils import pad_and_move


def create_Z_orthogonal_to_X(X, M=None):
    n, d = X.shape
    if M is None:
        M = cp.random.randn(n, d)

    M_XT = M @ X.T
    diag_elements = cp.diag(M_XT)
    norms_sq = cp.sum(X ** 2, axis=1)

    epsilon = 1e-12
    norms_sq_safe = norms_sq.copy()
    norms_sq_safe[norms_sq_safe == 0] = epsilon

    alpha = diag_elements / norms_sq_safe
    D = cp.diag(alpha)
    Z = M - D @ X
    Z = Z.astype(cp.float32)
    return Z, M, alpha


class OrthoDepthwiseTreeBuilder(DepthwiseTreeBuilder):
    def build_tree(self, X, grad, hess, sample_weight=None, grad_fn=None, *val_arrays):
        # U from left singular vectors of quantized features X, then
        # ortho_grad = create_Z_orthogonal_to_X(grad, U)[0].
        # create_Z_orthogonal_to_X requires M.shape == grad.shape for M @ grad.T, so when
        # min(n, n_features) != n_outputs we truncate U or pad (rare).
        n, d = grad.shape
        U_feat, _, _ = cp.linalg.svd(X, full_matrices=False)
        k = U_feat.shape[1]
        if d <= k:
            U = U_feat[:, :d]
        else:
            pad = cp.random.standard_normal((n, d - k), dtype=grad.dtype)
            U = cp.concatenate([U_feat, pad], axis=1)
        ortho_grad = create_Z_orthogonal_to_X(grad, U)[0]
        return super().build_tree(X, ortho_grad, hess, sample_weight, grad_fn, *val_arrays)


def lstsqr(original, ortho, G):
    D = original - ortho
    numer = cp.trace(cp.matmul(D.T, ortho - G))
    denom = cp.power(D, 2).sum()
    eps = 1e-6
    return (-numer / (denom + eps)).astype(cp.float32)


def weighted_mean(A, B, alpha):
    assert 0 <= alpha <= 1
    return alpha * A + (1 - alpha) * B


def filter_signature(cls: type, kws):
    init = signature(cls.__init__).parameters
    return {k: v for k, v in kws.items() if k in init}


class MDOB(SketchBoost):
    def __init__(
        self,
        loss,
        metric=None,
        ntrees=100,
        lr=0.05,
        min_gain_to_split=0,
        lambda_l2=1,
        gd_steps=1,
        max_depth=6,
        min_data_in_leaf=10,
        colsample=1.0,
        subsample=1.0,
        target_splitter="Single",
        multioutput_sketch=None,
        use_hess=True,
        quantization="Quantile",
        quant_sample=2000000,
        max_bin=256,
        min_data_in_bin=3,
        es=100,
        seed=42,
        verbose=10,
        callbacks=None,
        debug=False,
        ortho_weight=None,
        **kwargs,
    ):
        params = {
            "loss": loss,
            "metric": metric,
            "ntrees": ntrees,
            "lr": lr,
            "min_gain_to_split": min_gain_to_split,
            "lambda_l2": lambda_l2,
            "gd_steps": gd_steps,
            "max_depth": max_depth,
            "min_data_in_leaf": min_data_in_leaf,
            "colsample": colsample,
            "subsample": subsample,
            "target_splitter": target_splitter,
            "multioutput_sketch": multioutput_sketch,
            "use_hess": use_hess,
            "quantization": quantization,
            "quant_sample": quant_sample,
            "max_bin": max_bin,
            "min_data_in_bin": min_data_in_bin,
            "es": es,
            "seed": seed,
            "verbose": verbose,
            "callbacks": callbacks,
            "ortho_weight": ortho_weight,
            "debug": debug,
        }
        params.update(kwargs)
        super().__init__(**filter_signature(SketchBoost, params))
        self.params.update(params)
        self.alpha = []

    def to_device(self):
        if not self._on_device:
            for tree in self.models:
                tree.to_device()
            for tree in self.ortho_branch:
                tree.to_device()
            self.base_score = cp.asarray(self.base_score)
            self._on_device = True

    def _fit_one(self, train, valid, builder: DepthwiseTreeBuilder, build_info, i):
        build_info["num_iter"] = i
        assert train["ensemble"] is not None
        train["grad"], train["hess"] = self.loss(train["target"], train["ensemble"])

        self.callbacks.before_iteration(build_info)
        ortho_builder: OrthoDepthwiseTreeBuilder = build_info["ortho_builder"]
        otree, oleaves, opreds, oval_leaves, oval_preds = ortho_builder.build_tree(
            train["features_gpu"],
            train["grad"],
            train["hess"],
            train["sample_weight"],
            lambda x: self.loss(train["target"], train["ortho_ensemble"] + x),
            *valid["features_gpu"],
        )

        tree, leaves, preds, val_leaves, val_preds = builder.build_tree(
            train["features_gpu"],
            train["grad"],
            train["hess"],
            train["sample_weight"],
            lambda x: self.loss(train["target"], train["ensemble"] + x),
            *valid["features_gpu"],
        )

        self.models.append(tree)
        if i:
            assert not (
                cp.isnan(train["ensemble"]).any()
                and cp.isnan(train["ortho_ensemble"]).any()
                and cp.isnan(train["grad"]).any()
            ), str(i)
        alpha_orth = lstsqr(train["ensemble"], train["ortho_ensemble"], train["target"])
        assert alpha_orth != cp.nan
        alpha_orig = 1 - alpha_orth
        self.alpha.append(alpha_orth)

        train["ensemble"] += preds * alpha_orig
        train["last_tree"] = {"leaves": leaves, "preds": preds}

        train["ortho_ensemble"] += opreds * alpha_orth
        self.ortho_branch.append(otree)
        train["ortho_last_tree"] = {"leaves": oleaves, "preds": opreds}

        for vp, tp, otp in zip(valid["ensemble"], val_preds, oval_preds):
            vp += tp * alpha_orig + otp * alpha_orth

        valid["last_tree"] = {"leaves": val_leaves, "preds": val_preds}
        valid["ortho_last_tree"] = {"leaves": oval_leaves, "preds": oval_preds}
        for vp, tp in zip(valid["ortho_ensemble"], oval_preds):
            vp += tp * alpha_orth

        if self.callbacks.after_iteration(build_info):
            tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
            otree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
            return True

        tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
        otree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
        return False

    def _fit(self, builder, build_info):
        train, valid = build_info["data"]["train"], build_info["data"]["valid"]
        self.callbacks.before_train(build_info)

        for i in range(self.ntrees):
            should_exit = self._fit_one(train, valid, builder, build_info, i)
            if should_exit:
                break

        self.callbacks.after_train(build_info)
        self.base_score = self.base_score.get()

    def _create_build_info(self, mempool, X, X_enc, y, sample_weight, max_bin, borders, eval_sets, eval_enc):
        y = cp.array(y, order="C", dtype=cp.float32)

        if sample_weight is not None:
            sample_weight = cp.array(sample_weight, order="C", dtype=cp.float32)

        X_cp = pad_and_move(X_enc)
        X_val = [cp.array(x, order="C") for x in eval_enc]
        y_val = [cp.array(x["y"], order="C", dtype=cp.float32) for x in eval_sets]
        w_val = [
            None
            if x["sample_weight"] is None
            else cp.array(x["sample_weight"], order="C", dtype=cp.float32)
            for x in eval_sets
        ]

        self.nfeats = X.shape[1]

        builder = DepthwiseTreeBuilder(
            borders,
            use_hess=self.use_hess,
            colsampler=self.colsample,
            subsampler=self.subsample,
            target_splitter=self.target_splitter,
            multioutput_sketch=self.multioutput_sketch,
            gd_steps=self.gd_steps,
            lr=self.lr,
            min_gain_to_split=self.min_gain_to_split,
            min_data_in_leaf=self.min_data_in_leaf,
            lambda_l2=self.lambda_l2,
            max_depth=self.max_depth,
            max_bin=max_bin,
        )

        ortho_builder = OrthoDepthwiseTreeBuilder(
            borders,
            use_hess=True,
            colsampler=self.colsample,
            subsampler=self.subsample,
            target_splitter=self.target_splitter,
            multioutput_sketch=self.multioutput_sketch,
            gd_steps=self.gd_steps,
            lr=self.lr,
            min_gain_to_split=self.min_gain_to_split,
            min_data_in_leaf=self.min_data_in_leaf,
            lambda_l2=self.lambda_l2,
            max_depth=self.max_depth,
            max_bin=max_bin,
        )
        cp.random.seed(self.seed)

        y = self.loss.preprocess_input(y)
        y_val = [self.loss.preprocess_input(x) for x in y_val]
        self.base_score = self.loss.base_score(y)

        ens = cp.empty((y.shape[0], self.base_score.shape[0]), order="C", dtype=cp.float32)
        ens[:] = self.base_score
        ortho_ens = cp.copy(ens)
        val_ens = [cp.empty((x.shape[0], self.base_score.shape[0]), order="C") for x in y_val]
        for ve in val_ens:
            ve[:] = self.base_score

        ortho_val_ens = [cp.copy(val_en) for val_en in val_ens]
        self.models = []
        self.ortho_branch = []

        build_info = {
            "data": {
                "train": {
                    "features_cpu": X,
                    "features_gpu": X_cp,
                    "target": y,
                    "sample_weight": sample_weight,
                    "ensemble": ens,
                    "ortho_ensemble": ortho_ens,
                    "grad": None,
                    "hess": None,
                },
                "valid": {
                    "features_cpu": [dat["X"] for dat in eval_sets],
                    "features_gpu": X_val,
                    "target": y_val,
                    "sample_weight": w_val,
                    "ensemble": val_ens,
                    "ortho_ensemble": ortho_val_ens,
                },
            },
            "borders": borders,
            "model": self,
            "mempool": mempool,
            "builder": builder,
            "ortho_builder": ortho_builder,
        }

        return builder, build_info

    def predict(self, X, batch_size=100000):
        assert batch_size > 0, "Batch size must be a positive integer"

        n_out = self.base_score.shape[0]

        self.to_device()
        if type(X) is cp.ndarray or X.shape[0] <= batch_size:
            is_on_gpu = True
            if type(X) is not cp.ndarray:
                is_on_gpu = False
                X = cp.array(X, order="C", dtype=cp.float32)
            if not (X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]):
                warnings.warn(
                    "X is not contiguous, contiguous copy of array will be created."
                )
                X = cp.ascontiguousarray(X)

            gpu_pred = cp.empty((X.shape[0], n_out), dtype=cp.float32)
            gpu_pred[:] = self.base_score
            ogpu_pred = cp.copy(gpu_pred)
            ogpu_pred[:] = self.base_score
            # Preallocate leaf buffers once to avoid allocator calls in the hot path.
            # If CUDA gets poisoned by a prior kernel, later allocations can fail and
            # mask the true failing operation.
            max_tree_ngroups = max((t.ngroups for t in self.models), default=1)
            max_otree_ngroups = max((t.ngroups for t in self.ortho_branch), default=1)
            leaves_tree = cp.empty((X.shape[0], max_tree_ngroups), dtype=cp.int32)
            leaves_otree = cp.empty((X.shape[0], max_otree_ngroups), dtype=cp.int32)

            def _sync(stage: str) -> None:
                try:
                    cp.cuda.get_current_stream().synchronize()
                except Exception as e:
                    raise RuntimeError(
                        f"MDOB.predict CUDA sync failed at stage='{stage}' "
                        f"(X_shape={getattr(X, 'shape', None)}, "
                        f"n_out={n_out}, max_tree_ngroups={max_tree_ngroups}, "
                        f"max_otree_ngroups={max_otree_ngroups})"
                    ) from e

            _sync("post_alloc_and_init")

            alpha = 0.0
            for tree, otree, alpha in zip(self.models, self.ortho_branch, self.alpha):
                # Strict mode: synchronize before and after each branch call so
                # the failing kernel surfaces at the exact call site.
                _sync("pre_tree_predict")
                try:
                    pred = cp.ascontiguousarray(tree.predict(X, gpu_pred, leaves_tree[:, :tree.ngroups]))
                    _sync("post_tree_predict")
                except Exception as e:
                    raise RuntimeError(
                        f"MDOB tree.predict failed: tree_ngroups={tree.ngroups}, "
                        f"X_shape={getattr(X, 'shape', None)}, pred_shape={gpu_pred.shape}, "
                        f"leaves_shape={leaves_tree[:, :tree.ngroups].shape}"
                    ) from e
                gpu_pred -= alpha * pred
                _sync("post_tree_blend")
                try:
                    pred = cp.ascontiguousarray(otree.predict(X, ogpu_pred, leaves_otree[:, :otree.ngroups]))
                    _sync("post_otree_predict")
                except Exception as e:
                    raise RuntimeError(
                        f"MDOB ortho_tree.predict failed: otree_ngroups={otree.ngroups}, "
                        f"X_shape={getattr(X, 'shape', None)}, pred_shape={ogpu_pred.shape}, "
                        f"leaves_shape={leaves_otree[:, :otree.ngroups].shape}"
                    ) from e
                ogpu_pred -= (1 - alpha) * pred
                _sync("post_otree_blend")

            gpu_pred += alpha * ogpu_pred
            _sync("post_final_blend")
            pred = self.postprocess_fn(gpu_pred)
            if is_on_gpu:
                return pred
            return pred.get()

        raise Exception("too big; not implemented yet")


def lstsqr_per_column(original, ortho, G, eps=1e-6):
    assert original.ndim == 2, "Matrices must be 2D"
    D = original - ortho
    numer = cp.sum(D * (ortho - G), axis=0)
    denom = cp.sum(D * D, axis=0)
    alphas = -numer / (denom + eps)
    return alphas


class MDOBSepAlpha(MDOB):
    def _fit_one(self, train, valid, builder: DepthwiseTreeBuilder, build_info, i):
        build_info["num_iter"] = i
        assert train["ensemble"] is not None
        train["grad"], train["hess"] = self.loss(train["target"], train["ensemble"])
        self.callbacks.before_iteration(build_info)

        ortho_builder: OrthoDepthwiseTreeBuilder = build_info["ortho_builder"]
        otree, oleaves, opreds, oval_leaves, oval_preds = ortho_builder.build_tree(
            train["features_gpu"],
            train["grad"],
            train["hess"],
            train["sample_weight"],
            lambda x: self.loss(train["target"], train["ortho_ensemble"] + x),
            *valid["features_gpu"],
        )

        tree, leaves, preds, val_leaves, val_preds = builder.build_tree(
            train["features_gpu"],
            train["grad"],
            train["hess"],
            train["sample_weight"],
            lambda x: self.loss(train["target"], train["ensemble"] + x),
            *valid["features_gpu"],
        )

        self.models.append(tree)
        train["ensemble"] += preds
        train["last_tree"] = {"leaves": leaves, "preds": preds}

        train["ortho_ensemble"] += opreds
        self.ortho_branch.append(otree)
        train["ortho_last_tree"] = {"leaves": oleaves, "preds": opreds}

        for vp, tp in zip(valid["ensemble"], val_preds):
            vp += tp

        valid["last_tree"] = {"leaves": val_leaves, "preds": val_preds}
        valid["ortho_last_tree"] = {"leaves": oval_leaves, "preds": oval_preds}
        for vp, tp in zip(valid["ortho_ensemble"], oval_preds):
            vp += tp

        if self.callbacks.after_iteration(build_info):
            tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
            otree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
            return True
        tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
        otree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
        return False

    def _fit(self, builder, build_info):
        train, valid = build_info["data"]["train"], build_info["data"]["valid"]
        self.callbacks.before_train(build_info)

        for i in range(self.ntrees):
            should_exit = self._fit_one(train, valid, builder, build_info, i)
            if should_exit:
                break

        self.callbacks.after_train(build_info)
        self.base_score = self.base_score.get()
        self.ortho_branch = self.ortho_branch[: len(self.models)]
        self.alpha = lstsqr_per_column(train["ensemble"], train["ortho_ensemble"], train["target"])

    def predict(self, X, batch_size=100000):
        assert batch_size > 0, "Batch size must be a positive integer"

        ngroups = max((x.ngroups for x in self.models))
        n_out = self.base_score.shape[0]

        self.to_device()
        if type(X) is cp.ndarray or X.shape[0] <= batch_size:
            is_on_gpu = True
            if type(X) is not cp.ndarray:
                is_on_gpu = False
                X = cp.array(X, order="C", dtype=cp.float32)
            if not (X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]):
                warnings.warn(
                    "X is not contiguous, contiguous copy of array will be created."
                )
                X = cp.ascontiguousarray(X)
            oX = cp.copy(X)
            gpu_pred = cp.empty((X.shape[0], n_out), dtype=cp.float32)
            gpu_pred_leaves = cp.empty((X.shape[0], ngroups), dtype=cp.int32)
            gpu_pred[:] = self.base_score

            for tree in self.models:
                tree.predict(X, gpu_pred, gpu_pred_leaves)

            ogpu_pred = cp.empty((oX.shape[0], n_out), dtype=cp.float32)
            ogpu_pred_leaves = cp.empty((oX.shape[0], ngroups), dtype=cp.int32)
            ogpu_pred[:] = self.base_score

            for otree in self.ortho_branch:
                otree.predict(oX, ogpu_pred, ogpu_pred_leaves)

            gpu_pred = (1 - self.alpha) * gpu_pred + self.alpha * ogpu_pred
            pred = self.postprocess_fn(gpu_pred)
            if is_on_gpu:
                return pred
            return pred.get()

        raise Exception("too big; not implemented yet")
