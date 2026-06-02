import cupy as cp

from py_boost.gpu.mdob import filter_signature
from py_boost.gpu.sketch_boost import SketchBoost
from py_boost.gpu.tree import DepthwiseTreeBuilder
from py_boost.gpu.utils import pad_and_move


class ApproxDepthwiseTreeBuilder(DepthwiseTreeBuilder):
    def build_tree(
        self,
        X,
        grad,
        hess,
        sample_weight=None,
        grad_fn=None,
        singular_thr=0.9,
        mode="straight",
        *val_arrays,
    ):
        U, S, Vh = cp.linalg.svd(grad, full_matrices=False)
        cs = cp.cumsum(S)
        n = (cs / cs[-1] < singular_thr).astype(int).sum()
        if mode == "straight":
            S[n:] = 0
        elif mode == "ortho":
            S[:n] = 0
        else:
            raise ValueError("Unknown mode")
        grad = (U * S) @ Vh
        return super().build_tree(X, grad, hess, sample_weight, grad_fn, *val_arrays)


class MDOBSeq(SketchBoost):
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
        singular_thr=1,
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
            "singular_thr": singular_thr,
            "debug": debug,
        }
        params.update(kwargs)
        super().__init__(**filter_signature(SketchBoost, params))
        self.params.update(params)
        self.singular_thr = singular_thr

    def _fit_one(self, train, valid, builder: ApproxDepthwiseTreeBuilder, build_info, i, mode):
        build_info["num_iter"] = i
        assert train["ensemble"] is not None
        train["grad"], train["hess"] = self.loss(train["target"], train["ensemble"])

        self.callbacks.before_iteration(build_info)

        tree, leaves, preds, val_leaves, val_preds = builder.build_tree(
            train["features_gpu"],
            train["grad"],
            train["hess"],
            train["sample_weight"],
            lambda x: self.loss(train["target"], train["ensemble"] + x),
            self.singular_thr,
            mode,
            *valid["features_gpu"],
        )

        self.models.append(tree)
        train["ensemble"] += preds
        train["last_tree"] = {"leaves": leaves, "preds": preds}

        for vp, tp in zip(valid["ensemble"], val_preds):
            vp += tp

        valid["last_tree"] = {"leaves": val_leaves, "preds": val_preds}

        if self.callbacks.after_iteration(build_info):
            tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
            return True
        tree.reformat(nfeats=self.nfeats, debug=self.params["debug"])
        return False

    def _fit(self, builder, build_info):
        train, valid = build_info["data"]["train"], build_info["data"]["valid"]
        self.callbacks.before_train(build_info)
        mode = "straight"
        straight = max(1, int(self.ntrees * self.singular_thr))
        for i in range(straight):
            should_exit = self._fit_one(train, valid, builder, build_info, i, mode=mode)
            if should_exit:
                break
        mode = "ortho"
        for i in range(self.ntrees - len(self.models)):
            should_exit = self._fit_one(train, valid, builder, build_info, i, mode=mode)
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
        builder = ApproxDepthwiseTreeBuilder(
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
        cp.random.seed(self.seed)
        y = self.loss.preprocess_input(y)
        y_val = [self.loss.preprocess_input(x) for x in y_val]
        self.base_score = self.loss.base_score(y)

        ens = cp.empty((y.shape[0], self.base_score.shape[0]), order="C", dtype=cp.float32)
        ens[:] = self.base_score
        val_ens = [cp.empty((x.shape[0], self.base_score.shape[0]), order="C") for x in y_val]
        for ve in val_ens:
            ve[:] = self.base_score

        self.models = []
        build_info = {
            "data": {
                "train": {
                    "features_cpu": X,
                    "features_gpu": X_cp,
                    "target": y,
                    "sample_weight": sample_weight,
                    "ensemble": ens,
                    "grad": None,
                    "hess": None,
                },
                "valid": {
                    "features_cpu": [dat["X"] for dat in eval_sets],
                    "features_gpu": X_val,
                    "target": y_val,
                    "sample_weight": w_val,
                    "ensemble": val_ens,
                },
            },
            "borders": borders,
            "model": self,
            "mempool": mempool,
            "builder": builder,
        }
        return builder, build_info
