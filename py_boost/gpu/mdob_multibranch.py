from functools import partial
from logging import log
import time
from typing import List

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
from pynndescent import NNDescent
from scipy.linalg import qr
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

from cupyx.scipy.sparse.linalg import svds
from py_boost.gpu.base import Ensemble
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
from py_boost.gpu.accumulation.history_callback import FullGHistorySampling
from py_boost.gpu.utils import validate_input


class MultilabelLogisticRegressionGPU:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=None, C=1.0, tol=1e-4, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.C = C
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = cp.clip(z, -500, 500)
        return 1.0 / (1.0 + cp.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred, epsilon=1e-15):
        y_pred = cp.clip(y_pred, epsilon, 1 - epsilon)
        return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        cp.random.seed(42)
        self.weights = cp.random.randn(n_features, n_labels) * 0.01
        self.bias = cp.zeros(n_labels)

        if self.batch_size is None:
            self.batch_size = n_samples
        else:
            self.batch_size = min(self.batch_size, n_samples)

        prev_loss = float("inf")
        for iteration in range(self.n_iterations):
            indices = cp.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                logits = cp.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(logits)
                loss = self._binary_cross_entropy(y_batch, y_pred)
                reg_loss = 0.5 * cp.sum(self.weights ** 2) / self.C
                _ = loss + reg_loss
                error = y_pred - y_batch
                dw = cp.dot(X_batch.T, error) / X_batch.shape[0]
                db = cp.mean(error, axis=0)
                dw += self.weights / (self.C * X_batch.shape[0])
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = float(self._binary_cross_entropy(y_val, val_pred))

            if self.verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                if val_loss:
                    print(f"Iteration {iteration}: train loss = {avg_loss:.6f}, val loss = {val_loss:.6f}")
                else:
                    print(f"Iteration {iteration}: train loss = {avg_loss:.6f}")

            if abs(prev_loss - avg_loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            prev_loss = avg_loss

        return self

    def predict_proba(self, X):
        logits = cp.dot(X, self.weights) + self.bias
        return self._sigmoid(logits)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(cp.float32)

    def score(self, X, y, threshold=0.5):
        y_pred = self.predict(X, threshold)
        return cp.mean(cp.all(y_pred == y, axis=1))

    def get_params(self):
        return {"weights": self.weights.copy(), "bias": self.bias.copy(), "loss_history": self.loss_history.copy()}


class DataClusterMDOB:
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
        quantization="Quantile",
        quant_sample=2000000,
        max_bin=256,
        min_data_in_bin=3,
        es=100,
        seed=42,
        verbose=10,
        sketch_outputs=1,
        sketch_method="proj",
        use_hess=False,
        callbacks=None,
        sketch_params=None,
        n_branches=1,
        warm_start=False,
        smoothing_alpha=0.9,
        branching_threshold=0.1,
        stop_mode="norm_grad",
        save_GH_ckpt=None,
    ):
        if sketch_params is None:
            sketch_params = {}

        params = dict(
            loss=loss,
            metric=metric,
            ntrees=ntrees,
            lr=lr,
            min_gain_to_split=min_gain_to_split,
            lambda_l2=lambda_l2,
            gd_steps=gd_steps,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            colsample=colsample,
            subsample=subsample,
            quantization=quantization,
            quant_sample=quant_sample,
            max_bin=max_bin,
            min_data_in_bin=min_data_in_bin,
            use_hess=use_hess,
            es=es,
            seed=seed,
            verbose=verbose,
            callbacks=callbacks,
            smoothing_alpha=smoothing_alpha,
        )
        self._base_model = partial(
            HistoryBasedBoostingModel,
            multioutput_sketch=partial(
                FullGHistorySampling, branching_threshold=branching_threshold, stop_mode=stop_mode
            ),
        )
        self.root = self._base_model(**params)
        self.n_branches = n_branches
        self.warm_start = warm_start
        self.branches = [
            self._base_model(**(params | {"ntrees": ntrees // n_branches, "branching_threshold": -1}))
            for _ in range(n_branches)
        ]
        self.head = MultilabelLogisticRegressionGPU()
        self.save_GH_ckpt = save_GH_ckpt
        self.root_fit_time = None
        self.branch_fit_times = []

    def fit(self, X, y, sample_weight=None, eval_size=0.0):
        log(0, "prediction started")
        tr_idx, eval_idx = train_test_split(np.arange(X.shape[0]), test_size=eval_size, random_state=1)
        new_eval_set = {
            "X": X[eval_idx],
            "y": y[eval_idx],
            "sample_weight": sample_weight[eval_idx] if sample_weight is not None else None,
        }
        X = X[tr_idx]
        y = y[tr_idx]
        sample_weight = sample_weight[tr_idx] if sample_weight is not None else None
        self.root._infer_params()

        X, y, sample_weight, eval_sets = validate_input(X, y, sample_weight, [new_eval_set])
        self.n_classes = y.shape[1]
        mempool = cp.cuda.MemoryPool()
        total_preds = cp.zeros((X.shape[0], y.shape[1] * (self.n_branches + 1)))
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            X_enc, max_bin, borders, eval_enc = self.root.quantize(X, eval_sets)
            builder, build_info = self.root._create_build_info(
                mempool, X, X_enc, y, sample_weight, max_bin, borders, eval_sets, eval_enc
            )
            build_info["save_GH_ckpt"] = self.save_GH_ckpt
            root_fit_start = time.perf_counter()
            self.root._fit(builder, build_info)
            self.root_fit_time = time.perf_counter() - root_fit_start
            total_preds[:, -self.n_classes:] = build_info["data"]["train"]["ensemble"]
            self.branch_fit_times = []
            for i, sample_idx in enumerate(self._analize_root(self.n_branches, self.root)):
                sample_idx, eval_idx = train_test_split(sample_idx, test_size=eval_size, random_state=1)
                X_i = X[sample_idx]
                y_i = y[sample_idx]
                sample_weight_i = sample_weight[sample_idx] if sample_weight is not None else None
                eval_sets_i = [
                    {
                        "X": X[eval_idx],
                        "y": y[eval_idx],
                        "sample_weight": sample_weight[eval_idx] if sample_weight is not None else None,
                    }
                ]
                X_enc_i, max_bin, borders, eval_enc_i = self.root.quantize(X, eval_sets_i)
                X_enc_i = X_enc[sample_idx]
                branch = self.branches[i]
                branch._infer_params()
                builder_i, build_info_i = branch._create_build_info(
                    mempool, X_i, X_enc_i, y_i, sample_weight_i, max_bin, borders, eval_sets_i, eval_enc_i
                )
                if self.warm_start:
                    build_info_i["data"]["train"]["ensemble"][:, :] = build_info["data"]["train"]["ensemble"][
                        sample_idx, :
                    ]
                    for j in range(len(build_info["data"]["valid"]["ensemble"])):
                        build_info_i["data"]["valid"]["ensemble"][j][:, :] = build_info["data"]["valid"]["ensemble"][
                            j
                        ][eval_idx, :]
                branch_fit_start = time.perf_counter()
                branch._fit(builder_i, build_info_i)
                self.branch_fit_times.append(time.perf_counter() - branch_fit_start)
                total_preds[sample_idx, self.n_classes * i : self.n_classes * (i + 1)] = build_info_i["data"][
                    "train"
                ]["ensemble"]
        self.total_preds = total_preds
        self.train_y = y
        self.head.fit(total_preds, cp.asarray(y))
        mempool.free_all_blocks()
        return self

    def _analize_root(self, n, ensemble: Ensemble) -> List[List[int]]:
        U, S, Vh = cp.linalg.svd(self.root.sketch.full_grad_hist, full_matrices=False)
        n_cls = max(1, U.shape[1] // 2)
        U, S = U[:, :n_cls], S[:n_cls]
        to_cluster = (U * S).get()
        init_clusterer = AgglomerativeClustering(distance_threshold=None, n_clusters=self.n_branches)
        init_clusterer.fit(to_cluster)
        idx = np.arange(U.shape[0])
        labels = init_clusterer.labels_.flatten()
        return [idx[labels == i] for i in range(self.n_branches)]

    def predict(self, X):
        total_preds = np.zeros((X.shape[0], self.n_classes * (self.n_branches + 1)))
        for i, branch in enumerate(self.branches):
            pred = branch.predict(X)
            total_preds[:, i * self.n_classes : (i + 1) * self.n_classes] = pred
        total_preds[:, -self.n_classes:] = self.root.predict(X)
        final_preds = self.head.predict_proba(cp.asarray(total_preds)).get()
        return final_preds

    @property
    def lengths(self):
        return {"root": len(self.root.models), "branches": [len(branch.models) for branch in self.branches]}


def turnoff_branch(weights, bias, branch_indices):
    if not isinstance(branch_indices, (tuple, list)):
        branch_indices = [branch_indices]
    ncls = weights.shape[-1]
    w = cp.asarray(weights).copy()
    b = cp.asarray(bias).copy()
    for i in branch_indices:
        w[i * ncls : (i + 1) * ncls] = 0
        b[i * ncls : (i + 1) * ncls] = 0
    return w, b


def cluster_qr_gpu(vectors):
    k = vectors.shape[1]
    vectors_cpu = cp.asnumpy(vectors)
    _, _, piv = qr(vectors_cpu.T, pivoting=True, mode="economic")
    piv = cp.asarray(piv)
    selected = vectors[piv[:k], :].T
    ut, _, v = cp.linalg.svd(selected, full_matrices=False)
    transformed = vectors @ (ut @ v.T)
    labels = cp.argmax(cp.abs(transformed), axis=1)
    return cp.asnumpy(labels)


def spectral_nn_clustering_gpu(X, n_clusters, n_neighbors=None, edge_proportion=0.005, metric="euclidean"):
    if not n_neighbors:
        n_neighbors = int(X.shape[0] * np.sqrt(edge_proportion))
    index = NNDescent(X, n_neighbors=n_neighbors, metric=metric)
    indices, distances = index.query(X, k=n_neighbors)
    row = np.repeat(np.arange(X.shape[0]), n_neighbors)
    col = indices.flatten()
    data = np.ones(X.shape[0] * n_neighbors)
    adjacency = csr_matrix((data, (row, col)), shape=(X.shape[0], X.shape[0]))
    adjacency = (adjacency + adjacency.T) / 2
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = diags(np.minimum(1 / np.sqrt(degrees + 1e-10), 1e8))
    L = (D @ adjacency) @ D
    L_cp = cpx_sparse.csr_matrix(L)
    eigvecs, eigvals, _ = svds(L_cp, k=n_clusters)
    labels = cluster_qr_gpu(eigvecs)
    return labels


class RealMDOB_staged:
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
        quantization="Quantile",
        quant_sample=2000000,
        max_bin=256,
        min_data_in_bin=3,
        es=100,
        seed=42,
        verbose=10,
        sketch_outputs=1,
        sketch_method="proj",
        use_hess=False,
        callbacks=None,
        sketch_params=None,
        n_branches=1,
        warm_start=False,
        smoothing_alpha=0.9,
        branching_threshold=0.1,
        stop_mode="norm_grad",
        save_GH_ckpt=None,
        edge_proportion=0.01,
    ):
        if sketch_params is None:
            sketch_params = {}

        params = dict(
            loss=loss,
            metric=metric,
            ntrees=ntrees,
            lr=lr,
            min_gain_to_split=min_gain_to_split,
            lambda_l2=lambda_l2,
            gd_steps=gd_steps,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            colsample=colsample,
            subsample=subsample,
            quantization=quantization,
            quant_sample=quant_sample,
            max_bin=max_bin,
            min_data_in_bin=min_data_in_bin,
            use_hess=use_hess,
            es=es,
            seed=seed,
            verbose=verbose,
            callbacks=callbacks,
            smoothing_alpha=smoothing_alpha,
        )
        self._base_model = partial(
            HistoryBasedBoostingModel,
            multioutput_sketch=partial(
                FullGHistorySampling,
                branching_threshold=branching_threshold,
                stop_mode=stop_mode,
                **params,
            ),
        )
        self.root = self._base_model(**params)
        self.params = params
        self.n_branches = n_branches
        self.tree_limit = ntrees
        self.warm_start = warm_start
        self.branches = []
        self.head = MultilabelLogisticRegressionGPU()
        self.save_GH_ckpt = save_GH_ckpt
        self.clustering_func = partial(
            spectral_nn_clustering_gpu, n_clusters=n_branches, edge_proportion=edge_proportion
        )
        self.root_fit_time = None
        self.branch_fit_times = []
        self.root_build_info = None
        self.root_borders = None
        self.root_max_bin = None
        self.root_X_enc = None
        self.root_eval_enc = None

    def fit_root(self, X, y, sample_weight=None, eval_size=0.0):
        log(0, "root fitting started")
        tr_idx, eval_idx = train_test_split(np.arange(X.shape[0]), test_size=eval_size, random_state=1)
        new_eval_set = {
            "X": X[eval_idx],
            "y": y[eval_idx],
            "sample_weight": sample_weight[eval_idx] if sample_weight is not None else None,
        }
        X_train = X[tr_idx]
        y_train = y[tr_idx]
        sample_weight_train = sample_weight[tr_idx] if sample_weight is not None else None
        self.root._infer_params()

        X_train, y_train, sample_weight_train, eval_sets = validate_input(
            X_train, y_train, sample_weight_train, [new_eval_set]
        )
        self.n_classes = y_train.shape[1]
        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            X_enc, max_bin, borders, eval_enc = self.root.quantize(X_train, eval_sets)
            builder, build_info = self.root._create_build_info(
                mempool, X_train, X_enc, y_train, sample_weight_train, max_bin, borders, eval_sets, eval_enc
            )
            build_info["save_GH_ckpt"] = self.save_GH_ckpt
            root_fit_start = time.perf_counter()
            self.root._fit(builder, build_info)
            self.root_fit_time = time.perf_counter() - root_fit_start
            self.root_build_info = build_info
            self.root_borders = borders
            self.root_max_bin = max_bin
            self.root_X_enc = X_enc
            self.root_eval_enc = eval_enc
            self.root_train_idx = tr_idx
            self.root_eval_idx = eval_idx

        mempool.free_all_blocks()
        log(0, "root fitting completed")
        return self

    def fit_branches(self, X, y, sample_weight=None, eval_size=0.0):
        if self.root_build_info is None:
            raise ValueError("Root must be fitted before fitting branches. Call fit_root() first.")

        log(0, "branches fitting started")
        tr_idx = self.root_train_idx
        X_train = X[tr_idx]
        y_train = y[tr_idx]
        sample_weight_train = sample_weight[tr_idx] if sample_weight is not None else None
        total_preds = cp.zeros((X_train.shape[0], y_train.shape[1] * (self.n_branches + 1)))
        total_preds[:, -self.n_classes:] = self.root_build_info["data"]["train"]["ensemble"]

        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            self.branch_fit_times = []
            for i, sample_idx in enumerate(self._analize_root(self.n_branches, self.root)):
                log(0, f"branch {i} training")
                branch_sample_idx, branch_eval_idx = train_test_split(sample_idx, test_size=eval_size, random_state=1)
                branch = self._base_model(
                    **(
                        self.params
                        | dict(
                            ntrees=(self.tree_limit - len(self.root.models)) // self.n_branches,
                            branching_threshold=-1,
                        )
                    )
                )
                branch._infer_params()
                self.branches.append(branch)
                X_i = X_train[branch_sample_idx]
                y_i = y_train[branch_sample_idx]
                sample_weight_i = sample_weight_train[branch_sample_idx] if sample_weight_train is not None else None
                eval_sets_i = [
                    {
                        "X": X_train[branch_eval_idx],
                        "y": y_train[branch_eval_idx],
                        "sample_weight": sample_weight_train[branch_eval_idx]
                        if sample_weight_train is not None
                        else None,
                    }
                ]
                X_enc_i, max_bin, borders, eval_enc_i = branch.quantize(X_i, eval_sets_i)
                builder_i, build_info_i = branch._create_build_info(
                    mempool, X_i, X_enc_i, y_i, sample_weight_i, max_bin, borders, eval_sets_i, eval_enc_i
                )
                if self.warm_start:
                    build_info_i["data"]["train"]["ensemble"][:, :] = self.root_build_info["data"]["train"][
                        "ensemble"
                    ][branch_sample_idx, :]
                    for j in range(len(self.root_build_info["data"]["valid"]["ensemble"])):
                        build_info_i["data"]["valid"]["ensemble"][j][:, :] = self.root_build_info["data"]["valid"][
                            "ensemble"
                        ][j][branch_eval_idx, :]
                branch_fit_start = time.perf_counter()
                branch._fit(builder_i, build_info_i)
                self.branch_fit_times.append(time.perf_counter() - branch_fit_start)
                total_preds[
                    branch_sample_idx, self.n_classes * i : self.n_classes * (i + 1)
                ] = build_info_i["data"]["train"]["ensemble"]

        self.total_preds = total_preds
        self.train_y = y_train
        self.head.fit(total_preds, cp.asarray(y_train))
        mempool.free_all_blocks()
        return self

    def fit(self, X, y, sample_weight=None, eval_size=0.0):
        self.fit_root(X, y, sample_weight, eval_size)
        self.fit_branches(X, y, sample_weight, eval_size)
        return self

    def save_root(self, path):
        import joblib

        save_dict = {
            "root_model": self.root,
            "root_build_info": self.root_build_info,
            "root_borders": self.root_borders,
            "root_max_bin": self.root_max_bin,
            "root_train_idx": self.root_train_idx,
            "root_eval_idx": self.root_eval_idx,
            "n_classes": self.n_classes,
        }
        if self.root_X_enc is not None:
            save_dict["root_X_enc"] = (
                self.root_X_enc.get() if hasattr(self.root_X_enc, "get") else self.root_X_enc
            )
        if self.root_eval_enc is not None:
            save_dict["root_eval_enc"] = [
                enc.get() if hasattr(enc, "get") else enc for enc in self.root_eval_enc
            ]
        joblib.dump(save_dict, path)

    def load_root(self, path):
        import joblib

        save_dict = joblib.load(path)
        self.root = save_dict["root_model"]
        self.root_build_info = save_dict["root_build_info"]
        self.root_borders = save_dict["root_borders"]
        self.root_max_bin = save_dict["root_max_bin"]
        self.root_train_idx = save_dict["root_train_idx"]
        self.root_eval_idx = save_dict["root_eval_idx"]
        self.n_classes = save_dict["n_classes"]
        if "root_X_enc" in save_dict:
            self.root_X_enc = cp.asarray(save_dict["root_X_enc"])
        if "root_eval_enc" in save_dict:
            self.root_eval_enc = [cp.asarray(enc) for enc in save_dict["root_eval_enc"]]
        return self

    def _analize_root(self, n, ensemble: Ensemble) -> List[List[int]]:
        # HistoryBasedBoostingModel stores the history callback instance in
        # `multioutput_sketch` (not `sketch`).
        sketch_obj = getattr(self.root, "multioutput_sketch", None)
        full_grad_hist = getattr(sketch_obj, "full_grad_hist", None)
        if full_grad_hist is None:
            raise AttributeError(
                "Root model does not provide `full_grad_hist` in `multioutput_sketch`; "
                "cannot cluster branches."
            )
        full_grad_hist_np = (
            full_grad_hist.get() if hasattr(full_grad_hist, "get") else np.asarray(full_grad_hist)
        )
        labels = self.clustering_func(full_grad_hist_np).flatten()
        idx = np.arange(len(labels))
        return [idx[labels == i] for i in range(self.n_branches)]

    def predict(self, X):
        total_preds = np.zeros((X.shape[0], self.n_classes * (self.n_branches + 1)))
        for i, branch in enumerate(self.branches):
            pred = branch.predict(X)
            total_preds[:, i * self.n_classes : (i + 1) * self.n_classes] = pred
        total_preds[:, -self.n_classes:] = self.root.predict(X)
        final_preds = self.head.predict_proba(cp.asarray(total_preds)).get()
        return final_preds

    @property
    def lengths(self):
        return {"root": len(self.root.models), "branches": [len(branch.models) for branch in self.branches]}
