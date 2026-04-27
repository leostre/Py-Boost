from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import mlflow
import numpy as np
import cupy as cp

from experiments.constants import DEFAULTS, SAMPLE_RATIO
from experiments.core.experiment import BaseExperiment, ExperimentContext
from experiments.core.metrics import METRICS, ROCAUC_SCORE
from py_boost.gpu.mdob_multibranch import turnoff_branch


class _BaseMDOBExperiment(BaseExperiment):
    @staticmethod
    def _to_numpy_array(value):
        def _empty():
            return np.asarray([], dtype=np.float32)

        if isinstance(value, cp.ndarray):
            try:
                return value.get()
            except Exception:
                return _empty()

        if isinstance(value, (list, tuple)):
            try:
                parts = []
                for x in value:
                    if isinstance(x, cp.ndarray):
                        parts.append(np.ravel(x.get()))
                    else:
                        parts.append(np.ravel(np.asarray(x)))
                if not parts:
                    return _empty()
                return np.concatenate(parts)
            except Exception:
                return _empty()

        try:
            return np.asarray(value)
        except Exception:
            return _empty()

    def _fundamentals_sketch_grid(self, context: ExperimentContext) -> Dict[str, Sequence[Any]]:
        sketch_outputs = sorted(
            {max(1, int(context.n_classes * ratio)) for ratio in SAMPLE_RATIO}
        )
        return {
            "sketch_outputs": sketch_outputs,
            "subsample": SAMPLE_RATIO,
        }

    def build_default_params(self, context: ExperimentContext) -> Dict[str, Any]:
        params = dict(DEFAULTS)
        params.update(
            {
                "es": 15,
                "ntrees": 10000,
                "loss": "multilabel",
                "sketch_method": "topk",
                "subsample": 0.5,
            }
        )
        return params

    def estimate_ensemble_structure(self, model) -> Tuple[float, float]:
        nodes = leaves = 0
        models = getattr(model, "models", None)
        if not models:
            return 0.0, 0.0
        for tree in models:
            nodes += getattr(tree, "max_nodes", 0)
            leaves += getattr(tree, "max_leaves", 0)
        n = len(models)
        if n == 0:
            return 0.0, 0.0
        return nodes / n, leaves / n


class MDOBExperiment(_BaseMDOBExperiment):
    name = "mdob"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        return {
            **self._fundamentals_sketch_grid(context),
            "lr": [0.005, 0.1],
            "ortho_weight": [0.01, 0.05, 0.1, 0.25]
        }

    def after_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        model,
        y_test,
        probas,
        predictions,
        fold_metrics: Dict[str, float],
    ) -> None:
        model.history = {"alpha": self._to_numpy_array(getattr(model, "alpha", []))}
        return None


class MDOBSepAlphaExperiment(_BaseMDOBExperiment):
    name = "mdob_sep_alpha"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        return {
            **self._fundamentals_sketch_grid(context),
            "lr": [0.005, 0.1],
            "ortho_weight": [0.01, 0.05, 0.1, 0.25],
        }

    def after_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        model,
        y_test,
        probas,
        predictions,
        fold_metrics: Dict[str, float],
    ) -> None:
        model.history = {"alpha": self._to_numpy_array(getattr(model, "alpha", []))}
        return None


class MDOBSeqExperiment(_BaseMDOBExperiment):
    name = "mdob_seq"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        return {
            **self._fundamentals_sketch_grid(context),
            "lr": [0.1, 0.005],
            "singular_thr": [0.2, 0.4, 0.6, 0.8, 1.0]
        }


class _BranchingMDOBExperiment(_BaseMDOBExperiment):
    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        return {
            "lr": [0.1, 0.005],
            "eval_size": [0.2],
            "n_branches": [2, 3],
            "branching_threshold": np.linspace(1e-4, 1e-2, 4).tolist(),
        }

    def make_model(self, model_factory, params: Dict[str, Any], context: ExperimentContext):
        params = dict(params)
        self._eval_size = float(params.pop("eval_size", 0.2))
        return model_factory(**params)

    def fit_model(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        context: ExperimentContext,
        fold_idx: int,
    ):
        self._last_x_test = X_test
        model.fit(X_train, y_train, eval_size=self._eval_size)
        return model

    def _compose_total_preds(self, model, X):
        n_classes = model.n_classes
        n_branches = model.n_branches
        total_preds = np.zeros((X.shape[0], n_classes * (n_branches + 1)))

        for i, branch in enumerate(model.branches):
            pred = branch.predict(X)
            total_preds[:, i * n_classes : (i + 1) * n_classes] = pred
        total_preds[:, -n_classes:] = model.root.predict(X)
        return total_preds

    def _predict_with_disabled_branch(self, model, X, disable_branch_idx: int):
        total_preds = self._compose_total_preds(model, X)
        weights = model.head.weights
        bias = model.head.bias
        disabled_w, disabled_b = turnoff_branch(weights, bias, disable_branch_idx)

        total_preds_cp = cp.asarray(total_preds)
        logits = cp.dot(total_preds_cp, disabled_w) + disabled_b
        return model.head._sigmoid(logits).get()

    def after_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        model,
        y_test,
        probas,
        predictions,
        fold_metrics: Dict[str, float],
    ) -> None:
        # Persist branch lengths in history artifact.
        model.history = {
            "lengths": getattr(model, "lengths", {}),
            "root_fit_time": getattr(model, "root_fit_time", None),
            "branch_fit_times": list(getattr(model, "branch_fit_times", [])),
        }

        root_fit_time = getattr(model, "root_fit_time", None)
        if root_fit_time is not None:
            mlflow.log_metric(f"root_fit_time_fold_{fold_idx}", float(root_fit_time), step=fold_idx)

        for i, t in enumerate(getattr(model, "branch_fit_times", [])):
            mlflow.log_metric(f"branch_{i}_fit_time_fold_{fold_idx}", float(t), step=fold_idx)

        # Save metrics with each branch disabled (including root).
        x_test = getattr(self, "_last_x_test", None)
        if x_test is None:
            return None

        for branch_idx in range(model.n_branches + 1):
            try:
                disabled_proba = self._predict_with_disabled_branch(model, x_test, branch_idx)
                disabled_pred = (disabled_proba > 0.5).astype(int)
                for metric_name, metric_fn in METRICS.items():
                    score = metric_fn(y_test, disabled_pred)
                    mlflow.log_metric(
                        f"{metric_name}_branch_disabled_{branch_idx}_fold_{fold_idx}",
                        float(score),
                        step=fold_idx,
                    )
                try:
                    rocauc = ROCAUC_SCORE(y_test, disabled_proba)
                    mlflow.log_metric(
                        f"roc_auc_branch_disabled_{branch_idx}_fold_{fold_idx}",
                        float(rocauc),
                        step=fold_idx,
                    )
                except Exception:
                    pass
            except Exception:
                continue
        return None


class MDOBMultibranchExperiment(_BranchingMDOBExperiment):
    name = "mdob_multibranch"


class MDOBStagedExperiment(_BranchingMDOBExperiment):
    name = "mdob_staged"

    def build_default_params(self, context: ExperimentContext) -> Dict[str, Any]:
        params = dict(super().build_default_params(context))
        params.update(
            {
                "loss": "multilabel",
                "metric": "bce",
                "ntrees": 1000,
                "es": 15,
                "warm_start": True,
                "stop_mode": "norm_grad",
            }
        )
        return params

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        base = dict(super().build_search_space(context))
        base["edge_proportion"] = [0.0001, 0.01]
        return base
