from __future__ import annotations

from typing import Dict, Mapping, Sequence

from inspect import signature

from experiments.constants import SAMPLE_RATIO, LR
from experiments.core.experiment import BaseExperiment, ExperimentContext


class AnaloguesExperiment(BaseExperiment):
    name = "analogues"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence]:
        return {
            "learning_rate": LR,
            "subsample": SAMPLE_RATIO,
        }

    def build_default_params(self, context: ExperimentContext) -> Dict:
        return {}

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
        sig = signature(model.fit)
        if "eval_set" in sig.parameters:
            kwargs = {"eval_set": (X_test, y_test)}
        else:
            kwargs = {}
        model.fit(X_train, y_train, **kwargs)
        return model
