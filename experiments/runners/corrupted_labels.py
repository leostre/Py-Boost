from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from experiments.constants import SAMPLE_RATIO, DEFAULTS, LR
from experiments.core.experiment import ExperimentContext
from experiments.runners.fundamentals import FundamentalsExperiment


class LabelCorruptor:
    def __init__(self, corruption_level: float):
        self.corruption_level = corruption_level

    def _flatten(self, y):
        assert len(y.shape) == 2, "Assumed y is binarized"
        shape = y.shape
        y = np.argmax(y, axis=-1)
        return y, shape

    def _onelabel(self, y: np.array):
        y, shape = self._flatten(y)
        counts = np.unique_counts(y)
        preproba = np.zeros((y.shape[0], len(counts.values)))
        preproba[:, counts.values] = counts.counts
        preproba[np.arange(len(y)), y] = 0
        preproba = preproba / preproba.sum(axis=-1)[..., None]
        new_labels = np.array([np.random.choice(counts.values, p=pp) for pp in preproba])
        output = np.zeros(shape)
        output[np.arange(shape[0]), new_labels] = 1
        return output

    def one_label_corruption(self, target):
        y, shape = self._flatten(target)
        counts = np.unique_counts(y).counts
        proba = counts[y]
        proba = proba / proba.sum()
        selection = np.random.choice(
            np.arange(len(y)),
            size=max(1, int(self.corruption_level * len(y))),
            p=proba,
        )
        corrupted_labels = target.copy()
        corrupted_labels[selection] = self._onelabel(target[selection])
        return corrupted_labels

    def multilabel_corruption(self, target):
        assert len(target.shape) == 2, "Assumed y is binarized"
        mask = np.random.random(target.shape) < self.corruption_level
        target = target.astype(bool)
        target[mask] = ~target[mask]
        return target.astype(int)

    @classmethod
    def check_type(cls, y):
        if len(y.shape) == 1:
            return "one"
        if (y.sum(-1) > 1).any():
            return "multi"
        return "one"

    def corrupt(self, target):
        if self.check_type(target) == "one":
            return self.one_label_corruption(target)
        else:
            return self.multilabel_corruption(target)


class CorruptedLabelsExperiment(FundamentalsExperiment):
    name = "corrupted_labels"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        sketch_outputs = [
            max(1, int(context.n_classes * ratio)) for ratio in SAMPLE_RATIO
        ]
        return {
            "sketch_method": ["topk"],
            "lr": LR,
            "sketch_outputs": sketch_outputs,
            "subsample": SAMPLE_RATIO,
            "label_corruption": [0.05, 0.1, 0.2],
        }

    def build_default_params(self, context: ExperimentContext) -> Dict[str, Any]:
        params = dict(DEFAULTS)
        params.update(
            {
                "es": 15,
                "ntrees": 10_000,
                "loss": "multilabel",
            }
        )
        return params

    def before_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        params: Dict[str, Any],
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        corruption_level = params.get("label_corruption", 0.0)
        if corruption_level > 0:
            label_corruptor = LabelCorruptor(corruption_level)
            y_train = label_corruptor.corrupt(y_train)
        return X_train, y_train, X_test, y_test

