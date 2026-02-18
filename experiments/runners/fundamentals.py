from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple, Any

from experiments.constants import SAMPLE_RATIO, DEFAULTS
from experiments.core.experiment import BaseExperiment, ExperimentContext


class FundamentalsExperiment(BaseExperiment):
    name = "fundamentals"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        sketch_outputs = set(
            max(1, int(context.n_classes * ratio)) for ratio in SAMPLE_RATIO
        )
        sketch_outputs = sorted(list(sketch_outputs))
        return {
            "sketch_method": ["topk", "rand"],
            "lr": [0.1, 0.005],
            "sketch_outputs": sketch_outputs,
            "subsample": SAMPLE_RATIO,
        }

    def build_default_params(self, context: ExperimentContext) -> Dict[str, Any]:
        params = dict(DEFAULTS)
        params.update(
            {
                "es": 15,
                "ntrees": 100_000,
                "loss": "multilabel",
            }
        )
        return params

    def methods_to_time(self, context: ExperimentContext) -> Sequence[str]:
        return ["get_weights", "get_indexers"]

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

