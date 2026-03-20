from __future__ import annotations

from typing import Dict, Mapping, Sequence, Any, Tuple

import os
import pickle

import numpy as np
import pandas as pd

from experiments.constants import SAMPLE_RATIO, SKETCH_METHODS, LR, DEFAULTS
from experiments.core.experiment import BaseExperiment, ExperimentContext


class BaselinesExperiment(BaseExperiment):
    name = "baselines"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        sketch_outputs = set(
            max(1, int(context.n_classes * ratio)) for ratio in SAMPLE_RATIO
        )
        sketch_outputs = sorted(list(sketch_outputs))
        return {
            "sketch_method": ["topk"],
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

    def wants_results_dataframe(self) -> bool:
        return True

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

    def after_dataset(
        self,
        context: ExperimentContext,
        all_results: Sequence[Dict[str, Any]],
    ) -> None:
        if not all_results:
            return

        results_frames = []
        all_histories = []

        for result in all_results:
            artifacts = result.get("artifact_files", {})
            results_path = artifacts.get("results")
            histories_path = artifacts.get("histories")

            if results_path and os.path.exists(results_path):
                df = pd.read_csv(results_path)
                results_frames.append(df)

            if histories_path and os.path.exists(histories_path):
                with open(histories_path, "rb") as f:
                    histories = pickle.load(f)
                if isinstance(histories, list):
                    all_histories.extend(histories)

        if results_frames:
            final_results = pd.concat(results_frames, ignore_index=True)
            final_results_file = f"baselines_{context.dataset_name}.csv"
            final_results.to_csv(final_results_file, index=False)

        if all_histories:
            final_histories_file = f"histories_{context.dataset_name}.pkl"
            with open(final_histories_file, "wb") as file:
                pickle.dump(all_histories, file)

