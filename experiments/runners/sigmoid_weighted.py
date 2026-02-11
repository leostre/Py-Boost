from __future__ import annotations

from typing import Dict, Mapping, Sequence

import os
import pickle

import mlflow
import pandas as pd

from experiments.constants import SAMPLE_RATIO, SKETCH_METHODS, LR, DEFAULTS
from experiments.core.experiment import ExperimentContext
from experiments.runners.baselines import BaselinesExperiment


class SigmoidWeightedExperiment(BaselinesExperiment):

    name = "sigmoid_weighted"

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        sketch_methods = [m for m in SKETCH_METHODS if m not in ("svd", None)]
        sketch_outputs = [
            max(1, int(context.n_classes * ratio)) for ratio in SAMPLE_RATIO
        ]
        return {
            "sketch_method": sketch_methods,
            "lr": LR,
            "sketch_outputs": sketch_outputs,
            "subsample": SAMPLE_RATIO,
        }

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

        dataset_name = context.dataset_name

        if results_frames:
            final_results = pd.concat(results_frames, ignore_index=True)
            final_results_file = f"baselines_{self.run_name}_{dataset_name}.csv"
            final_results.to_csv(final_results_file, index=False)

            with mlflow.start_run(run_name=f"{dataset_name}_summary"):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("total_runs", len(results_frames))
                mlflow.log_param("successful_runs", len(results_frames))
                mlflow.log_artifact(final_results_file)

        if all_histories:
            final_histories_file = f"histories_{self.run_name}_{dataset_name}.pkl"
            with open(final_histories_file, "wb") as file:
                pickle.dump(all_histories, file)

            with mlflow.start_run(run_name=f"{dataset_name}_summary"):
                mlflow.log_artifact(final_histories_file)

