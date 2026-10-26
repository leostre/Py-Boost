"""Pure analysis helpers for the preregistered H5-3 experiment."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.rmt_subspace_reproducibility_core import spearman_correlation


FEATURE_NAMES = (
    "effective_rank_fraction",
    "tail_energy_after_top4",
    "inverse_relative_gap4",
)


def spectral_summaries(eigenvalues: Sequence[float], top_k: int = 4) -> dict[str, float]:
    """Return complexity-oriented scalar summaries of a descending spectrum."""

    values = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    if values.ndim != 1 or not len(values):
        raise ValueError("eigenvalues must be a non-empty vector")
    if not 1 <= top_k < len(values):
        raise ValueError("top_k must leave at least one tail component")
    total = float(values.sum())
    if total <= 0.0:
        return {name: 0.0 for name in FEATURE_NAMES}
    probabilities = values / total
    positive = probabilities > 0.0
    effective_rank = float(np.exp(-np.sum(
        probabilities[positive] * np.log(probabilities[positive])
    )))
    gap = max(0.0, float(values[top_k - 1] - values[top_k]))
    relative_gap = gap / max(float(values[top_k - 1]), 1e-12)
    return {
        "effective_rank_fraction": effective_rank / len(values),
        "tail_energy_after_top4": float(1.0 - probabilities[:top_k].sum()),
        "inverse_relative_gap4": float(1.0 - np.clip(relative_gap, 0.0, 1.0)),
    }


def relative_pair_errors(
    left: Sequence[float], right: Sequence[float], eps: float = 1e-12
) -> np.ndarray:
    """Symmetric relative disagreement for paired scalar estimates."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired estimates must be aligned vectors")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    denominator = np.maximum((np.abs(left) + np.abs(right)) / 2.0, eps)
    return np.abs(left - right) / denominator


def summarize_scalar_verdict(
    rows: Iterable[Mapping[str, object]],
    primary_feature: str,
    secondary_features: Sequence[str],
    maximum_relative_error: float,
    minimum_stable_fraction: float,
    minimum_split_rank_rho: float,
    minimum_risk_rho: float,
) -> dict[str, object]:
    """Apply the frozen H5-3 reproducibility and action-link gates."""

    materialized = [dict(row) for row in rows]
    feature_order = (primary_feature, *secondary_features)
    if not materialized:
        raise ValueError("no scalar rows")
    summaries = []
    for feature in feature_order:
        feature_rows = [row for row in materialized if row["feature"] == feature]
        if not feature_rows:
            raise ValueError(f"no rows for feature {feature}")
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in feature_rows:
            grouped[str(row["dataset"])].append(row)
        datasets = []
        for dataset, dataset_rows in sorted(grouped.items()):
            stable_fraction = float(np.mean([
                float(row["median_relative_error"]) <= maximum_relative_error
                for row in dataset_rows
            ]))
            by_trajectory: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in dataset_rows:
                by_trajectory[int(row["model_seed"])].append(row)
            split_rhos = []
            risk_rhos = []
            trajectories = []
            for model_seed, trajectory_rows in sorted(by_trajectory.items()):
                trajectory_rows.sort(key=lambda row: int(row["end"]))
                repeats = len(trajectory_rows[0]["partition_a_values"])
                trajectory_split_rhos = []
                for repeat in range(repeats):
                    rho = spearman_correlation(
                        [float(row["partition_a_values"][repeat])
                         for row in trajectory_rows],
                        [float(row["partition_b_values"][repeat])
                         for row in trajectory_rows],
                    )
                    if rho is not None:
                        trajectory_split_rhos.append(rho)
                        split_rhos.append(rho)
                risk_rho = spearman_correlation(
                    [float(row["full_value"]) for row in trajectory_rows],
                    [float(row["current_risk"]) for row in trajectory_rows],
                )
                if risk_rho is not None:
                    risk_rhos.append(risk_rho)
                trajectories.append({
                    "model_seed": model_seed,
                    "windows": len(trajectory_rows),
                    "median_split_rank_rho": (
                        float(np.median(trajectory_split_rhos))
                        if trajectory_split_rhos else None
                    ),
                    "risk_rho": risk_rho,
                })
            median_split_rho = float(np.median(split_rhos)) if split_rhos else None
            median_risk_rho = float(np.median(risk_rhos)) if risk_rhos else None
            reproducibility_pass = (
                stable_fraction >= minimum_stable_fraction
                and median_split_rho is not None
                and median_split_rho >= minimum_split_rank_rho
            )
            risk_link_pass = (
                median_risk_rho is not None and median_risk_rho >= minimum_risk_rho
            )
            datasets.append({
                "dataset": dataset,
                "windows": len(dataset_rows),
                "stable_fraction": stable_fraction,
                "median_split_rank_rho": median_split_rho,
                "median_within_trajectory_risk_rho": median_risk_rho,
                "reproducibility_pass": bool(reproducibility_pass),
                "risk_link_pass": bool(risk_link_pass),
                "trajectories": trajectories,
            })
        reproducibility_pass = bool(datasets) and all(
            item["reproducibility_pass"] for item in datasets
        )
        risk_link_pass = bool(datasets) and all(
            item["risk_link_pass"] for item in datasets
        )
        summaries.append({
            "feature": feature,
            "role": "primary" if feature == primary_feature else "secondary",
            "reproducibility_pass": reproducibility_pass,
            "risk_link_pass": risk_link_pass,
            "actionable_pass": reproducibility_pass and risk_link_pass,
            "datasets": datasets,
        })
    primary = next(item for item in summaries if item["role"] == "primary")
    secondary_actionable = [
        item["feature"] for item in summaries
        if item["role"] == "secondary" and item["actionable_pass"]
    ]
    any_reproducible = any(item["reproducibility_pass"] for item in summaries)
    verdict = (
        "confirmed_actionable" if primary["actionable_pass"]
        else "secondary_candidate" if secondary_actionable
        else "descriptive_only" if any_reproducible
        else "rejected"
    )
    return {
        "verdict": verdict,
        "primary_feature": primary_feature,
        "primary_actionable": bool(primary["actionable_pass"]),
        "secondary_actionable": secondary_actionable,
        "features": summaries,
        "gates": {
            "maximum_relative_error": maximum_relative_error,
            "minimum_stable_fraction": minimum_stable_fraction,
            "minimum_split_rank_rho": minimum_split_rank_rho,
            "minimum_risk_rho": minimum_risk_rho,
        },
    }
