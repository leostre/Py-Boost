"""Pure transformations and verdicts for H5-3A/H5-3B residual signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.rmt_subspace_reproducibility_core import spearman_correlation


def lin_concordance(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Lin's concordance coefficient for two aligned finite vectors."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("vectors must be one-dimensional and aligned")
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    left = left[finite]
    right = right[finite]
    covariance = float(np.mean((left - left.mean()) * (right - right.mean())))
    denominator = float(
        np.var(left) + np.var(right) + (left.mean() - right.mean()) ** 2
    )
    if denominator <= np.finfo(float).eps:
        return None
    return float(2.0 * covariance / denominator)


def _materialize_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("no scalar rows")
    keys = [
        (str(row["dataset"]), int(row["model_seed"]), int(row["end"]),
         str(row["feature"]))
        for row in materialized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("scalar rows are not unique")
    return materialized


def _signal_row(
    row: Mapping[str, object],
    expected_feature: float,
    expected_risk: float,
    partition_a: Sequence[float],
    partition_b: Sequence[float],
) -> dict[str, object]:
    return {
        "dataset": str(row["dataset"]),
        "model_seed": int(row["model_seed"]),
        "split_seed": int(row["split_seed"]),
        "end": int(row["end"]),
        "feature": str(row["feature"]),
        "partition_repeats": int(row["partition_repeats"]),
        "expected_feature": float(expected_feature),
        "expected_risk": float(expected_risk),
        "feature_signal": float(row["full_value"]) - float(expected_feature),
        "risk_signal": float(row["current_risk"]) - float(expected_risk),
        "partition_a_signal": np.asarray(partition_a, dtype=float).tolist(),
        "partition_b_signal": np.asarray(partition_b, dtype=float).tolist(),
    }


def cross_trajectory_residual_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Subtract a leave-one-trajectory-out curve at the same endpoint."""

    materialized = _materialize_rows(rows)
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in materialized:
        grouped[(str(row["dataset"]), str(row["feature"]), int(row["end"]))].append(row)
    transformed = []
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            raise ValueError(f"leave-one-trajectory-out group is too small: {key}")
        seeds = [int(row["model_seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate trajectory in group: {key}")
        for row in group:
            others = [candidate for candidate in group if candidate is not row]
            expected_feature = float(np.mean([
                float(candidate["full_value"]) for candidate in others
            ]))
            expected_risk = float(np.mean([
                float(candidate["current_risk"]) for candidate in others
            ]))
            partition_a = np.asarray(row["partition_a_values"], dtype=float)
            partition_b = np.asarray(row["partition_b_values"], dtype=float)
            if partition_a.shape != partition_b.shape or partition_a.ndim != 1:
                raise ValueError("partition estimates must be aligned vectors")
            transformed.append(_signal_row(
                row,
                expected_feature,
                expected_risk,
                partition_a - expected_feature,
                partition_b - expected_feature,
            ))
    return sorted(
        transformed,
        key=lambda row: (row["dataset"], row["model_seed"], row["end"], row["feature"]),
    )


def local_linear_innovation_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return causal second differences relative to a local linear forecast."""

    materialized = _materialize_rows(rows)
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in materialized:
        grouped[(str(row["dataset"]), int(row["model_seed"]),
                 str(row["feature"]))].append(row)
    transformed = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["end"]))
        if len(group) < 3:
            raise ValueError(f"trajectory needs at least three endpoints: {key}")
        steps = np.diff([int(row["end"]) for row in group])
        if np.any(steps != steps[0]):
            raise ValueError(f"trajectory endpoints are not equally spaced: {key}")
        for previous_two, previous, current in zip(group, group[1:], group[2:]):
            expected_feature = (
                2.0 * float(previous["full_value"])
                - float(previous_two["full_value"])
            )
            expected_risk = (
                2.0 * float(previous["current_risk"])
                - float(previous_two["current_risk"])
            )
            previous_two_a = np.asarray(previous_two["partition_a_values"], dtype=float)
            previous_a = np.asarray(previous["partition_a_values"], dtype=float)
            current_a = np.asarray(current["partition_a_values"], dtype=float)
            previous_two_b = np.asarray(previous_two["partition_b_values"], dtype=float)
            previous_b = np.asarray(previous["partition_b_values"], dtype=float)
            current_b = np.asarray(current["partition_b_values"], dtype=float)
            if len({value.shape for value in (
                previous_two_a, previous_a, current_a,
                previous_two_b, previous_b, current_b,
            )}) != 1:
                raise ValueError("partition histories must have equal shapes")
            transformed.append(_signal_row(
                current,
                expected_feature,
                expected_risk,
                current_a - 2.0 * previous_a + previous_two_a,
                current_b - 2.0 * previous_b + previous_two_b,
            ))
    return sorted(
        transformed,
        key=lambda row: (row["dataset"], row["model_seed"], row["end"], row["feature"]),
    )


def summarize_residual_verdict(
    rows: Iterable[Mapping[str, object]],
    primary_feature: str,
    secondary_features: Sequence[str],
    minimum_ccc: float,
    minimum_sign_agreement: float,
    minimum_risk_rho: float,
) -> dict[str, object]:
    """Apply the preregistered reproducibility and action-link gates."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("no residual rows")
    summaries = []
    for feature in (primary_feature, *secondary_features):
        feature_rows = [row for row in materialized if row["feature"] == feature]
        if not feature_rows:
            raise ValueError(f"no rows for feature {feature}")
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in feature_rows:
            grouped[str(row["dataset"])].append(row)
        datasets = []
        for dataset, dataset_rows in sorted(grouped.items()):
            trajectories: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in dataset_rows:
                trajectories[int(row["model_seed"])].append(row)
            concordances = []
            risk_rhos = []
            sign_matches = []
            trajectory_details = []
            for model_seed, trajectory_rows in sorted(trajectories.items()):
                trajectory_rows.sort(key=lambda row: int(row["end"]))
                repeat_counts = {int(row["partition_repeats"]) for row in trajectory_rows}
                if len(repeat_counts) != 1:
                    raise ValueError("partition repeat count changes within a trajectory")
                trajectory_cccs = []
                for repeat in range(repeat_counts.pop()):
                    left = np.asarray([
                        float(row["partition_a_signal"][repeat])
                        for row in trajectory_rows
                    ])
                    right = np.asarray([
                        float(row["partition_b_signal"][repeat])
                        for row in trajectory_rows
                    ])
                    ccc = lin_concordance(left, right)
                    if ccc is not None:
                        trajectory_cccs.append(ccc)
                        concordances.append(ccc)
                    sign_matches.extend((np.sign(left) == np.sign(right)).tolist())
                risk_rho = spearman_correlation(
                    [float(row["feature_signal"]) for row in trajectory_rows],
                    [float(row["risk_signal"]) for row in trajectory_rows],
                )
                if risk_rho is not None:
                    risk_rhos.append(risk_rho)
                trajectory_details.append({
                    "model_seed": model_seed,
                    "windows": len(trajectory_rows),
                    "median_ccc": (
                        float(np.median(trajectory_cccs)) if trajectory_cccs else None
                    ),
                    "risk_rho": risk_rho,
                })
            median_ccc = float(np.median(concordances)) if concordances else None
            sign_agreement = float(np.mean(sign_matches)) if sign_matches else None
            median_risk_rho = float(np.median(risk_rhos)) if risk_rhos else None
            reproducibility_pass = (
                median_ccc is not None
                and median_ccc >= minimum_ccc
                and sign_agreement is not None
                and sign_agreement >= minimum_sign_agreement
            )
            risk_link_pass = (
                median_risk_rho is not None and median_risk_rho >= minimum_risk_rho
            )
            datasets.append({
                "dataset": dataset,
                "windows": len(dataset_rows),
                "median_ccc": median_ccc,
                "sign_agreement": sign_agreement,
                "median_within_trajectory_risk_rho": median_risk_rho,
                "reproducibility_pass": bool(reproducibility_pass),
                "risk_link_pass": bool(risk_link_pass),
                "trajectories": trajectory_details,
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
            "minimum_ccc": minimum_ccc,
            "minimum_sign_agreement": minimum_sign_agreement,
            "minimum_risk_rho": minimum_risk_rho,
        },
    }
