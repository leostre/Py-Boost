"""Pure analysis helpers for the preregistered H5-1 experiment."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the normalized Frobenius distance between two projectors."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("bases must be two-dimensional and have equal shapes")
    rank = left.shape[1]
    if rank < 1:
        raise ValueError("basis rank must be positive")
    overlap = np.linalg.norm(left.T @ right, ord="fro") ** 2 / rank
    return float(1.0 - np.clip(overlap, 0.0, 1.0))


def batched_subspace_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized version of :func:`subspace_distance`."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("batched bases must have equal (batch, outputs, rank) shapes")
    rank = left.shape[2]
    if rank < 1:
        raise ValueError("basis rank must be positive")
    cross = np.einsum("nor,nos->nrs", left, right)
    overlap = np.sum(cross * cross, axis=(1, 2)) / rank
    return 1.0 - np.clip(overlap, 0.0, 1.0)


def balanced_partitions(rows: int, repeats: int, seed: int) -> np.ndarray:
    """Create deterministic balanced A/B partitions without replacement."""

    if rows < 2 or rows % 2:
        raise ValueError("rows must be an even integer greater than one")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(rows).reshape(2, rows // 2)
                     for _ in range(repeats)])


def paired_distance_samples(
    current_a: np.ndarray,
    current_b: np.ndarray,
    previous_a: np.ndarray,
    previous_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare simultaneous estimator spread with within-half temporal drift."""

    shapes = {np.asarray(value).shape for value in
              (current_a, current_b, previous_a, previous_b)}
    if len(shapes) != 1:
        raise ValueError("all basis batches must have equal shapes")
    noise = batched_subspace_distances(current_a, current_b)
    temporal = (
        batched_subspace_distances(current_a, previous_a)
        + batched_subspace_distances(current_b, previous_b)
    ) / 2.0
    return {"noise": noise, "temporal": temporal, "delta": temporal - noise}


def bootstrap_median_interval(
    values: Sequence[float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile-bootstrap interval for the median."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite vector")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    medians = np.median(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "median": float(np.median(values)),
        "low": float(np.quantile(medians, alpha)),
        "high": float(np.quantile(medians, 1.0 - alpha)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks and explicit constant handling."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("vectors must be one-dimensional and aligned")
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    left_rank = _average_ranks(left[finite])
    right_rank = _average_ranks(right[finite])
    if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize_verdict(
    rows: Iterable[Mapping[str, object]],
    primary_rank: int,
    stable_overlap: float,
    minimum_stable_fraction: float,
    minimum_identifiable_fraction: float,
    minimum_risk_rho: float,
) -> dict[str, object]:
    """Apply the frozen H5-1 gates without tuning them to observed results."""

    primary = [dict(row) for row in rows if int(row["rank"]) == primary_rank]
    if not primary:
        raise ValueError("no rows for the primary rank")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in primary:
        grouped[str(row["dataset"])].append(row)
    datasets = []
    for dataset, dataset_rows in sorted(grouped.items()):
        by_trajectory: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in dataset_rows:
            by_trajectory[int(row["model_seed"])].append(row)
        trajectory_rhos = []
        trajectory_details = []
        for model_seed, trajectory_rows in sorted(by_trajectory.items()):
            rho = spearman_correlation(
                [float(row["median_delta"]) for row in trajectory_rows],
                [float(row["current_risk"]) for row in trajectory_rows],
            )
            if rho is not None:
                trajectory_rhos.append(rho)
            trajectory_details.append({
                "model_seed": model_seed,
                "windows": len(trajectory_rows),
                "risk_rho": rho,
            })
        stable_fraction = float(np.mean([
            float(row["median_overlap"]) >= stable_overlap
            for row in dataset_rows
        ]))
        identifiable_fraction = float(np.mean([
            float(row["delta_ci_low"]) > 0.0
            for row in dataset_rows
        ]))
        median_rho = (float(np.median(trajectory_rhos))
                      if trajectory_rhos else None)
        geometry_pass = (
            stable_fraction >= minimum_stable_fraction
            and identifiable_fraction >= minimum_identifiable_fraction
        )
        risk_link_pass = median_rho is not None and median_rho >= minimum_risk_rho
        datasets.append({
            "dataset": dataset,
            "windows": len(dataset_rows),
            "stable_fraction": stable_fraction,
            "identifiable_fraction": identifiable_fraction,
            "median_within_trajectory_risk_rho": median_rho,
            "geometry_pass": bool(geometry_pass),
            "risk_link_pass": bool(risk_link_pass),
            "trajectories": trajectory_details,
        })
    geometry_pass = bool(datasets) and all(item["geometry_pass"] for item in datasets)
    risk_link_pass = bool(datasets) and all(item["risk_link_pass"] for item in datasets)
    verdict = (
        "confirmed_actionable" if geometry_pass and risk_link_pass
        else "geometric_only" if geometry_pass
        else "rejected"
    )
    return {
        "verdict": verdict,
        "geometry_pass": geometry_pass,
        "risk_link_pass": risk_link_pass,
        "datasets": datasets,
        "gates": {
            "primary_rank": primary_rank,
            "stable_overlap": stable_overlap,
            "minimum_stable_fraction": minimum_stable_fraction,
            "minimum_identifiable_fraction": minimum_identifiable_fraction,
            "minimum_risk_rho": minimum_risk_rho,
        },
    }
