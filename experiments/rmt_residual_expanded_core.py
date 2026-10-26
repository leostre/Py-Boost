"""Pure protocol helpers for the expanded H5-3A replication."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def nearest_fraction_budget(
    n_outputs: int,
    numerator: int,
    denominator: int,
) -> int:
    """Round an exact output fraction to nearest integer, with ties to even."""

    if n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    if numerator < 1 or denominator < 1 or numerator > denominator:
        raise ValueError("fraction must satisfy 0 < numerator <= denominator")
    quotient, remainder = divmod(n_outputs * numerator, denominator)
    doubled = 2 * remainder
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def validate_scalar_grid(
    rows: Iterable[Mapping[str, object]],
    datasets: Sequence[str],
    model_seeds: Sequence[int],
    split_seeds: Sequence[int],
    endpoints: Sequence[int],
    features: Sequence[str],
) -> dict[str, int]:
    """Reject incomplete, duplicated, or seed-misaligned scalar result grids."""

    materialized = [dict(row) for row in rows]
    if len(model_seeds) != len(split_seeds):
        raise ValueError("model and split seeds must be aligned")
    split_by_model = dict(zip(model_seeds, split_seeds))
    expected = {
        (str(dataset), int(seed), int(end), str(feature))
        for dataset in datasets
        for seed in model_seeds
        for end in endpoints
        for feature in features
    }
    actual = [
        (
            str(row["dataset"]),
            int(row["model_seed"]),
            int(row["end"]),
            str(row["feature"]),
        )
        for row in materialized
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("scalar grid contains duplicate rows")
    if set(actual) != expected:
        missing = len(expected - set(actual))
        unexpected = len(set(actual) - expected)
        raise ValueError(
            f"scalar grid mismatch: {missing} missing and {unexpected} unexpected rows"
        )
    for row in materialized:
        model_seed = int(row["model_seed"])
        if int(row["split_seed"]) != split_by_model[model_seed]:
            raise ValueError("scalar row uses an unexpected split seed")
    return {
        "datasets": len(datasets),
        "trajectories": len(datasets) * len(model_seeds),
        "rows": len(materialized),
    }


def validate_reused_effective_config(
    payload: Mapping[str, object],
    *,
    model_seed: int,
    split_seed: int,
    structure_budget: int,
    risk_budget: int,
    outputs: int,
    trees: int,
    max_depth: int,
    expected_rows: int,
) -> None:
    """Reject a saved trajectory whose effective configuration drifted."""

    expected_top_level = {
        "case_id": "matched_trajectory",
        "variant": "hasboost_current_topk",
        "scheduler_mode": "never",
        "sketch_outputs": structure_budget,
        "model_seed": model_seed,
        "split_seed": split_seed,
    }
    for key, expected in expected_top_level.items():
        if payload.get(key) != expected:
            raise ValueError(f"reused effective config differs at {key}")

    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("reused effective config has no parameters mapping")
    expected_parameters = {
        "dataset_seed": 42,
        "ntrees": trees,
        "max_depth": max_depth,
        "trajectory": "gh",
        "trajectory_subset_rows": expected_rows,
    }
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise ValueError(f"reused effective config differs at parameters.{key}")
    if list(parameters.get("research_shadow_budgets", [])) != [risk_budget]:
        raise ValueError("reused effective config has an unexpected risk budget")
    if list(payload.get("adaptive_budgets", [])) != [risk_budget, outputs]:
        raise ValueError("reused effective config has unexpected adaptive budgets")
    if not payload.get("dataset_fingerprint"):
        raise ValueError("reused effective config has no dataset fingerprint")
