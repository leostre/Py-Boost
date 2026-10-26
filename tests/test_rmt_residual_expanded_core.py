"""Protocol tests for the expanded H5-3A replication."""

import pytest

from experiments.rmt_residual_expanded_core import (
    nearest_fraction_budget,
    validate_reused_effective_config,
    validate_scalar_grid,
)


def test_nearest_fraction_budget_preserves_registered_output_ratios():
    outputs = (100, 101, 206, 355)

    assert [nearest_fraction_budget(value, 1, 4) for value in outputs] == [
        25, 25, 52, 89
    ]
    assert [nearest_fraction_budget(value, 1, 2) for value in outputs] == [
        50, 50, 103, 178
    ]


def test_validate_scalar_grid_accepts_complete_seed_aligned_cartesian_grid():
    rows = [
        {
            "dataset": dataset,
            "model_seed": model_seed,
            "split_seed": split_seed,
            "end": end,
            "feature": feature,
        }
        for dataset in ("a", "b")
        for model_seed, split_seed in ((1, 11), (2, 12))
        for end in (20, 24)
        for feature in ("x", "y")
    ]

    summary = validate_scalar_grid(
        rows,
        datasets=("a", "b"),
        model_seeds=(1, 2),
        split_seeds=(11, 12),
        endpoints=(20, 24),
        features=("x", "y"),
    )

    assert summary == {"datasets": 2, "trajectories": 4, "rows": 16}


def test_validate_scalar_grid_rejects_missing_or_misaligned_rows():
    row = {
        "dataset": "a",
        "model_seed": 1,
        "split_seed": 99,
        "end": 20,
        "feature": "x",
    }
    with pytest.raises(ValueError, match="split seed"):
        validate_scalar_grid([row], ("a",), (1,), (11,), (20,), ("x",))

    with pytest.raises(ValueError, match="grid mismatch"):
        validate_scalar_grid([], ("a",), (1,), (11,), (20,), ("x",))


def _effective_config():
    return {
        "case_id": "matched_trajectory",
        "variant": "hasboost_current_topk",
        "scheduler_mode": "never",
        "sketch_outputs": 25,
        "adaptive_budgets": [50, 100],
        "model_seed": 204,
        "split_seed": 132,
        "dataset_fingerprint": "sha256-like-value",
        "parameters": {
            "datasets": ["helena"],
            "dataset_seed": 42,
            "ntrees": 100,
            "max_depth": 5,
            "trajectory": "gh",
            "trajectory_subset_rows": 1024,
            "research_shadow_budgets": [50],
        },
    }


def test_validate_reused_effective_config_accepts_frozen_source():
    validate_reused_effective_config(
        _effective_config(),
        model_seed=204,
        split_seed=132,
        structure_budget=25,
        risk_budget=50,
        outputs=100,
        trees=100,
        max_depth=5,
        expected_rows=1024,
    )


def test_validate_reused_effective_config_rejects_protocol_drift():
    payload = _effective_config()
    payload["parameters"]["ntrees"] = 99

    with pytest.raises(ValueError, match="parameters.ntrees"):
        validate_reused_effective_config(
            payload,
            model_seed=204,
            split_seed=132,
            structure_budget=25,
            risk_budget=50,
            outputs=100,
            trees=100,
            max_depth=5,
            expected_rows=1024,
        )
