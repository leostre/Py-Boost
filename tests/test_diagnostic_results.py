"""Tests for experiment diagnostic aggregation."""

from types import SimpleNamespace

import pytest

from experiments.diagnostic_results import (
    add_baseline_ratios,
    add_grouped_baseline_ratios,
    aggregate_records,
    summarize_history,
)


def test_history_summary_reports_activation_budget_and_unique_indices():
    sketch = SimpleNamespace(
        scheduler_trace=[
            {"iteration": 1, "volatility": None, "stable_regime": False},
            {"iteration": 2, "volatility": 0.2, "stable_regime": True},
        ],
        decision_trace=[
            {
                "iteration": 2,
                "rows": 50,
                "outputs": 10,
                "unique_rows": 50,
                "unique_outputs": 10,
            }
        ],
        split_diagnostics_trace=[
            {
                "iteration": 2,
                "regret": 0.1,
                "full_best_gain": 10.0,
                "same_split": False,
                "spurious_split": False,
            },
            {
                "iteration": 2,
                "regret": None,
                "full_best_gain": 0.0,
                "same_split": True,
                "spurious_split": False,
            }
        ],
    )

    summary = summarize_history(sketch, trees_built=2)

    assert summary["activation_iteration"] == 2
    assert summary["history_decision_rate"] == 0.5
    assert summary["mean_sampled_rows"] == 50
    assert summary["mean_sampled_outputs"] == 10
    assert summary["all_indices_unique"] is True
    assert summary["split_diagnostic_nodes"] == 2
    assert summary["split_risk_nodes"] == 1
    assert summary["split_regret_weighted_p95"] == 0.1
    assert summary["split_match_rate"] == 0.5
    assert summary["spurious_split_rate"] == 0.0


def test_baseline_ratios_are_added_to_every_record():
    records = [
        {
            "variant": "full",
            "accuracy": 0.8,
            "f1_weighted": 0.75,
            "train_seconds": 10.0,
            "inference_median_ms": 4.0,
        },
        {
            "variant": "hasboost",
            "accuracy": 0.81,
            "f1_weighted": 0.76,
            "train_seconds": 5.0,
            "inference_median_ms": 2.0,
        },
    ]

    summarized = add_baseline_ratios(records)

    assert summarized[1]["train_speedup_vs_full"] == 2.0
    assert summarized[1]["inference_speedup_vs_full"] == 2.0
    assert summarized[1]["accuracy_delta_vs_full"] == pytest.approx(0.01)


def test_grouped_ratios_use_the_baseline_from_the_matching_seed():
    records = [
        {
            "dataset": "helena",
            "scenario": "main",
            "case_id": "full",
            "seed": 1,
            "accuracy": 0.8,
            "f1_weighted": 0.7,
            "train_seconds": 10.0,
            "inference_median_ms": 4.0,
        },
        {
            "dataset": "helena",
            "scenario": "main",
            "case_id": "full",
            "seed": 2,
            "accuracy": 0.6,
            "f1_weighted": 0.5,
            "train_seconds": 20.0,
            "inference_median_ms": 8.0,
        },
        {
            "dataset": "helena",
            "scenario": "main",
            "case_id": "hasboost",
            "seed": 2,
            "accuracy": 0.65,
            "f1_weighted": 0.55,
            "train_seconds": 5.0,
            "inference_median_ms": 2.0,
        },
    ]

    summarized = add_grouped_baseline_ratios(records)

    assert summarized[2]["accuracy_delta_vs_full"] == pytest.approx(0.05)
    assert summarized[2]["train_speedup_vs_full"] == 4.0


def test_aggregation_is_order_invariant_and_reports_sample_standard_deviation():
    records = [
        {
            "dataset": "helena",
            "scenario": "main",
            "case_id": "full",
            "variant": "full",
            "seed": 2,
            "accuracy": 0.6,
        },
        {
            "dataset": "helena",
            "scenario": "main",
            "case_id": "full",
            "variant": "full",
            "seed": 1,
            "accuracy": 0.8,
        },
    ]

    forward = aggregate_records(records, metrics=("accuracy",))
    backward = aggregate_records(reversed(records), metrics=("accuracy",))

    assert forward == backward
    assert forward[0]["runs"] == 2
    assert forward[0]["seeds"] == "1,2"
    assert forward[0]["accuracy_mean"] == pytest.approx(0.7)
    assert forward[0]["accuracy_std"] == pytest.approx(2**0.5 / 10)


def test_empty_history_has_no_uniqueness_claim():
    summary = summarize_history(
        SimpleNamespace(scheduler_trace=[], decision_trace=[]),
        trees_built=0,
    )

    assert summary["all_indices_unique"] is None
