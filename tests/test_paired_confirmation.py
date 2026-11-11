"""Tests for paired experiment confirmation."""

import pytest

from experiments.analyze_paired_confirmation import analyze_records


def test_paired_analysis_uses_matching_dataset_and_seed():
    records = [
        {"dataset": dataset, "case_id": case, "seed": seed, "f1_weighted": value}
        for dataset in ("a", "b")
        for seed in (1, 2)
        for case, value in (("full", 0.8), ("adaptive", 0.79 + 0.01 * seed))
    ]

    rows, validation = analyze_records(
        records,
        baseline_case="full",
        candidate_cases=("adaptive",),
        metrics=("f1_weighted",),
    )

    assert validation["complete"] is True
    assert len(rows) == 2
    assert all(row["pairs"] == 2 for row in rows)
    assert all(row["mean_delta_candidate_minus_baseline"] == pytest.approx(0.005) for row in rows)


def test_paired_analysis_rejects_duplicate_run_keys():
    record = {"dataset": "a", "case_id": "full", "seed": 1, "f1_weighted": 0.8}

    with pytest.raises(ValueError, match="duplicate run keys"):
        analyze_records([record, record], "full", ("adaptive",), ("f1_weighted",))
