"""Pure gates for the multi-dataset deterministic histogram round."""

from experiments.deterministic_histogram_article_suite_core import (
    aggregate_suite_status,
    analyze_layout_dispatch,
)


def _record(policy, selected, nodes):
    return {
        "policy": policy,
        "selected_layout": selected,
        "nodes": nodes,
        "calls": 1,
    }


def test_hybrid_dispatch_allows_datasets_without_wide_levels():
    result = analyze_layout_dispatch(
        "hybrid",
        [_record("hybrid", "perf1", nodes) for nodes in (1, 2, 4, 8)],
    )

    assert result["gate_passed"] is True
    assert result["both_layouts_observed"] is False


def test_hybrid_dispatch_checks_the_threshold_exactly():
    valid = analyze_layout_dispatch(
        "hybrid",
        [
            _record("hybrid", "perf1", 8),
            _record("hybrid", "perf2", 9),
            _record("hybrid", "perf2", 16),
        ],
    )
    invalid = analyze_layout_dispatch(
        "hybrid",
        [_record("hybrid", "perf2", 8)],
    )

    assert valid["gate_passed"] is True
    assert valid["both_layouts_observed"] is True
    assert invalid["gate_passed"] is False


def test_perf1_dispatch_rejects_any_perf2_call():
    result = analyze_layout_dispatch(
        "perf1",
        [_record("perf1", "perf1", 8), _record("perf1", "perf2", 16)],
    )

    assert result["gate_passed"] is False


def test_suite_status_lists_failed_and_full_dispatch_datasets():
    status = aggregate_suite_status({
        "birds": {
            "correctness_gate_passed": True,
            "dispatch_coverage": {"both_layouts_observed": False},
        },
        "mnist": {
            "correctness_gate_passed": False,
            "dispatch_coverage": {"both_layouts_observed": True},
        },
    })

    assert status == {
        "correctness_gate_passed": False,
        "failed_datasets": ["mnist"],
        "datasets_exercising_both_layouts": ["mnist"],
        "dataset_count": 2,
    }
