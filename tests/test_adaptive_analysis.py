"""Tests for adaptive-budget and row/output experiment analysis."""

from experiments.analyze_adaptive_budget import summarize_budget_trace
from experiments.analyze_row_output_factorial import factorial_contrast


def test_budget_trace_summary_counts_transitions_and_first_reduction():
    trace = [
        {"iteration": 1, "output_budget": 100},
        {"iteration": 5, "output_budget": 100},
        {"iteration": 6, "output_budget": 75},
        {"iteration": 11, "output_budget": 50},
        {"iteration": 16, "output_budget": 75},
    ]

    summary = summarize_budget_trace(trace)

    assert summary["budget_transitions"] == 3
    assert summary["first_reduction_iteration"] == 6
    assert summary["budget_histogram"] == "50:1;75:2;100:2"


def test_factorial_contrast_is_zero_for_additive_row_and_output_effects():
    assert factorial_contrast(full=1.0, rows=0.9, outputs=0.8, joint=0.7) == 0.0
