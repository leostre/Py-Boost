"""Tests for scheduler-hypothesis aggregation."""

from experiments.analyze_scheduler_diagnostics import (
    aggregate_tree_risks,
    cliffs_delta,
)


def test_tree_risk_is_aggregated_separately_per_iteration():
    risks = aggregate_tree_risks(
        (
            {'iteration': 1, 'regret': 0.1, 'full_best_gain': 10.0},
            {'iteration': 1, 'regret': 0.8, 'full_best_gain': 0.1},
            {'iteration': 1, 'regret': None, 'full_best_gain': 0.0},
            {'iteration': 2, 'regret': 0.3, 'full_best_gain': 1.0},
        )
    )

    assert risks == {1: 0.1, 2: 0.3}


def test_cliffs_delta_sign_matches_early_risk_dominance():
    assert cliffs_delta((0.8, 0.9), (0.1, 0.2)) == 1.0
    assert cliffs_delta((0.1, 0.2), (0.8, 0.9)) == -1.0
