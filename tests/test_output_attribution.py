"""Tests for leave-one-output attribution analysis."""

import pytest

from experiments.analyze_output_attribution import (
    aggregate_output_attribution,
    budget_capture,
    evaluate_output_probabilities,
)


def test_output_attribution_is_aggregated_per_tree_and_output():
    records = (
        {
            'iteration': 1,
            'regret': 0.1,
            'full_best_gain': 10.0,
            'leave_one_output_regret': [0.1, 0.3],
        },
        {
            'iteration': 1,
            'regret': 0.2,
            'full_best_gain': 0.1,
            'leave_one_output_regret': [0.9, 0.8],
        },
        {
            'iteration': 1,
            'regret': 1.0,
            'full_best_gain': 0.0,
            'full_split_available': False,
            'leave_one_output_regret': [None, None],
        },
    )

    assert aggregate_output_attribution(records) == {1: (0.1, 0.3)}


def test_budget_capture_reports_spectral_optimum_and_uniform_expectation():
    captured, optimal, uniform = budget_capture(
        probabilities=(0.1, 0.8, 0.1),
        importance=(1.0, 3.0, 2.0),
        budget=1,
    )

    assert captured == pytest.approx(0.5)
    assert optimal == pytest.approx(0.5)
    assert uniform == pytest.approx(1 / 3)


def test_spectral_evaluation_joins_on_iteration():
    rows = evaluate_output_probabilities(
        (
            {
                'representation': 'z',
                'window': 4,
                'end_iteration': 4,
                'output_probabilities': [0.1, 0.2, 0.7],
            },
        ),
        {4: (0.1, 0.2, 0.9)},
        budgets=(1,),
    )

    assert rows[0]['output_spearman'] == 1.0
    assert rows[0]['capture_ratio_to_optimal_1'] == 1.0
