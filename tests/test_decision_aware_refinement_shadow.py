from __future__ import annotations

from experiments.decision_aware_refinement_shadow_core import (
    assess_refinement_shadow_gate,
    evaluate_refinement_shadow,
)


def _entry(budget=2, ratio=0.5, close=True, depth=0):
    return {
        'depth': depth,
        'diagnostic_output_budget': budget,
        'shortlist_size': 4,
        'initial_statistics_seconds': 0.3,
        'shortlist_selection_seconds': 0.1,
        'refinement_seconds': 0.1,
        'two_stage_seconds': ratio,
        'full_statistics_seconds': 0.9,
        'full_selection_seconds': 0.1,
        'full_search_seconds': 1.0,
        'maximum_candidate_gain_absolute_error': 1e-5,
        'maximum_candidate_gain_relative_error': 1e-6,
        'candidate_gains_allclose': close,
        'all_refined_winners_gain_equivalent': close,
        'nodes': [
            {
                'full_split_available': True,
                'full_best_gain': 10.0,
                'baseline_regret': 0.5,
                'refined_regret': 0.1,
            },
            {
                'full_split_available': True,
                'full_best_gain': 5.0,
                'baseline_regret': 0.4,
                'refined_regret': 0.05,
            },
        ],
    }


def test_shadow_gate_requires_correctness_regret_and_runtime_for_every_budget():
    evaluation = evaluate_refinement_shadow([_entry(2), _entry(3)], (2, 3), 1e-6)
    gate = assess_refinement_shadow_gate(evaluation, (2, 3), 0.2)
    assert gate['passed'] is True
    assert evaluation['per_budget']['2']['runtime_ratio_to_full_search'] == 0.5


def test_shadow_gate_rejects_a_fast_but_inexact_kernel():
    evaluation = evaluate_refinement_shadow([_entry(close=False)], (2,), 1e-6)
    gate = assess_refinement_shadow_gate(evaluation, (2,), 0.2)
    assert gate['per_budget']['2']['correctness_passed'] is False
    assert gate['passed'] is False


def test_shadow_gate_rejects_exact_refinement_slower_than_full_search():
    evaluation = evaluate_refinement_shadow([_entry(ratio=1.1)], (2,), 1e-6)
    gate = assess_refinement_shadow_gate(evaluation, (2,), 0.2)
    assert gate['per_budget']['2']['runtime_passed'] is False
    assert gate['passed'] is False


def test_root_runtime_is_reported_but_does_not_replace_aggregate_gate():
    trace = [
        _entry(ratio=1.1, depth=0),
        _entry(ratio=0.5, depth=1),
    ]
    evaluation = evaluate_refinement_shadow(trace, (2,), 1e-6)
    metrics = evaluation['per_budget']['2']
    gate = assess_refinement_shadow_gate(evaluation, (2,), 0.2)
    assert metrics['root_runtime_ratio_to_full_search'] == 1.1
    assert metrics['runtime_ratio_to_full_search'] == 0.8
    assert gate['per_budget']['2']['runtime_passed'] is True
    assert gate['passed'] is True
