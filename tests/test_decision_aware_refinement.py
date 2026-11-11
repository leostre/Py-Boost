from __future__ import annotations

from hypothesis import given, strategies as st

from experiments.decision_aware_refinement_core import (
    assess_refinement_gate,
    evaluate_exact_refinement,
    exact_shortlist_action,
    refinement_work_ratio,
)


def _record(gains, budget=2, full_best_gain=None):
    gains = [float(value) for value in gains]
    best = max(gains) if full_best_gain is None else float(full_best_gain)
    return {
        'diagnostic_output_budget': budget,
        'n_outputs': 4,
        'full_best_gain': best,
        'full_split_available': True,
        'min_gain_to_split': 0.0,
        'regret': max(0.0, (best - gains[0]) / (abs(best) + 1e-6)),
        'diagnostic_top_k': [
            {
                'feature': index,
                'split_bin': index + 1,
                'nan_left': bool(index % 2),
                'full_gain': gain,
            }
            for index, gain in enumerate(gains)
        ],
        'candidate_output_contributions': [
            [gain / 4.0] * 4 for gain in gains
        ],
    }


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=32,
    )
)
def test_exact_refinement_regret_is_monotone_with_nested_shortlists(gains):
    record = _record(gains, full_best_gain=max(gains) + 1.0)
    regrets = [
        float(exact_shortlist_action(record, size)['regret'])
        for size in range(1, len(gains) + 1)
    ]
    assert all(right <= left + 1e-12 for left, right in zip(regrets, regrets[1:]))


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=2, max_value=5000),
)
def test_two_stage_work_is_below_full_when_budget_and_shortlist_are_strict(budget, missing, candidates):
    outputs = budget + missing
    shortlist = candidates - 1
    ratio = refinement_work_ratio(budget, outputs, shortlist, candidates)
    assert 0 < ratio < 1


def test_refinement_uses_all_output_profile_not_recorded_diagnostic_order():
    record = _record([4.0, 9.0, 6.0], full_best_gain=10.0)
    action = exact_shortlist_action(record, 3)
    assert action['candidate_index'] == 1
    assert action['full_gain_from_profile'] == 9.0
    assert float(action['regret']) < float(record['regret'])


def test_evaluation_reports_both_policies_and_applies_primary_gate():
    records = [_record([5.0, 10.0], budget=2, full_best_gain=10.0)]
    policies = {'fixed': {2: 2}, 'adaptive': {2: 2}}
    evaluation = evaluate_exact_refinement(records, policies, (2,), 100, 1e-6)
    metrics = evaluation['per_policy']['adaptive']['2']
    assert metrics['refined']['weighted_p95_regret'] == 0.0
    assert metrics['weighted_p95_regret_reduction_fraction'] == 1.0
    gate = assess_refinement_gate(evaluation, 'adaptive', (2,), 0.2)
    assert gate['passed'] is True


def test_profile_shape_is_validated():
    record = _record([1.0, 2.0])
    record['candidate_output_contributions'][0] = [1.0]
    try:
        exact_shortlist_action(record, 2)
    except ValueError as error:
        assert 'wrong output count' in str(error)
    else:
        raise AssertionError('invalid profile shape was accepted')
