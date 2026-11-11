"""Tests for the RMT budget-labeling contract."""

import pytest
from hypothesis import given, strategies as st

from experiments.analyze_rmt_budget import choose_budget_label, evaluate_budget_actions


def test_nonmonotone_budget_risk_is_evaluated_at_selected_action():
    row = {'n_outputs': 100, 'budget_label': 0,
           'risk_25': 0.08, 'risk_50': 0.12, 'risk_75': 0.03}
    result = evaluate_budget_actions([row], [1])
    assert result['unsafe_rate'] == 1.0
    assert result['mean_selected_risk'] == 0.12


def test_risk_age_is_metadata_not_a_budget():
    row = {'n_outputs': 100, 'risk_25': .08, 'risk_50': .12, 'risk_75': .03,
           'risk_age': 7, 'risk_50_extra': 1., 'last_risk_25': .01}
    result = evaluate_budget_actions([row], [1])
    assert result['unsafe_count'] == 1
    assert result['mean_selected_risk'] == .12


@given(st.tuples(*(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
                  for _ in range(3))), st.integers(min_value=0, max_value=3),
       st.dictionaries(st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=20),
                       st.integers(), max_size=8))
def test_service_risk_fields_cannot_change_action_evaluation(risks, label, metadata):
    row = {'n_outputs': 100, **dict(zip(('risk_25', 'risk_50', 'risk_75'), risks))}
    decorated = {**row, **{f'risk_{key}': value for key, value in metadata.items()}}
    result = evaluate_budget_actions([decorated], [label])
    expected_risk = (*risks, 0.)[label]
    assert result['mean_selected_risk'] == expected_risk
    assert result['unsafe_count'] == int(expected_risk > .10)
    assert result['evaluated_actions'] == 1
    assert result['unknown_actions'] == 0


def test_unknown_probe_is_not_silently_treated_as_safe():
    row = {'n_outputs': 100, 'risk_25': None}
    assert evaluate_budget_actions([row], [0])['unknown_actions'] == 1
    assert evaluate_budget_actions([row], [1])['unsafe_rate'] == 0.0


def test_smallest_safe_budget_is_selected():
    assert choose_budget_label({25: 0.2, 50: 0.08, 75: 0.03}, 100, 0.1) == (50, 1)


def test_full_budget_is_fallback_when_all_probes_are_unsafe():
    assert choose_budget_label({25: 0.2, 50: 0.18, 75: None}, 100, 0.1) == (100, 3)


def test_negative_safe_threshold_is_rejected():
    with pytest.raises(ValueError, match="safe threshold"):
        choose_budget_label({25: 0.1}, 100, -0.1)
