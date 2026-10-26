"""Independent lattice/endpoint laws for the preregistered size match."""
import pytest
from hypothesis import given, strategies as st
from experiments.matched_budget import BudgetRegime, budget_regime


@given(outputs=st.integers(1, 1000), leaves=st.integers(100, 100000))
def test_cap_is_the_largest_affordable_lattice_point(outputs, leaves):
    regime = BudgetRegime(True, 200, 500)
    reference = outputs * leaves
    cap = regime.cap(reference, outputs, .75)
    # Integer oracle avoids the implementation's floating-point formula.
    assert cap == outputs * ((3 * leaves) // 4)
    assert cap % outputs == 0 and cap <= .75 * reference < cap + outputs


@given(outputs=st.integers(1, 101), units=st.integers(100, 10000),
       remaining=st.integers(0, 3), reason=st.sampled_from(['cap', 'horizon', 'quality_confirmed_twice']))
def test_endpoint_requires_cap_stop_and_strict_minimum_tree_residual(outputs, units, remaining, reason):
    regime = BudgetRegime(True, 200, 500)
    cap = units * outputs
    result = regime.endpoint(cap - remaining * outputs, cap, outputs, reason)
    assert result['matched'] == (reason == 'cap' and remaining in (0, 1))
    assert result['remaining_parameters'] == remaining * outputs


def test_legacy_defaults_are_preserved():
    r = budget_regime(dict(ntrees=200, parameter_fraction=.75))
    assert r == BudgetRegime(False, 200, 200) and r.stop_on_quality
    assert r.cap(1001, 10, .75) == 750


def test_matched_contract_and_stop_flag():
    r = budget_regime(dict(study_mode='matched_parameters', reference_trees=200,
                          ntrees=500, parameter_fraction=.75))
    assert r == BudgetRegime(True, 200, 500) and not r.stop_on_quality


@pytest.mark.parametrize('field,value', [('reference_trees', 201), ('ntrees', 200),
                                       ('ntrees', True), ('parameter_fraction', .8),
                                       ('study_mode', 'unknown')])
def test_invalid_matched_protocol_rejected(field, value):
    config = dict(study_mode='matched_parameters', reference_trees=200, ntrees=500, parameter_fraction=.75)
    config[field] = value
    with pytest.raises(ValueError):
        budget_regime(config)


@pytest.mark.parametrize('parameters,cap,outputs', [(1001,1000,10), (0,1000,10), (1,10,0)])
def test_invalid_endpoint_is_not_silently_a_nonmatch(parameters, cap, outputs):
    with pytest.raises(ValueError):
        BudgetRegime(True,200,500).endpoint(parameters,cap,outputs,'cap')
