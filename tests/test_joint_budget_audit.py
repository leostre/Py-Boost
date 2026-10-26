from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.joint_budget_audit import array_cost, guard, independent_static_choice
from experiments.joint_budget_core import choose_joint
from experiments.sparse_leaf_core import Cost, budget_for
from experiments.shared_leaf_basis import compress, reference_class_model
from experiments.uniform_threshold_core import ThresholdModel
from test_sparse_leaf_core import teacher


@pytest.mark.parametrize('representation', ['dense', 'reference_class', 'low_rank'])
def test_actual_npz_cost_oracle_counts_all_arrays_and_only_one_implicit_zero(tmp_path, representation):
    dense = teacher('multiclass' if representation == 'reference_class' else 'multilabel')
    base = reference_class_model(dense) if representation == 'reference_class' else (
        compress(dense, np.zeros(8), np.eye(8)[:, :2]) if representation == 'low_rank' else dense)
    model = ThresholdModel(base, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32),
                          np.zeros(8, np.float32))
    path = tmp_path/'model.npz'
    model.save(path)
    assert array_cost(path) == (model.parameter_count, model.payload_bytes)


@given(percent=st.sampled_from([0, 1, 2, 5]), fraction=st.sampled_from([.5, .65, .75]),
       family=st.sampled_from(['leaf_only', 'rank_only', 'rank_residual', 'joint_static']))
def test_independent_audit_oracle_agrees_with_controller_on_mixed_cost_quality_grid(percent, fraction, family):
    rows = []
    reference = dict(loss=1., f1_macro=.5, f1_weighted=.6)
    for i in range(12):
        rows.append(dict(candidate_id=str(i), depth=3+i%3, kind=['full', 'rank_only', 'sparse'][i%3],
                         nnz=i%6, parameters=30+5*i, payload_bytes=100+25*i,
                         validation=dict(loss=1.+.02*(i%5), f1_macro=.4+.01*i, f1_weighted=.6)))
    budget = budget_for(Cost(100, 400), 'joint', fraction)
    decision = dict(family=family, cap_percent=percent, budget=budget.__dict__)
    state, chosen = independent_static_choice(rows, reference, decision, 100)
    actual = choose_joint(rows, reference, budget, family, percent, 100)
    assert state == actual.state.value and chosen == actual.candidate


def test_guard_requires_all_three_metrics_not_only_macro():
    reference = dict(loss=1., f1_macro=.5, f1_weighted=.6)
    assert guard(reference, reference)
    assert not guard({**reference, 'loss': 1.051}, reference)
    assert not guard({**reference, 'f1_macro': .474}, reference)
    assert not guard({**reference, 'f1_weighted': .569}, reference)
