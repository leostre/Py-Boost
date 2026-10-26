import inspect
import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.joint_budget_core import choose_joint, SelectionState
from experiments.sparse_leaf_core import Cost, budget_for
from experiments.shared_leaf_basis import functional_basis
from experiments.uniform_threshold_study import uniform_plan
from test_sparse_leaf_core import teacher

Q = dict(loss=1., f1_macro=.4, f1_weighted=.5)


def row(depth=5, kind='rank_only', nnz=0, p=20, b=80, macro=.4):
    return dict(candidate_id=f'{depth}-{kind}-{nnz}-{p}-{b}', depth=depth, kind=kind,
                nnz=nnz, parameters=p, payload_bytes=b, validation={**Q, 'f1_macro': macro})


def test_no_fallback_outside_actual_memory_and_quality_failure_is_explicit():
    budget = budget_for(Cost(100, 400), 'joint', .75)
    assert choose_joint([row(p=76)], Q, budget, 'joint_static', 5, 100).state == SelectionState.NO_MEMORY_FEASIBLE
    assert choose_joint([row(b=301)], Q, budget, 'joint_static', 5, 100).candidate is None
    choice = choose_joint([row(macro=.2)], Q, budget, 'joint_static', 5, 100)
    assert choice.state == SelectionState.NO_QUALITY_FEASIBLE and choice.candidate is not None


@given(depth=st.sampled_from([3, 4, 5]), nnz=st.integers(0, 8), percent=st.sampled_from([0, 1, 2, 5]))
def test_family_and_residual_boundary_have_independent_oracle(depth, nnz, percent):
    r = row(depth=depth, kind='sparse', nnz=nnz)
    budget = budget_for(Cost(100, 400), 'joint', .75)
    choice = choose_joint([r], Q, budget, 'rank_residual', percent, 100)
    assert (choice.candidate is not None) == (depth == 5 and nnz <= percent)


def test_shallow_full_candidate_competes_only_in_leaf_or_joint_family():
    budget = budget_for(Cost(100, 400), 'joint', .75)
    r = row(depth=3, kind='full')
    for family in ('leaf_only', 'joint_static'):
        assert choose_joint([r], Q, budget, family, 0, 100).state == SelectionState.ACCEPTED
    for family in ('rank_only', 'rank_residual'):
        assert choose_joint([r], Q, budget, family, 5, 100).candidate is None
    assert not any('test' in k for k in inspect.signature(choose_joint).parameters)


def test_grid_mode_allows_no_feasible_rank_without_oversize_fallback():
    dense = teacher('multilabel', c=8)
    x = np.array([[-1], [1]]*9, np.float32)
    y = np.arange(18*8).reshape(18, 8)%2
    leaves = dense.leaf_indices(x[:6])
    mean, basis, _ = functional_basis(dense.contributions(leaves), 'multilabel')
    cfg = dict(budget_kinds=['joint'], budget_fractions=[.5, .75], base_rank_fractions=[.5],
               residual_capacity_fractions=[1.], tiny_selectors=['balanced_mse'], margin_scale=.5,
               decision_grid=[0], decision_shrinkage=[1.])
    plan = uniform_plan(dense, dense, mean, basis, leaves, y[:6], leaves, y[6:12], leaves, y[12:],
                        Cost(1, 4), cfg, build_selections=False)
    assert not plan['rows'] and not plan['selections'] and plan['full_candidates']
    with pytest.raises(RuntimeError, match='no calibrated rank'):
        uniform_plan(dense, dense, mean, basis, leaves, y[:6], leaves, y[6:12], leaves, y[12:],
                     Cost(1, 4), cfg)
