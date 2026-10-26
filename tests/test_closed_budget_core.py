from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.closed_budget_core import (Topology, compact_cost, rank_limit, extend_bound,
    plan_depth, bounded_allocations, independent_snapshot)
from experiments.sparse_leaf_core import Budget, Cost, budget_for
from experiments.uniform_threshold_core import ThresholdModel
from test_sparse_leaf_core import teacher


@given(t=st.integers(1, 50), c=st.integers(2, 40), spare_p=st.integers(0, 1000),
       spare_b=st.integers(0, 4000))
def test_rank_limit_matches_brute_force_for_independent_real_caps(t, c, spare_p, spare_b):
    topology = Topology(t, 3*t, 2*t)  # shared leaf pointers: deliberately M != I+T
    one = compact_cost(topology, c, 1, 0)
    budget = Budget('joint', .75, one.parameters+spare_p, one.payload_bytes+spare_b)
    oracle = [r for r in range(1, c+1) if budget.allows(compact_cost(topology, c, r, 0))]
    assert rank_limit(topology, c, budget) == max(oracle, default=0)


@given(t=st.integers(1, 30), c=st.integers(2, 30), fraction=st.sampled_from([.5, .65, .75]))
def test_every_allocation_meets_both_caps_and_rejects_next_coo_entry(t, c, fraction):
    topology = Topology(t, 3*t, 4*t)
    budget = budget_for(Cost(4*t*c+c, 16*3*t+16*(t+1)+16*t*c+4*c), 'joint', fraction)
    for a in bounded_allocations(topology, c, budget, 4*t*c+c, c):
        assert budget.allows(a.cost)
        assert a.nnz <= a.cap_percent*(4*t*c+c)//100
        assert a.cost == compact_cost(topology, c, a.rank, a.nnz)
        if a.nnz < a.cap_percent*(4*t*c+c)//100 and a.nnz < 4*t*c:
            assert not budget.allows(compact_cost(topology, c, a.rank, a.nnz+1))


def test_no_action_is_not_a_full_model_fallback():
    topology = Topology(20, 100, 120)
    budget = Budget('joint', .75, 1, 4)
    assert rank_limit(topology, 20, budget) == 0
    assert bounded_allocations(topology, 20, budget, 10000, 20) == ()
    assert plan_depth(topology, 180, 20, 1, 0, budget) is None


def test_serialized_array_oracle_does_not_assume_binary_leaf_count():
    dense = teacher('multilabel', c=8)
    from experiments.shared_leaf_basis import compress
    base = compress(dense, np.zeros(8), np.eye(8)[:, :2])
    model = ThresholdModel(base, np.array([0], np.int32), np.array([0], np.int32),
                           np.array([.1], np.float32), np.zeros(8, np.float32))
    t = Topology(base.n_trees, len(base.nodes), len(base.coefficients))
    assert compact_cost(t, 8, 2, 1) == Cost(model.parameter_count, model.payload_bytes)
    frozen = independent_snapshot(model)
    for field in ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'):
        assert not np.shares_memory(getattr(frozen.base, field), getattr(model.base, field))
    for field in ('rows', 'columns', 'values', 'decision_shifts'):
        assert not np.shares_memory(getattr(frozen, field), getattr(model, field))
    model.base.coefficients[:] = 99
    model.values[:] = 88
    assert frozen.base.coefficients.max() < 99 and frozen.values[0] == np.float32(.1)


@pytest.mark.parametrize('value', [True, -1, 1.5])
def test_invalid_counts_do_not_reach_memory_plans(value):
    with pytest.raises(ValueError):
        Topology(value, 2, 3)


def test_reserve_uses_future_structure_not_only_current_rank():
    initial = Topology(0, 0, 0)
    future = extend_bound(initial, 200, 4)
    exact = compact_cost(future, 20, 10, 0)
    budget = Budget('joint', .75, exact.parameters, exact.payload_bytes)
    assert plan_depth(initial, 200, 20, 10, 0, budget) == 4
