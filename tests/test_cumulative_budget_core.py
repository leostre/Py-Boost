from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given,settings,strategies as st

from experiments.cumulative_budget_core import (serving_cost,absolute_allocations,absolute_depth,
    CumulativeError,Quality,CandidateValue,Accept,Reject,decide_candidate,widen_roles)
from experiments.closed_budget_core import Topology,extend_bound
from experiments.sparse_leaf_core import contrast
from test_model_bank_core import tiny_model


@given(st.integers(2,20),st.integers(0,40))
def test_both_representations_match_independent_serialized_array_oracle(c,entries):
    source=tiny_model(c,1)
    s=min(entries,2*c)
    keys=np.arange(s)
    model=replace(source,rows=(keys//c).astype(np.int32),columns=(keys%c).astype(np.int32),
                  values=np.ones(s,np.float32))
    cost=serving_cost(Topology(1,1,2),c,1,s)
    assert cost.coo_bytes==model.payload_bytes
    assert cost.csr_bytes==model.prepare_inference().resident_bytes
    assert cost.maximum_bytes==max(model.payload_bytes,model.prepare_inference().resident_bytes)
    assert cost.parameters==model.parameter_count


@given(st.integers(1,30),st.integers(2,30),st.integers(1,100000))
def test_absolute_plans_never_hide_csr_pointer_cost(t,c,budget):
    topology=Topology(t,3*t,4*t)
    for a in absolute_allocations(topology,c,c,budget):
        assert a.cost.maximum_bytes<=budget
        assert a.cost==serving_cost(topology,c,a.rank,a.nnz)
        assert a.nnz<=a.cap_percent*(budget//4)//100


def test_small_nonempty_residual_csr_is_more_expensive_than_coo():
    topology=Topology(40,100,140)
    cost=serving_cost(topology,14,3,1)
    assert cost.csr_bytes>cost.coo_bytes
    plans=absolute_allocations(topology,14,14,cost.coo_bytes)
    assert all(a.cost.csr_bytes<=cost.coo_bytes for a in plans)


def test_depth_reserves_all_future_actual_representation_costs():
    empty=Topology(0,0,0)
    result=absolute_depth(empty,200,14,3,100,131072)
    assert result in (3,4,5)
    future=extend_bound(empty,200,result)
    assert serving_cost(future,14,3,100).maximum_bytes<=131072
    assert absolute_depth(empty,200,14,3,100,10) is None


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_error_telescopes_same_actual_increments_and_cannot_mutate_previous_state(task):
    rng=np.random.default_rng(145)
    previous=rng.normal(size=(11,7))
    reference=previous.copy()
    debt=CumulativeError(np.zeros_like(previous),task)
    for _ in range(6):
        increment=rng.normal(size=previous.shape)
        proposal=previous+increment
        compact=proposal+rng.normal(size=previous.shape)*.1
        reference+=increment
        prior=debt.values.copy()
        old=debt
        debt=debt.advance(proposal,compact)
        np.testing.assert_allclose(debt.values,contrast(reference-compact,task),rtol=0,atol=2e-15)
        np.testing.assert_array_equal(old.values,prior)
        assert not np.shares_memory(debt.values,old.values)
        with pytest.raises(ValueError): debt.values[0,0]=999
        previous=compact


def test_common_multiclass_shift_does_not_look_like_compression_debt():
    before=np.arange(12).reshape(3,4).astype(float)
    after=before+np.array([3.,-2.,5.])[:,None]
    debt=CumulativeError(np.zeros((3,4)),'multiclass').advance(before,after)
    np.testing.assert_array_equal(debt.values,np.zeros((3,4)))


def test_debt_rule_does_not_override_quality_or_budget_guards():
    ref=Quality(.3,.5,.6)
    c=serving_cost(Topology(1,1,2),4,1,0)
    bad=CandidateValue(0,'bad',Quality(.9,.5,.6),0.,c)
    good=CandidateValue(1,'good',ref,1.,c)
    assert decide_candidate([bad,good],ref,100000,'cumulative')==Accept(1)
    assert isinstance(decide_candidate([bad],ref,100000,'cumulative'),Reject)
    assert decide_candidate([good],ref,c.maximum_bytes-1,'cumulative').reason=='no_memory_feasible'
    higher=CandidateValue(2,'better_macro',Quality(.3,.51,.6),2.,c)
    assert decide_candidate([good,higher],ref,100000,'local')==Accept(2)
    assert decide_candidate([good,higher],ref,100000,'cumulative')==Accept(1)


@given(st.integers(2,8),st.integers(0,10000),st.sampled_from(['multiclass','multilabel']))
@settings(max_examples=30,deadline=None)
def test_widened_roles_conserve_ids_and_never_touch_test_or_calibration(c,seed,task):
    old=dict(train=np.arange(120),calibration=np.arange(120,140),control=np.arange(140,150),
             selection=np.arange(150,160),test=np.arange(160,200))
    saved={k:v.copy() for k,v in old.items()}
    target=np.arange(120)%c if task=='multiclass' else ((np.arange(120)[:,None]+np.arange(c))%3==0).astype(int)
    one=widen_roles(target,old,c,task,seed)
    two=widen_roles(target.copy(),old,c,task,seed)
    assert sorted(np.concatenate(list(one.values())).tolist())==list(range(200))
    for name in old:
        np.testing.assert_array_equal(old[name],saved[name])
        np.testing.assert_array_equal(one[name],two[name])
    for name in ('calibration','test'):
        np.testing.assert_array_equal(one[name],old[name])
    assert len(one['train'])<len(old['train'])
    assert len(one['selection'])>len(old['selection'])
    assert len(one['control'])>len(old['control'])


def test_role_function_rejects_full_target_axis_instead_of_reading_test_labels():
    old=dict(train=np.arange(12),calibration=np.arange(12,14),control=np.arange(14,16),
             selection=np.arange(16,18),test=np.arange(18,20))
    with pytest.raises(ValueError,match='axis mismatch'):
        widen_roles(np.zeros((20,3)),old,3,'multilabel',17)
