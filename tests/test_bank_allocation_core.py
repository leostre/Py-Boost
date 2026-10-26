from copy import deepcopy
import pytest
from hypothesis import given,strategies as st,settings
from experiments.bank_allocation_core import choose_allocation,key,retention


def row(task,family,cost,utility=1.,passed=True):
    q=dict(loss=1./utility,f1_macro=utility,f1_weighted=utility)
    return dict(task_id=task,family=family,csr_bytes=cost,sha256=f'{task}-{family}',
        validation=q,reference_validation=dict(loss=1.,f1_macro=1.,f1_weighted=1.),development_pass=passed)


def test_joint_spends_unequal_resource_without_test_knowledge():
    groups=[[row('small','full',2),row('small','joint_static',1,.96)],
            [row('large','full',8),row('large','joint_static',6,.96)]]
    joint=choose_allocation(groups,8,'joint')
    assert [r['family'] for r in joint]==['full','joint_static']
    assert len([r for r in choose_allocation(groups,8,'equal') if r is not None])==1
    dirty=deepcopy(groups)
    for i,g in enumerate(dirty):
        for r in g: r['test_quality']=999 if i==0 else -999
    assert [r['sha256'] for r in choose_allocation(dirty,8,'joint')]==[r['sha256'] for r in joint]


@given(st.lists(st.tuples(st.integers(1,20),st.integers(1,20)),min_size=1,max_size=4),
       st.integers(0,100),st.integers(0,100))
@settings(max_examples=60,deadline=None)
def test_resource_conservation_and_coverage_monotonicity(costs,b1,b2):
    groups=[[row(str(i),'full',a),row(str(i),'joint_static',b,.97)] for i,(a,b) in enumerate(costs)]
    small=choose_allocation(groups,min(b1,b2),'joint')
    big=choose_allocation(groups,max(b1,b2),'joint')
    assert sum(r['csr_bytes'] for r in small if r is not None)<=min(b1,b2)
    assert sum(r is not None for r in small)<=sum(r is not None for r in big)
    for policy in ('equal','independent'):
        other=choose_allocation(groups,b1,policy)
        exact=choose_allocation(groups,b1,'joint')
        assert key(exact)>=key(other)


def test_failed_development_candidate_is_not_revived_and_budget_boundary_exact():
    groups=[[row('a','full',9),row('a','joint_static',1,.2,False)]]
    assert choose_allocation(groups,8,'joint')==(None,)
    assert choose_allocation(groups,9,'joint')[0]['family']=='full'


def test_not_just_model_count_worst_task_quality_has_priority():
    groups=[[row('a','full',4),row('a','joint_static',2,.96)],
            [row('b','full',5),row('b','joint_static',1,.98)]]
    chosen=choose_allocation(groups,6,'joint')
    assert [r['family'] for r in chosen]==['full','joint_static']
    assert key(chosen)[1]==pytest.approx(.98)


def test_identical_artifact_deduplication_preserves_independent_policy_membership():
    rank=row('a','rank_residual',2,.99)
    joint=dict(rank,family='joint_static')  # Exact same artifact, different selector family.
    full=row('a','full',9)
    for options in ([full,rank,joint],[joint,rank,full],[rank,full,joint]):
        chosen=choose_allocation([options],3,'independent')
        assert chosen[0] is not None
        assert chosen[0]['family']=='joint_static'
        assert chosen[0]['sha256']==rank['sha256']


@pytest.mark.parametrize('policy',['equal','joint','independent'])
def test_no_hidden_mutation(policy):
    groups=[[row('a','full',4),row('a','joint_static',2,.96)]]
    before=deepcopy(groups)
    choose_allocation(groups,3,policy)
    assert groups==before
