from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import budget_quality_worker
from abl_budget_revision_core import (fixed_support_projection, bounded_allocate,
    support_order, maximum_capacity, marginal_price, prune_tree, reference_basis,
    decode_model, blank, pack, deficit, select_point)


@settings(max_examples=45, deadline=None)
@given(st.integers(2,10), st.integers(1,7), st.integers(0,1000))
def test_fixed_support_projection_against_independent_lstsq(c, requested, seed):
    r=min(requested,c); rng=np.random.default_rng(seed); b=np.linalg.qr(rng.normal(size=(c,r)))[0]
    target=rng.normal(size=(4,c)); d=np.exp(rng.normal(size=(4,c)))
    keys=np.flatnonzero(rng.random(target.size)<.35)
    before=[x.copy() for x in (target,d,b)]
    a,rows,cols,values=fixed_support_projection(target,d,b,keys)
    free=d.copy();free.ravel()[keys]=0
    for i in range(len(target)):
        root=np.sqrt(free[i]); expected=np.linalg.lstsq(root[:,None]*b,root*target[i],rcond=1e-6)[0]
        np.testing.assert_allclose(a[i],expected,atol=2e-6,rtol=2e-6)
    decoded=a.astype(float)@b.T;decoded[rows,cols]+=values
    np.testing.assert_allclose(decoded.ravel()[keys],target.ravel()[keys],atol=3e-6)
    for old,new in zip(before,(target,d,b)):np.testing.assert_array_equal(old,new)


def test_full_support_and_full_rank_are_valid_extreme_cases():
    target=np.arange(18.).reshape(3,6);weights=np.ones_like(target)
    a,r,c,v=fixed_support_projection(target,weights,np.eye(6),())
    np.testing.assert_allclose(a,target)
    a,r,c,v=fixed_support_projection(target,weights,np.eye(6)[:,:2],np.arange(target.size))
    expected=a.astype(float)@np.eye(6)[:,:2].T;expected[r,c]+=v
    np.testing.assert_allclose(expected,target)


def test_quota_saturation_redistributes_before_rounding():
    np.testing.assert_array_equal(bounded_allocate(5,[100,2,1],[1,100,100]),[1,3,1])
    np.testing.assert_array_equal(bounded_allocate(5,[0,0,0],[1,2,2]),[1,2,2])
    with pytest.raises(ValueError):bounded_allocate(10,[1,1],[2,3])


@settings(max_examples=70, deadline=None)
@given(st.lists(st.tuples(st.integers(0,15),st.integers(0,15)),min_size=1,max_size=10),st.integers(0,300))
def test_apportionment_conserves_budget_and_respects_output_support(items, requested):
    weights,caps=map(np.asarray,zip(*items));total=min(requested,int(caps.sum()))
    q=bounded_allocate(total,weights,caps)
    assert q.sum()==total and (q>=0).all() and (q<=caps).all()
    if total and np.all(caps>total) and weights.sum()>0:
        ideal=total*weights/weights.sum()
        assert (np.abs(q-ideal)<1+1e-12).all()


def test_deficit_quota_is_addressed_and_zero_deficit_keeps_old_order():
    order=np.array([0,3,1,4,2,5,6,7,8]);keys,info=support_order(order,3,[0,0,1],3)
    assert set(keys)=={2,5,8} and info['active'] and info['quotas']==[0,0,3]
    old,info=support_order(order,3,[0,0,0],3)
    np.testing.assert_array_equal(old,order[:3]);assert not info['active']
    keys,info=support_order(order,3,[0,0,1],5)
    assert len(set(keys))==5 and set([2,5,8]).issubset(keys) and info['fallback']==2


def test_capacity_checks_csr_first_entry_and_finds_highest_feasible_count():
    leaves=[11,7];masses=[11,7];caps=[15,12]
    for budget in range(100,401,11):
        result=maximum_capacity(budget,100,95,leaves,masses,caps)
        feasible=[q for q in range(sum(caps)+1) if marginal_price(100,95,leaves,bounded_allocate(q,masses,caps))<=budget]
        assert result==(max(feasible) if feasible else None)


def balanced_eight_leaf_tree():
    # Thresholds implement eight intervals. Complete binary tree, preorder ids.
    nodes=np.array([[1,0,1,4],[1,-.5,2,3],[1,-.75,-1,-2],[1,-.25,-3,-4],
        [1,.5,5,6],[1,.25,-5,-6],[1,.75,-7,-8]],np.float32)
    return nodes


@pytest.mark.parametrize('cap',[4,6,8])
def test_pruning_is_a_legal_coarsening_with_exact_full_cap(cap):
    from audit_abl_budget_revision import topology_mapping
    rng=np.random.default_rng(22);nodes=balanced_eight_leaf_tree();v=rng.normal(size=(8,5));d=rng.uniform(1,20,(8,5))
    p=prune_tree(nodes,v,d,np.eye(5),cap)
    assert len(p['target'])==cap and len(p['nodes'])==cap-1
    np.testing.assert_array_equal(topology_mapping(nodes,p['nodes'],p['origin_nodes']),p['mapping'])
    for j in range(cap):
        which=p['mapping']==j
        np.testing.assert_allclose(p['target'][j],(v[which]*d[which]).sum(0)/d[which].sum(0))
    if cap==8:
        np.testing.assert_array_equal(p['nodes'],nodes)
        np.testing.assert_allclose(p['coefficients'],v,atol=2e-7)


def test_reference_gauge_preserves_nested_function_spaces_and_full_probabilities():
    from experiments.shared_leaf_basis import probabilities
    rng=np.random.default_rng(11);signal=rng.normal(size=(50,10));wire,contrast=reference_basis(signal)
    for rank in (4,6,8,9):
        transformed=contrast[:-1,:rank]-contrast[-1:,:rank]
        q=np.linalg.qr(transformed)[0]
        np.testing.assert_allclose(q@q.T,wire[:-1,:rank]@wire[:-1,:rank].T,atol=3e-7)
    target=rng.normal(size=(10,10));ref=target[:,:-1]-target[:,-1:]
    a,_,_,_=fixed_support_projection(ref,np.ones_like(ref),wire[:-1],())
    decoded=a.astype(float)@wire.T
    np.testing.assert_allclose(probabilities(target,'multiclass'),probabilities(decoded,'multiclass'),atol=3e-7)


def test_real_sparse_model_price_equals_both_predicted_envelopes():
    from test_abl_quality_core import fixture
    from abl_protected_core import block_cost
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.shared_leaf_basis import compress
    model,x,design,y=fixture(9,4,7);base=compress(model.base,np.zeros(4),np.eye(4)[:,:2])
    zero=pack([blank(base)],np.zeros(4,np.float32));p=block_cost(zero)
    for s in (0,1,7,13):
        rows=(np.arange(s)//4).astype(np.int32);cols=(np.arange(s)%4).astype(np.int32)
        b=ThresholdModel(base,rows,cols,np.ones(s,np.float32),np.zeros(4,np.float32))
        actual=block_cost(pack([b],np.zeros(4,np.float32)))
        assert actual['cost_bytes']==marginal_price(p['coo_resident_bytes'],p['prepared_blocks_bytes'],[len(base.coefficients)],[s])


def test_selection_checks_all_metrics_and_not_only_macro():
    ref=dict(loss=1.,f1_macro=.8,f1_weighted=.9)
    a=dict(uid='higher-f1',cost_bytes=50,selection=dict(loss=1.5,f1_macro=.9,f1_weighted=.95),control_guard=True)
    b=dict(uid='passes',cost_bytes=60,selection=dict(loss=1.,f1_macro=.79,f1_weighted=.88),control_guard=True)
    assert select_point([a,b],'selection',ref,100,True)['uid']=='passes'
    assert deficit(b['selection'],ref)==0


def fixture_saved_case(tmp_path,task='multiclass'):
    from test_abl_quality_core import fixture
    from abl_protected_core import as_protected
    from budget_quality_worker import write,digest
    from experiments.shared_leaf_basis import reference_class_model
    c=10 if task=='multiclass' else 101
    teacher,_,_,_=fixture(29,c,200)
    if task=='multiclass':teacher=replace(teacher,base=reference_class_model(teacher.base))
    else:teacher=replace(teacher,base=replace(teacher.base,task='multilabel'))
    x=np.tile(np.linspace(-1,1,100,dtype=np.float32),12)[:,None]
    y=np.arange(len(x))%c if task=='multiclass' else ((np.arange(len(x))[:,None]+np.arange(c))%7==0).astype(np.uint8)
    roles=dict(train=np.arange(800),calibration=np.arange(800,920),control=np.arange(920,1040),selection=np.arange(1040,1160),test=np.arange(1160,1200))
    data=tmp_path/'data.npz';splits=tmp_path/'roles.npz';np.savez(data,features=x,target=y);np.savez(splits,**roles)
    from experiments.parallel_leaf_core import prefix
    metas={}
    for n in (40,160,200):
        m=as_protected(prefix(teacher,n));p=tmp_path/f'full{n}.npz';m.save(p)
        metas[f'full/b{n//40-1:02d}']=dict(uid=f'full/b{n//40-1:02d}',artifact=str(p),sha256=digest(p))
    case=dict(task=task,record=dict(case=dict(task=task,outputs=c,split_file=str(splits),split_sha256=digest(splits)),dataset=dict(path=str(data),sha256=digest(data))),metas=metas)
    p=tmp_path/'original.json';write(p,case)
    from abl_protected_core import block_cost
    full=metas['full/b04'];fm=__import__('abl_protected_core').load_protected(full['artifact'])
    reg=dict(case='mnist_3104' if task=='multiclass' else 'mediamill_1104',task=task,budget_bytes=10**7,
        original_case=dict(path=str(p),sha256=digest(p)),initial_snapshot=metas['full/b00'],
        frozen_full_reference=dict(full,trees=200,cost_bytes=block_cost(fm)['cost_bytes'],token='frozen',policy=dict(temperature=1.,shifts=[0.]*c)))
    return reg,case


def test_mnist_worker_boundary_on_real_saved_toy_models_without_training(tmp_path):
    from abl_budget_revision import mnist_case
    from audit_abl_budget_revision import audit
    from budget_quality_worker import locked
    reg,source=fixture_saved_case(tmp_path)
    cfg=dict(decision_grid=[0,.5,-.5],decision_shrinkage=[.5,1.])
    ref=mnist_case(reg,cfg,tmp_path/'output',{})
    result=locked(ref['path'],ref['sha256'])
    assert len(result['rows'])==24 and result['new_tree_fits']==0 and not result['test_accessed']
    checked=audit([ref],dict(cases=[reg]))
    assert checked['status']=='passed'


def test_media_worker_boundary_sparse_archive_and_independent_audit(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from budget_quality_worker import digest,locked
    from abl_budget_revision import media_case
    from audit_abl_budget_revision import audit
    from abl_protected_core import load_protected,block_cost
    from abl_budget_revision_core import decode_model,blank,pack
    from experiments.shared_leaf_basis import compress
    reg,source=fixture_saved_case(tmp_path,'multilabel')
    original=load_protected(source['metas']['full/b03']['artifact']);view,_=decode_model(original)
    rng=np.random.default_rng(40);b=np.linalg.qr(rng.normal(size=(101,50)))[0]
    blocks=[]
    for i in range(4):
        base=view.slice(40*i,40*(i+1)).base()
        blocks.append(blank(base if i==0 else compress(base,np.zeros(101),b)))
    model=pack(blocks,view.bias);fullpath=tmp_path/'compact160.npz';model.save(fullpath)
    short=pack(blocks[:2],view.bias);shortpath=tmp_path/'compact80.npz';short.save(shortpath)
    policy=dict(temperature=1.,shifts=[0.]*101)
    reg['initial_snapshot']=dict(artifact=str(shortpath),sha256=digest(shortpath))
    reg['budget_bytes']=180000
    item=dict(trees=160,artifact=str(fullpath),sha256=digest(fullpath),policy=policy)
    cfg=dict(decision_grid=[0,.5,-.5],decision_shrinkage=[.5,1.])
    monkeypatch.setattr('abl_budget_revision.MEDIA_METHODS',('M-F','M-BS'))
    ref=media_case(reg,cfg,tmp_path/'out',{},[dict(case=reg['case'],paths=[dict(path='functional',snapshots=[item])])])
    result=locked(ref['path'],ref['sha256'])
    assert len(result['rows'])==16 and result['exact_recode_raw_error']<2e-5
    assert all(r['cost_bytes']<=reg['budget_bytes'] for r in result['rows'])
    checked=audit([ref],dict(cases=[reg]));assert checked['status']=='passed'
