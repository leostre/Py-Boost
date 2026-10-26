"""Independent cost, numeric, serialization and leakage oracles; CPU only."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.shared_leaf_basis import LeafEnsemble, compress, functional_basis
from experiments.sparse_leaf_core import (Budget,Cost,SparseLeafModel,budget_for,
    coordinate_scores,contrast,leaf_design,make_sparse_model,residual_capacity,
    score_order,select_candidate,training_weights)
from experiments.sparse_leaf_study import development_targets,plan_candidates


def teacher(task='multilabel',c=8):
    return LeafEnsemble(np.array([[1,0,-1,-2],[1,0,-1,-1]],np.float32),
        np.array([0,1,2],np.int64),np.array([0,2,3],np.int64),
        np.arange(3*c,dtype=np.float32).reshape(3,c)/11,
        np.empty((0,0),np.float32),np.linspace(-1,1,c,dtype=np.float32),1,task)


@settings(max_examples=80,deadline=None)
@given(parameters=st.integers(20,1000),memory=st.integers(100,10000),
       p_slack=st.integers(0,100),b_slack=st.integers(0,1200))
def test_capacity_is_maximum_under_both_limits(parameters,memory,p_slack,b_slack):
    base=Cost(parameters,memory)
    budget=Budget('joint',.75,parameters+p_slack,memory+b_slack)
    maximum=residual_capacity(base,budget)
    # Brute-force oracle protects missing COO index costs and off-by-one errors.
    legal=[k for k in range(101) if parameters+k<=budget.parameters and memory+12*k<=budget.payload_bytes]
    assert maximum==max(legal)
    assert budget.allows(Cost(parameters+maximum,memory+12*maximum))
    assert not budget.allows(Cost(parameters+maximum+1,memory+12*(maximum+1)))


def test_parameter_and_memory_caps_are_different_not_fake_padding():
    reference=Cost(100,800)
    assert budget_for(reference,'parameters',.75)==Budget('parameters',.75,75,None)
    assert budget_for(reference,'memory',.75)==Budget('memory',.75,None,600)
    base=Cost(50,500)
    assert residual_capacity(base,budget_for(reference,'parameters',.75))==25
    assert residual_capacity(base,budget_for(reference,'memory',.75))==8
    assert residual_capacity(Cost(80,500),budget_for(reference,'joint',.75))==0


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_full_centered_residual_reconstructs_teacher_on_unseen_routes(task):
    dense=teacher(task)
    x=np.array([[-1],[-1],[-1],[1],[1]],np.float32)
    leaves=dense.leaf_indices(x);design=leaf_design(leaves,3)
    mean,basis,_=functional_basis(dense.contributions(leaves),task)
    base=compress(dense,mean,basis[:,:1])
    residual=(contrast(dense.coefficients,task)
              -contrast(base.coefficients.astype(float)@base.basis.T,task)).astype(np.float32)
    frequency=np.asarray(design.mean(axis=0)).ravel()
    order=np.flatnonzero(residual.ravel()!=0)
    model=make_sparse_model(base,residual,order,len(order),frequency)
    test=np.array([[1],[-1],[np.nan]],np.float32)
    raw=model.raw_from_design(leaf_design(dense.leaf_indices(test),3))
    np.testing.assert_allclose(contrast(raw,task),contrast(dense.raw_from_leaves(dense.leaf_indices(test)),task),atol=4e-7)


def test_sparse_inference_matches_scalar_oracle_and_roundtrips(tmp_path):
    dense=teacher();base=compress(dense,np.zeros(8),np.eye(8)[:,:2])
    residual=np.zeros((3,8),np.float32);residual[0,4]=.3;residual[1,5]=-.2
    x=np.array([[-1],[-1],[1]],np.float32)
    design=leaf_design(dense.leaf_indices(x),3)
    frequency=np.asarray(design.mean(axis=0)).ravel()
    model=make_sparse_model(base,residual,np.array([4,13]),2,frequency)
    expected=[]
    for t in range(len(x)):
        indices=[0 if x[t,0]<=0 else 1,2]
        row=base.bias.astype(float).copy()
        for index in indices:
            row+=base.coefficients[index].astype(float)@base.basis.T
            row+=residual[index]
        row+=model.base.bias-base.bias
        expected.append(row)
    np.testing.assert_allclose(model.raw_from_design(design),expected,atol=1e-7)
    assert model.parameter_count==base.parameter_count+2
    assert model.payload_bytes==base.payload_bytes+24
    path=tmp_path/'model.npz';model.save(path);loaded=SparseLeafModel.load(path)
    np.testing.assert_array_equal(loaded.predict(x),model.predict(x))
    assert loaded.base.coefficients.shape==(3,2) and loaded.base.coefficients.flags.owndata
    with np.load(path,allow_pickle=False) as f:
        assert 'dense' not in f.files and f['values'].shape==(2,)
        assert model.payload_bytes==sum(f[k].nbytes for k in ('nodes','node_offsets','leaf_offsets',
                                       'coefficients','basis','bias','rows','columns','values'))
    assert path.stat().st_size>model.payload_bytes


@pytest.mark.parametrize('bad',['duplicate','zero','nan','outside','dtype'])
def test_invalid_sparse_coordinates_rejected(bad):
    base=compress(teacher(),np.zeros(8),np.eye(8)[:,:1])
    rows=np.array([0,1],np.int32);cols=np.array([1,1],np.int32);values=np.array([.1,.2],np.float32)
    if bad=='duplicate':rows[1]=0
    if bad=='zero':values[0]=0
    if bad=='nan':values[0]=np.nan
    if bad=='outside':rows[0]=3
    if bad=='dtype':rows=rows.astype(np.int64)
    with pytest.raises(ValueError):SparseLeafModel(base,rows,cols,values)


@settings(max_examples=35,deadline=None)
@given(seed=st.integers(0,10000))
def test_influence_score_equals_independent_single_coordinate_loss_decrease(seed):
    rng=np.random.default_rng(seed)
    leaves=np.array([[0,1,0,1,0],[2,2,2,2,2]])
    phi=leaf_design(leaves,3)
    residual=rng.normal(size=(3,4)).astype(np.float32)
    error=rng.normal(size=(5,4));w=rng.uniform(.1,1,size=(5,4));w/=w.sum(axis=0)
    actual=coordinate_scores(phi,residual,error,'balanced_mse',w)
    frequency=np.asarray(phi.mean(axis=0)).ravel()
    for row in range(3):
        for col in range(4):
            correction=(phi[:,row].toarray().ravel()-frequency[row])*float(residual[row,col])
            before=np.sum(w[:,col]*error[:,col]**2)
            after=np.sum(w[:,col]*(error[:,col]-correction)**2)
            assert actual[row,col]==pytest.approx(max(0,before-after),abs=2e-6)


def test_training_weights_balance_classes_and_margin_is_gauge_invariant():
    target=np.array([[1,0],[0,1],[0,1],[0,1]])
    w=training_weights(target,np.ones((4,2)),'multilabel','balanced_mse')
    np.testing.assert_allclose(w.sum(axis=0),1)
    np.testing.assert_allclose((w*target).sum(axis=0),.5)
    raw=np.array([[2.,1,0],[0,2,1],[3,1,2],[1,3,0]])
    y=np.array([0,1,2,1])
    a=training_weights(y,raw,'multiclass','balanced_margin')
    b=training_weights(y,raw+np.array([[100],[-30],[20],[-70]]),'multiclass','balanced_margin')
    np.testing.assert_allclose(a,b,atol=1e-14)


def test_order_is_deterministic_and_ignores_nonpositive_or_zero_coordinates():
    np.testing.assert_array_equal(score_order(np.array([[2.,2.,0.,-0.]]),
                                             np.array([[1.,1.,1.,1.]])),[0,1])


def test_validation_selection_never_uses_test_fields_and_honors_both_caps():
    ref={'loss':1.,'f1_macro':.8,'f1_weighted':.8}
    rows=[{'candidate_id':'a','parameters':20,'payload_bytes':100,'validation':ref,'test':{'f1_macro':0}},
          {'candidate_id':'b','parameters':21,'payload_bytes':101,'validation':{**ref,'f1_macro':.81},'test':{'f1_macro':1}},
          {'candidate_id':'c','parameters':40,'payload_bytes':200,'validation':{**ref,'f1_macro':.99}}]
    budget=Budget('joint',.75,30,150)
    assert select_candidate(rows,ref,budget)[0]['candidate_id']=='b'
    rows[0]['test']={'f1_macro':1};rows[1]['test']={'f1_macro':0}
    assert select_candidate(rows,ref,budget)[0]['candidate_id']=='b'
    assert select_candidate(rows,ref,Budget('joint',.75,30,100))[0]['candidate_id']=='a'


def test_poisoned_heldout_targets_leave_actual_development_plan_unchanged():
    dense=teacher();x=np.array([[-1],[1]]*10,np.float32)
    y=(np.arange(160).reshape(20,8)%3==0).astype(int)
    indices={'train':np.arange(12),'validation':np.arange(12,16),'test':np.arange(16,20)}
    poisoned=y.astype(float);poisoned[indices['test']]=np.nan
    leaves={k:dense.leaf_indices(x[v]) for k,v in indices.items() if k!='test'}
    mean,basis,_=functional_basis(dense.contributions(leaves['train']),dense.task)
    config={'budget_kinds':['parameters','memory','joint'],'budget_fractions':[.95],
            'base_rank_fractions':[.5,.75,.9],'residual_capacity_fractions':[.25,.5,1.], 'margin_scale':.5}
    train_y,validation_y=development_targets(y,indices)
    a=plan_candidates(dense,mean,basis,leaves['train'],train_y,
                      leaves['validation'],validation_y,
                      Cost(dense.parameter_count,dense.payload_bytes),config)
    tr,va=development_targets(poisoned,indices)
    b=plan_candidates(dense,mean,basis,leaves['train'],tr,leaves['validation'],va,
                      Cost(dense.parameter_count,dense.payload_bytes),config)
    assert a['rows']==b['rows'] and a['selections']==b['selections']
    assert not any('test' in name for name in inspect.signature(plan_candidates).parameters)
