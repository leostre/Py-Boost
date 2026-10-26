"""Independent checks of calibration, decision/probability and storage laws."""
import inspect
import numpy as np
import pytest
from hypothesis import given,settings,strategies as st

from experiments.shared_leaf_basis import compress,functional_basis,probabilities
from experiments.sparse_leaf_core import SparseLeafModel,Cost,leaf_design,contrast
from experiments.output_calibration_core import (DecisionLeafModel,affine_objective,
    calibration_weights,indicator,fold_affine,fit_affine,fit_decision_shifts,
    calibrated_quality,labels_from_raw)
from experiments.tiny_residual_study import calibration_partition,tiny_plan,eligible_family,materialize_tiny
from test_sparse_leaf_core import teacher


@settings(max_examples=25,deadline=None)
@given(seed=st.integers(0,10000),task=st.sampled_from(['multilabel','multiclass']))
def test_folded_affine_matches_explicit_transform_without_extra_persistent_parameters(seed,task):
    rng=np.random.default_rng(seed);dense=teacher(task)
    x=np.array([[-1],[1],[np.nan]],np.float32);leaves=dense.leaf_indices(x)
    mean,basis,_=functional_basis(dense.contributions(np.tile(leaves,(1,3))),task)
    base=compress(dense,mean,basis[:,:2])
    a=rng.uniform(.25,4,8);b=rng.uniform(-2,2,8)
    folded=fold_affine(base,a,b)
    expected=contrast(base.raw_from_leaves(leaves),task)*a+b
    actual=folded.raw_from_leaves(leaves)
    np.testing.assert_allclose(contrast(actual,task),contrast(expected,task),atol=2e-6)
    np.testing.assert_allclose(probabilities(actual,task),probabilities(expected,task),atol=4e-7)
    assert folded.parameter_count==base.parameter_count and folded.payload_bytes==base.payload_bytes
    np.testing.assert_allclose(folded.basis.T@folded.basis,np.eye(2),atol=3e-7)


@pytest.mark.parametrize('task',['multilabel','multiclass'])
@pytest.mark.parametrize('balanced',[False,True])
def test_affine_gradient_matches_finite_difference_and_fit_improves_objective(task,balanced):
    rng=np.random.default_rng(91);raw=rng.normal(size=(15,4))
    target=rng.integers(0,4,15) if task=='multiclass' else rng.integers(0,2,(15,4))
    y=indicator(target,4,task);w=calibration_weights(target,4,task,balanced)
    theta=np.r_[rng.uniform(.5,2,4),rng.normal(size=4)]
    objective,gradient=affine_objective(theta,raw,y,w,task,.1)
    for j in range(8):
        plus=theta.copy();minus=theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
        numeric=(affine_objective(plus,raw,y,w,task,.1)[0]-affine_objective(minus,raw,y,w,task,.1)[0])/2e-6
        assert gradient[j]==pytest.approx(numeric,abs=2e-8)
    fit=fit_affine(raw,target,task,balanced,.1)
    assert fit['final_objective']<=fit['initial_objective']+1e-10
    assert fit['converged']


def test_decision_only_adjustment_changes_f1_but_never_probability_loss_and_counts_vector(tmp_path):
    dense=teacher(c=4);base=compress(dense,np.zeros(4),np.eye(4)[:,:2])
    leaf=SparseLeafModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32))
    shifts=np.full(4,-2,np.float32);model=DecisionLeafModel(leaf,shifts)
    x=np.array([[-1],[1]],np.float32);raw=leaf.raw_from_design(leaf_design(base.leaf_indices(x),3))
    y=np.ones((2,4),int)
    original=calibrated_quality(y,raw,'multilabel',np.empty(0,np.float32))
    changed=calibrated_quality(y,raw,'multilabel',shifts)
    assert changed['loss']==original['loss'] and changed['f1_macro']>original['f1_macro']
    assert model.parameter_count==leaf.parameter_count+4 and model.payload_bytes==leaf.payload_bytes+16
    path=tmp_path/'model.npz';model.save(path);loaded=DecisionLeafModel.load(path)
    np.testing.assert_array_equal(loaded.predict(x),model.predict(x))
    np.testing.assert_array_equal(loaded.predict_labels(x),model.predict_labels(x))
    with np.load(path,allow_pickle=False) as f:
        assert model.payload_bytes==sum(f[k].nbytes for k in ('nodes','node_offsets','leaf_offsets',
            'coefficients','basis','bias','rows','columns','values','decision_shifts'))
        assert not {'scale','shift','teacher','original_coefficients'}&set(f.files)


def test_decision_grid_fitting_has_scalar_oracle_and_zero_positive_fallback():
    raw=np.array([[-.5,-1],[-.2,1],[.1,2]])
    y=np.array([[1,0],[1,0],[0,0]])
    shifts=fit_decision_shifts(raw,y,'multilabel',[0,-1,1])
    np.testing.assert_array_equal(shifts,[-1,0])
    np.testing.assert_array_equal(labels_from_raw(raw,'multilabel',shifts),raw>=shifts)


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_calibration_partition_disjoint_repeatable_and_exhaustive(task):
    target=np.arange(40)%4 if task=='multiclass' else (np.arange(160).reshape(40,4)%3==0)
    a,b=calibration_partition(target,task,4,42);aa,bb=calibration_partition(target,task,4,42)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
    assert not set(a)&set(b) and set(a)|set(b)==set(range(40))


@given(parameters=st.integers(100,100000),nnz=st.integers(1,6000))
def test_small_caps_use_floor_and_never_admit_oversized_residual(parameters,nnz):
    row={'kind':'sparse','nnz':nnz}
    for family,percent in [('sparse_01',1),('sparse_02',2),('sparse_05',5)]:
        assert eligible_family(row,family,parameters)==(nnz<=int(np.floor(parameters*percent/100)))


def test_plan_uses_disjoint_calibration_and_selection_inputs_not_test_and_replays_choices(monkeypatch):
    dense=teacher(c=4);x=np.array([[-1],[1]]*12,np.float32)
    y=(np.arange(96).reshape(24,4)%3==0).astype(int)
    train,calibration,selection=np.arange(12),np.arange(12,18),np.arange(18,24)
    leaves=[dense.leaf_indices(x[v]) for v in (train,calibration,selection)]
    mean,basis,_=functional_basis(dense.contributions(leaves[0]),dense.task)
    config={'budget_kinds':['parameters','memory','joint'],'budget_fractions':[.99],
        'base_rank_fractions':[.5,.75,.9],'residual_capacity_fractions':[.25,.5,1.],
        'tiny_selectors':['balanced_mse','balanced_margin'],'margin_scale':.5,
        'calibration_regularization':[.1],'decision_grid':[0,-1,1],'decision_shrinkage':[.5,1.]}
    calibration_y=y[calibration].copy();selection_y=1-y[selection]
    import experiments.tiny_residual_study as study
    original_fit=study.fit_affine
    fit_calls=[]
    def checked_fit(raw,target,*args):
        np.testing.assert_array_equal(target,calibration_y)
        fit_calls.append(len(raw))
        return original_fit(raw,target,*args)
    monkeypatch.setattr(study,'fit_affine',checked_fit)
    plan=tiny_plan(dense,mean,basis,leaves[0],y[train],leaves[1],calibration_y,leaves[2],selection_y,
                    Cost(dense.parameter_count,dense.payload_bytes),config)
    design=leaf_design(leaves[2],3)
    for selected in plan['selections']:
        chosen=selected['candidate'];model=materialize_tiny(chosen,plan,dense,mean,basis)
        assert model.parameter_count==chosen['parameters'] and model.payload_bytes==chosen['payload_bytes']
        q=calibrated_quality(selection_y,model.raw_from_design(design),dense.task,model.decision_shifts)
        for key in q:assert q[key]==pytest.approx(chosen['validation'][key],abs=1e-9)
    assert not any('test' in p for p in inspect.signature(tiny_plan).parameters)
    assert fit_calls and all(n==len(calibration_y) for n in fit_calls)
