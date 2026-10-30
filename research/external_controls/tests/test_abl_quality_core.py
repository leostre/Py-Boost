"""Numerical and conservation contracts against independent tiny ensembles."""
from dataclasses import replace
import numpy as np
import pytest

# Resolve the mechanically pinned native implementation, never a GPU package.
import budget_quality_worker
from abl_quality_core import (representation_plans,rank_levels,fit_representation_family,
    prepare_rank,build_representation,design_raw,array_ledger,fingerprint,
    choose_budget,choose_mode_policy,source_models,RepresentationPlan,allowed_policies)
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel
from experiments.sparse_leaf_core import leaf_design
from budget_quality_core import probabilities


def fixture(seed=1,classes=4,trees=3):
    rng=np.random.default_rng(seed)
    nodes=np.array([[1.,.25,-1,-2]]*trees,np.float32)
    nodes[:,1]=np.linspace(-.7,.7,trees)
    base=LeafEnsemble(nodes,np.arange(trees+1,dtype=np.int64),np.arange(trees+1,dtype=np.int64)*2,
        rng.normal(0,.3,(2*trees,classes)).astype(np.float32),np.empty((0,0),np.float32),
        rng.normal(0,.1,classes).astype(np.float32),1,'multiclass')
    model=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(classes,np.float32))
    x=np.linspace(-1,1,37,dtype=np.float32)[:,None]
    design=leaf_design(base.leaf_indices(x),len(base.coefficients))
    return model,x,design,np.arange(len(x))%classes


@pytest.mark.parametrize('classes',[2,3,7])
@pytest.mark.parametrize('seed',[1,9,25])
def test_full_contrast_rank_preserves_probabilities_and_model_roundtrip(classes,seed,tmp_path):
    model,x,design,y=fixture(seed,classes,5)
    state=fit_representation_family(model,design,y)
    full,_,_=prepare_rank(model,design,state,classes-1)
    expected=sum(model.base.coefficients[model.base.leaf_indices(x)[t]].astype(float)
                 for t in range(model.base.n_trees))+model.base.bias
    np.testing.assert_allclose(design_raw(model,design),expected,atol=1e-12)
    np.testing.assert_allclose(probabilities(design_raw(full,design)),probabilities(expected),atol=2e-7)
    path=tmp_path/'model.npz';full.save(path);loaded=ThresholdModel.load(path)
    assert fingerprint(loaded)==fingerprint(full)
    np.testing.assert_allclose(loaded.prepare_inference().raw(x),design_raw(full,design),atol=2e-7)


def test_sparse_mean_preserved_and_positive_coordinates_match_direct_indicator_sum(tmp_path):
    model,x,design,y=fixture(3,5,8)
    state=fit_representation_family(model,design,y)
    pure,residual,order=prepare_rank(model,design,state,1)
    assert len(order)>0
    hybrid=build_representation(pure,residual,order,state,RepresentationPlan(1,.05,min(5,len(order))))
    raw=design_raw(hybrid,design)
    direct=design_raw(pure,design).copy()
    visits=design.toarray()
    for r,c,v in zip(hybrid.rows,hybrid.columns,hybrid.values):
        direct[:,c]+=v*(visits[:,r]-state['frequency'][r])
    np.testing.assert_allclose(raw,direct,atol=3e-7)
    np.testing.assert_allclose(raw.mean(0),state['teacher'].mean(0),atol=3e-7)
    np.testing.assert_allclose(hybrid.prepare_inference().raw(x),raw,atol=3e-7)
    path=tmp_path/'sparse.npz';hybrid.save(path)
    assert fingerprint(ThresholdModel.load(path))==fingerprint(hybrid)


@pytest.mark.parametrize('classes',[2,10,100])
def test_saved_reference_class_parent_expands_only_for_fitting(classes):
    from experiments.shared_leaf_basis import reference_class_model
    dense,x,design,y=fixture(17,classes,5)
    reference=ThresholdModel(reference_class_model(dense.base),dense.rows,dense.columns,
        dense.values,dense.decision_shifts)
    before=fingerprint(reference);cost=array_ledger(reference)['cost_bytes']
    state=fit_representation_family(reference,design,y)
    full,_,_=prepare_rank(reference,design,state,classes-1)
    expected=reference.prepare_inference().raw(x)
    np.testing.assert_allclose(probabilities(design_raw(full,design)),probabilities(expected),atol=2e-6)
    assert fingerprint(reference)==before
    assert array_ledger(reference)['cost_bytes']==cost
    assert reference.base.coefficients.shape[1]==classes-1


@pytest.mark.parametrize('seed',range(5))
def test_array_ledger_matches_actual_owned_arrays_and_does_not_use_max(seed):
    model,x,design,y=fixture(seed)
    state=fit_representation_family(model,design,y)
    pure,residual,order=prepare_rank(model,design,state,1)
    hybrid=build_representation(pure,residual,order,state,RepresentationPlan(1,.05,min(3,len(order))))
    prepared=hybrid.prepare_inference();ledger=array_ledger(hybrid,prepared)
    physical=sum(getattr(prepared.base,n).nbytes for n in
        ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'))
    physical+=sum(getattr(prepared,n).nbytes for n in ('indptr','columns','values','decision_shifts'))
    assert ledger['cost_bytes']==physical+np.array(1,np.float32).nbytes
    assert ledger['both_retained_bytes']==hybrid.payload_bytes+physical+4
    assert ledger['both_retained_bytes']>max(hybrid.payload_bytes,physical)
    assert not np.shares_memory(prepared.base.coefficients,hybrid.base.coefficients)
    assert not np.shares_memory(prepared.base.nodes,hybrid.base.nodes)


def test_budget_selection_respects_exact_boundary_and_ignores_unrelated_test_values():
    metas={'native':dict(family='pyboost',cost_bytes=100),'ab':dict(family='compact',nnz=0,cost_bytes=49),
           'sparse':dict(family='compact',nnz=3,cost_bytes=60)}
    def row(uid,loss,f1):
        return dict(uid=uid,validation={p:dict(loss=loss,f1_macro=f1) for p in ('raw','temperature')})
    rows=[row('native',.3,.8),row('ab',.4,.7),row('sparse',.35,.78)]
    assert choose_budget(rows,metas,'abl','temperature','loss',59)['uid']=='ab'
    assert choose_budget(rows,metas,'abl','temperature','loss',60)['uid']=='sparse'
    assert choose_budget(rows,metas,'native','raw','loss',99) is None
    for r in rows:r['test_quality']=-100 if r['uid']=='ab' else 100
    assert choose_budget(rows[::-1],metas,'abl','temperature','loss',60)['uid']=='sparse'


def test_modes_always_include_identity_and_never_assign_shift_as_nll_improvement():
    for mode in ('raw','temperature','shifts','temperature_shifts'):
        assert 'raw' in allowed_policies(mode,'f1_macro')
        assert not any('shift' in name for name in allowed_policies(mode,'loss'))
    scores={'raw':dict(loss=.2,f1_macro=.7),'temperature':dict(loss=.3,f1_macro=.7)}
    assert choose_mode_policy(scores,'temperature','loss')=='raw'


def test_source_selection_uses_origin_new_and_rejects_duplicate_new_grid():
    models=[dict(identity=f'd{d}-t{t}',depth=d,rounds=t,origin='new')
            for d in (3,5) for t in (40,80,160,200,400,800)]
    archival=dict(models[9],identity='archival-d5-t200',origin='archival_control')
    record=dict(case_id='a',family='pyboost',models=models+[archival])
    _,chosen=source_models([record],'a','pyboost',True)
    assert len(chosen)==8 and not any(m['identity'].startswith('archival') for m in chosen)
    record['models'].append(dict(models[0]))
    with pytest.raises(ValueError,match='duplicate'):source_models([record],'a','pyboost')


def test_rank_and_residual_contract_has_bounded_grid_and_no_invalid_contrast():
    assert rank_levels(2,20,30)==(1,)
    plans=representation_plans(100,500,500,1000)
    assert len(plans)==20 and max(p.rank for p in plans)==99
    assert {p.maximum_nnz for p in plans}=={0,10,20,50}
    with pytest.raises(ValueError):representation_plans(3,5,10,True)
