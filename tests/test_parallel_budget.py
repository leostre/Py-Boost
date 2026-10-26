from dataclasses import replace
import numpy as np
import pytest
from test_cumulative_budget_projection import dense,config
from experiments.uniform_threshold_core import ThresholdModel
from experiments.parallel_budget_core import VARIANTS,branch_plans,mean_increment,union_estimate
from experiments.closed_budget_study import combine_actual
from experiments.history_basis_study import raw


def model(trees,seed=1):
    d=dense(trees)
    d=replace(d,coefficients=np.random.default_rng(seed).normal(0,.1,d.coefficients.shape).astype(np.float32))
    return ThresholdModel(d,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(4,np.float32))


def test_plans_charge_equal_physical_trees_and_identical_prefix():
    assert len({branch_plans(v,0) for v in VARIANTS})==1
    for v in VARIANTS:
        assert sum(p.trees for b in range(5) for p in branch_plans(v,b))==200
    assert branch_plans('par_half',1)[0].lr==.05
    assert [p.seed_offset for p in branch_plans('parallel2_union',1)]==[0,1]
    with pytest.raises(ValueError): branch_plans('four-unapproved',1)


@pytest.mark.parametrize('seed',range(10))
def test_mean_increment_and_common_function_paid_once(seed):
    x=(np.arange(40,dtype=np.float32)%2).reshape(-1,1)
    a,b,checkpoint=model(20,seed),model(20,seed+100),model(40,seed+200)
    checkpoint=replace(checkpoint,base=replace(checkpoint.base,bias=np.ones(4,np.float32)*.7))
    combined=mean_increment([a,b])
    assert combined.base.n_trees==40
    np.testing.assert_allclose(raw(combined,x),(raw(a,x)+raw(b,x))/2,atol=3e-7,rtol=1e-6)
    merged=combine_actual(checkpoint,combined,np.zeros(4,np.float32))
    np.testing.assert_allclose(raw(merged,x),raw(checkpoint,x)+(raw(a,x)+raw(b,x))/2,atol=5e-7,rtol=1e-6)
    repeated=mean_increment([a,a])
    np.testing.assert_allclose(raw(repeated,x),raw(a,x),atol=3e-7)
    candidates=[combine_actual(checkpoint,m,np.zeros(4,np.float32)) for m in (a,b)]
    estimate=union_estimate(candidates,[a,b],x)
    np.testing.assert_allclose(estimate.basis.T@estimate.basis,np.eye(4),atol=1e-10)
    assert estimate.diagnostics['temporary_function_stack_bytes']==2*40*4*8


def test_effect_shell_builds_two20s_and_one_common40_without_gpu(tmp_path,monkeypatch):
    from experiments.cumulative_budget_study import train_cumulative_path
    from experiments import parallel_leaf_worker,parallel_leaf_study
    from experiments.a100_runner import sha256
    jobs=[]
    def fit(req):
        m=model(req['trees'],req['seed'])
        m=replace(m,base=replace(m.base,coefficients=np.zeros_like(m.base.coefficients)))
        m.save(req['output'])
        jobs.append(req)
        return dict(trees=req['trees'],artifact=req['output'],sha256=sha256(req['output']),
            initial_sha256=sha256(req['initial_path']),data_sha256=sha256(req['data_path']),seconds=0.,training_pool_peak_bytes=1)
    def execute(requests,pool):
        return [fit(r) for r in requests],dict(sum_worker_fit_seconds=0.,sum_individual_pool_peaks=2,
            execution='concurrent',fit_overlap_seconds=.1)
    monkeypatch.setattr(parallel_leaf_worker,'fit_increment',fit)
    monkeypatch.setattr(parallel_leaf_study,'execute_requests',execute)
    x=(np.arange(40,dtype=np.float32)%2).reshape(-1,1)
    y=np.tile((np.arange(40)%2)[:,None],(1,4)).astype(np.float32)
    dev={k:(x.copy(),y.copy()) for k in ('train','calibration','control')}
    case=dict(dataset='toy',seed=1,task='multilabel',outputs=4,absolute_budget_bytes=1000000)
    cfg=dict(config(),reserve_future_topology=True,serving_tree_horizon=200,cumulative_pilot_tree_cap=200)
    ledger=dict(trees=0)
    models,trajectory=train_cumulative_path('parallel2_union',case,dev,cfg,tmp_path/'path',ledger,execution_pool=object())
    assert [r['trees'] for r in jobs]==[40]+[20]*8
    assert ledger['trees']==200 and models[-1][1].base.n_trees==200
    assert len(trajectory['cycles'][1]['fit']['branch_sources'])==2
    assert trajectory['cycles'][1]['projection']['basis_source']['kind']=='parallel_union'
