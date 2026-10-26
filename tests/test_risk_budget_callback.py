"""CPU state-machine harness, not a substitute for GPU integration tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck


class GPU(np.ndarray):
    def get(self):
        return np.asarray(self)


def array(x):
    return np.asarray(x).view(GPU)


@pytest.fixture
def controller_class(monkeypatch):
    monkeypatch.setitem(sys.modules,'py_boost.callbacks.callback',SimpleNamespace(Callback=object))
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(float32=np.float32,
        empty=lambda shape,dtype:np.empty(shape,dtype=dtype).view(GPU)))
    def measured(target,p,task):
        # Deliberately simple fixture metric with the real >= .5 decision boundary.
        return dict(loss=float(-np.log(np.clip(p,1e-7,1-1e-7)).mean()),
                    f1_macro=float(np.all(p>=.5)),f1_weighted=float(np.all(p>=.5)))
    monkeypatch.setitem(sys.modules,'experiments.a100_runner',SimpleNamespace(quality=measured))
    path=Path(__file__).resolve().parents[1]/'experiments/risk_budget_callback.py'
    spec=importlib.util.spec_from_file_location('risk_callback_cpu',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QualityEfficiencyController


def state():
    model=SimpleNamespace(base_score=array([0.]),models=[],
        loss=SimpleNamespace(postprocess_output=lambda x:array(1/(1+np.exp(-x)))))
    return dict(model=model,builder=SimpleNamespace(params={}),num_iter=0,
        data=dict(train=dict(target=array(np.ones((100,1))),grad=array(np.ones((100,1))),hess=array(np.ones((100,1)))),
                  valid=dict(target=[array(np.ones((10,1)))],ensemble=[array(np.zeros((10,1),dtype=np.float64))],
                             last_tree=dict(preds=[array(np.zeros((10,1),dtype=np.float32))]))))


@pytest.mark.parametrize('mode',['fixed','alternating','latest','history','deceleration'])
def test_quality_stop_is_not_before_two_checks_after_warmup(controller_class,mode):
    c=controller_class(mode=mode,reference=dict(loss=.7,f1_macro=1.,f1_weighted=1.),
        parameter_cap=10000,seed=1,task='multilabel',horizon=200,emit=lambda *a,**k:None)
    info=state(); c.before_train(info)
    for step in range(1,81):
        info['num_iter']=step-1; c.before_iteration(info)
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=2**c.applied_depth,values=np.zeros((2,1))))
        stop=c.after_iteration(info)
        assert stop == (step==80)
    assert c.checks[-1]['stop_reason']=='quality_confirmed_twice'
    assert np.all(info['data']['valid']['ensemble'][0]==0)


def test_partial_cap_terminal_is_censored(controller_class):
    c=controller_class(mode='history',reference=dict(loss=.001,f1_macro=1.,f1_weighted=1.),
        parameter_cap=2200,seed=1,task='multilabel',horizon=200,emit=lambda *a,**k:None)
    info=state(); c.before_train(info)
    for step in range(1,201):
        info['num_iter']=step-1;c.before_iteration(info)
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=2**c.applied_depth,values=np.zeros((2,1))))
        if c.after_iteration(info):break
    assert c.parameters<=c.cap and step%20!=0
    assert c.checks[-1]['stop_reason']=='cap' and c.checks[-1]['censored_window']
    assert c.checks[-1]['completed_window'] is None


def test_deployment_precision_not_training_precision_controls_threshold(controller_class):
    c=controller_class(mode='fixed',reference=dict(loss=.001,f1_macro=1.,f1_weighted=1.),
        parameter_cap=10000,seed=1,task='multilabel',horizon=200,emit=lambda *a,**k:None)
    info=state();c.before_train(info);c.horizon=3
    for step,contribution in enumerate([1.,-1e-8,-1.],1):
        info['num_iter']=step-1;c.before_iteration(info)
        info['data']['valid']['last_tree']['preds']=[array(np.full((10,1),contribution,dtype=np.float32))]
        info['data']['valid']['ensemble'][0]+=contribution
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=2,values=np.zeros((2,1))))
        c.after_iteration(info)
    assert c.validation_prediction.dtype==np.float32
    assert c.checks[-1]['validation']['f1_macro']==1.
    assert np.all(info['data']['valid']['ensemble'][0]<0)


@pytest.mark.parametrize('mode',['fixed','alternating','latest','history','deceleration'])
def test_matched_regime_continues_after_quality_confirmation_until_cap(controller_class,mode):
    c=controller_class(mode=mode,reference=dict(loss=.7,f1_macro=1.,f1_weighted=1.),
        parameter_cap=3000,seed=1,task='multilabel',horizon=500,emit=lambda *a,**k:None,
        stop_on_quality=False)
    info=state(); c.before_train(info)
    for step in range(1,501):
        info['num_iter']=step-1; c.before_iteration(info)
        leaves=2**c.applied_depth
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=leaves,values=np.zeros((leaves,1))))
        if c.after_iteration(info):break
    assert step>80 and c.checks[-1]['stop_reason']=='cap'
    assert 0<=c.cap-c.parameters<2
    assert any(k['confirmations']>=2 and not k['stop'] for k in c.checks)
    if mode in ('latest','history','deceleration'):
        assert any(k['decision_reason']=='target_met_budget_continuation' for k in c.checks)


@given(cap=st.integers(2100,2600), mode=st.sampled_from(['fixed','alternating','history','deceleration']),
       leaf_fraction=st.integers(1,2))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_variable_actual_leaf_counts_still_fill_the_same_cap(controller_class,cap,mode,leaf_fraction):
    c=controller_class(mode=mode,reference=dict(loss=.7,f1_macro=1.,f1_weighted=1.),
        parameter_cap=cap,seed=1,task='multilabel',horizon=500,emit=lambda *a,**k:None,
        stop_on_quality=False)
    info=state(); c.before_train(info); independent=1
    for step in range(1,501):
        info['num_iter']=step-1; c.before_iteration(info)
        leaves=max(1,2**c.applied_depth//leaf_fraction)
        independent+=leaves
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=leaves,values=np.zeros((leaves,1))))
        if c.after_iteration(info):break
    assert c.parameters==independent<=cap and cap-independent<2
    assert c.checks[-1]['stop_reason']=='cap'
    assert all(r['applied_depth']==5 for r in c.trace[:60])


def test_matched_horizon_exhaustion_is_not_budget_exhaustion(controller_class):
    c=controller_class(mode='history',reference=dict(loss=.7,f1_macro=1.,f1_weighted=1.),
        parameter_cap=100000,seed=1,task='multilabel',horizon=100,emit=lambda *a,**k:None,
        stop_on_quality=False)
    info=state();c.before_train(info)
    for step in range(1,101):
        info['num_iter']=step-1;c.before_iteration(info)
        info['model'].models.append(SimpleNamespace(ngroups=1,max_leaves=2**c.applied_depth,values=np.zeros((2,1))))
        stop=c.after_iteration(info)
        assert stop==(step==100)
    assert c.checks[-1]['stop_reason']=='horizon' and c.cap-c.parameters>2
