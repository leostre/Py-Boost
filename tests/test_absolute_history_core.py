import numpy as np
import pytest
from experiments.absolute_history_core import AbsoluteHistory,diagonal_derivatives
from experiments.shared_leaf_basis import probabilities,compress


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_derivatives_follow_native_diagonal_convention_and_no_shifts(task):
    f=np.array([[.2,-.1,1.],[2.,-3.,-.5]])
    y=np.array([2,0],np.int32) if task=='multiclass' else np.array([[0,0,1],[1,0,0]])
    p=probabilities(f,task)
    g,h=diagonal_derivatives(y,f,task)
    expected=np.eye(3)[y] if task=='multiclass' else y
    np.testing.assert_array_equal(g,p-expected)
    np.testing.assert_array_equal(h,p*(1-p)*(1.5 if task=='multiclass' else 1))


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_first_observation_equal_then_history_differs_and_state_cost_paid(task):
    rng=np.random.default_rng(271)
    y=rng.integers(0,4,40,dtype=np.int32) if task=='multiclass' else rng.integers(0,2,(40,4))
    f=rng.normal(size=(40,4))
    states=[AbsoluteHistory(k,4,task) for k in ('current','mean','history')]
    first=[s.observe(y,f,1,2) for s in states]
    for a in first[1:]:
        np.testing.assert_allclose(a.basis[:,:2]@a.basis[:,:2].T,
            first[0].basis[:,:2]@first[0].basis[:,:2].T,atol=1e-6)
    second=[s.observe(y,f+rng.normal(size=f.shape),1,2) for s in states]
    for s,a in zip(states,second):
        np.testing.assert_allclose(a.basis.T@a.basis,np.eye(a.basis.shape[1]),atol=2e-6)
        if task=='multiclass': np.testing.assert_allclose(a.basis.sum(axis=0),0,atol=1e-6)
        assert a.diagnostics['observations']==2 and a.diagnostics['future_capture_of_previous'] is not None
        assert a.diagnostics['algorithm_state_bytes']==(40*4 if s.state.kind=='mean' else 4*4)*8
        assert a.diagnostics['diagnostic_state_bytes']==(0 if s.state.kind=='mean' else 2*4*4*8)


def test_supplied_full_basis_recodes_without_hidden_leaf_teacher():
    from test_cumulative_budget_projection import dense,config
    from experiments.cumulative_budget_projection import cumulative_proposal
    from experiments.cumulative_budget_core import CumulativeError
    model=dense(3)
    rng=np.random.default_rng(114)
    x=np.arange(40,dtype=np.float32).reshape(-1,1)%2
    y=rng.integers(0,2,(40,4)).astype(np.float32)
    dev={k:(x.copy(),y.copy()) for k in ('train','calibration','control')}
    source=AbsoluteHistory('history',4,'multilabel').observe(y,np.zeros((40,4)),1,2)
    accepted,info,_=cumulative_proposal(model,dev,100000,CumulativeError(np.zeros((40,4)),'multilabel'),
                                       config(),fixed=True,basis_estimate=source)
    assert accepted is not None and info['singular_values'] is None
    assert info['basis_source']['kind']=='history'
    assert accepted.base.basis.shape[1]==2
    np.testing.assert_allclose(compress(model,np.zeros(4),source.basis).predict(x),model.predict(x),atol=1e-7)
