import numpy as np
import pytest
from hypothesis import given,settings,strategies as st

from experiments.gradient_basis_core import (GradientBasisState,basis_from_gram,gradient_signal,
    projector_overlap,captured_energy)


@given(st.lists(st.tuples(st.floats(-2,2,allow_nan=False,allow_infinity=False),
                          st.floats(-2,2,allow_nan=False,allow_infinity=False)),min_size=1,max_size=12),
       st.sampled_from([0.,.2,.8,.95]))
@settings(max_examples=45,deadline=None)
def test_history_recurrence_matches_explicit_weighted_tensor_and_covariance_is_psd(rows,beta):
    mean,history,current = [GradientBasisState(k,beta) for k in ('mean','history','current')]
    snapshots = []
    for a,b in rows:
        z = np.array([[a,b],[b,-a],[a+b,a-b]])
        snapshots.append(z)
        for state in (mean,history,current):
            state.observe(z)
        weights = np.array([(1-beta)*beta**i for i in range(len(snapshots)-1,-1,-1)])
        weights /= weights.sum()
        explicit_mean = sum(w*z0 for w,z0 in zip(weights,snapshots))
        explicit_gram = sum(w*(z0.T@z0) for w,z0 in zip(weights,snapshots))
        np.testing.assert_allclose(mean.matrices()[0],explicit_mean.T@explicit_mean,atol=2e-12,rtol=1e-12)
        np.testing.assert_allclose(history.matrices()[0],explicit_gram,atol=2e-12,rtol=1e-12)
        np.testing.assert_allclose(current.matrices()[0],z.T@z,atol=2e-12,rtol=1e-12)
        covariance = explicit_gram-explicit_mean.T@explicit_mean
        assert np.linalg.eigvalsh((covariance+covariance.T)/2)[0] >= -1e-10
        for state in (mean,history,current):
            whole,halves = state.matrices()
            np.testing.assert_allclose(whole,halves[0]+halves[1],atol=2e-12,rtol=1e-12)


def test_sign_cancellation_retained_by_tensor_gram_not_mean():
    z = np.array([[1.,0],[2.,0]])
    # beta .5 gives a two-observation first-moment cancellation at second=-.5first.
    mean,history = GradientBasisState('mean',.5),GradientBasisState('history',.5)
    for value in (z,-.5*z):
        mean.observe(value)
        history.observe(value)
    previous = np.array([[0.],[1.]],np.float32)
    mean_basis,info = mean.estimate('multilabel',1,previous)
    history_basis,_ = history.estimate('multilabel',1,previous)
    np.testing.assert_array_equal(mean_basis,previous)
    assert info['status'] == 'zero_signal_keep_previous'
    assert projector_overlap(history_basis,np.array([[1.],[0.]])) == pytest.approx(1.)


def test_projector_metric_ignores_sign_and_coordinate_rotation():
    a = np.eye(4)[:,:2]
    rotated = a@np.array([[0.,-1.],[1.,0.]])
    assert projector_overlap(a,rotated) == pytest.approx(1.)
    assert projector_overlap(a,np.eye(4)[:,2:]) == pytest.approx(0.)
    assert captured_energy(np.eye(4),a) == pytest.approx(.5)


@pytest.mark.parametrize('task', ['multilabel','multiclass'])
def test_signal_scaling_and_contrast_and_rank_contract(task):
    g = np.array([[1.,-2.,3.],[-2.,1.,-1.],[.5,.1,-.2]])
    h = np.array([[.1,.2,.3],[.5,.4,.1],[.3,.2,.1]])
    z = gradient_signal(g,h,1.,task)
    np.testing.assert_allclose(z,gradient_signal(3*g,h,1.,task),atol=1e-14,rtol=0)
    assert np.linalg.norm(z) == pytest.approx(1.)
    if task == 'multiclass':
        np.testing.assert_allclose(z.sum(axis=1),0.,atol=1e-14)
        previous = np.linalg.qr(np.vstack((np.eye(2),-np.ones(2))))[0].astype(np.float32)[:,:1]
    else:
        previous = np.eye(3,dtype=np.float32)[:,:1]
    basis,_ = basis_from_gram(z.T@z,task,1,previous)
    np.testing.assert_allclose(basis.T@basis,np.eye(1),atol=1e-6)
    if task == 'multiclass':
        np.testing.assert_allclose(basis.sum(axis=0),0.,atol=1e-6)
    assert captured_energy(z,basis) >= captured_energy(z,previous)-1e-7


def test_state_rejects_axis_drift_and_empty_history_without_mutation():
    state = GradientBasisState('history')
    with pytest.raises(ValueError):
        state.matrices()
    state.observe(np.eye(3))
    before = state.matrices()[0].copy()
    for bad in (np.eye(4),np.full((3,3),np.nan)):
        with pytest.raises(ValueError):
            state.observe(bad)
    np.testing.assert_array_equal(state.matrices()[0],before)
    assert state.observations == 1
