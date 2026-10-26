import numpy as np
import pytest
from hypothesis import given,settings
from test_online_leaf_core import projection_cases
from experiments.direct_leaf_solver import direct_newton
from experiments.online_leaf_core import weighted_projection


@given(projection_cases())
@settings(max_examples=65,deadline=None)
def test_direct_GH_equals_weighted_projection_at_same_basis_and_support(case):
    target,d,b,fraction=case
    eta=.1
    g=-target*d/eta
    saved=[a.copy() for a in (g,d,b)]
    reference=weighted_projection(target,d,b,fraction)
    actual=direct_newton(g,d,b,eta,reference.rows,reference.columns,block_size=2)
    np.testing.assert_allclose(actual.decoded(b),reference.decoded(b),atol=2e-6,rtol=2e-6)
    assert abs(actual.weighted_error-reference.weighted_error)<=1e-7*(1+reference.weighted_error)
    for before,after in zip(saved,(g,d,b)): np.testing.assert_array_equal(before,after)


def test_full_rank_and_singular_complement_and_all_support():
    g=np.array([[1.,-3.,2.],[.5,.2,-1.]])
    d=np.array([[.2,2.,4.],[2.,.1,.5]])
    target=-.1*g/d
    full=direct_newton(g,d,np.eye(3),.1)
    np.testing.assert_allclose(full.decoded(np.eye(3)),target,atol=1e-8,rtol=1e-6)
    b=np.eye(3)[:,:2]
    rows=np.repeat(np.arange(2),3)
    cols=np.tile(np.arange(3),2)
    all_free=direct_newton(g,d,b,.1,rows,cols)
    np.testing.assert_allclose(all_free.decoded(b),target,atol=1e-8,rtol=1e-6)
    singular=direct_newton(g,d,b,.1,np.array([0,1]),np.array([0,0]))
    assert np.isfinite(singular.decoded(b)).all()
    np.testing.assert_array_equal(singular.coefficients[:,0],0)


def test_fixed_support_matches_independent_explicit_design_lstsq():
    rng=np.random.default_rng(3)
    b=np.linalg.qr(rng.normal(size=(6,3)))[0]
    g=rng.normal(size=(2,6))
    d=rng.uniform(.3,4,size=g.shape)
    rows,cols=np.array([0,0,1]),np.array([2,4,3])
    actual=direct_newton(g,d,b,.2,rows,cols)
    for leaf in range(2):
        support=cols[rows==leaf]
        design=np.column_stack((b,np.eye(6)[:,support]))
        root=np.sqrt(d[leaf])
        solved=np.linalg.lstsq(root[:,None]*design,root*(-.2*g[leaf]/d[leaf]),rcond=1e-12)[0]
        np.testing.assert_allclose(actual.decoded(b)[leaf],design@solved,atol=2e-7,rtol=1e-6)


@pytest.mark.parametrize('bad',['duplicate','negative','float_index','eta','diagonal','block'])
def test_domain_rejection(bad):
    g,d,b=np.ones((2,3)),np.ones((2,3)),np.eye(3)[:,:2]
    eta,rr,cc,block=.1,np.array([0]),np.array([1]),8
    if bad=='duplicate': rr,cc=np.array([0,0]),np.array([1,1])
    if bad=='negative': rr=np.array([-1])
    if bad=='float_index': rr=np.array([0.])
    if bad=='eta': eta=0
    if bad=='diagonal': d[0,0]=0
    if bad=='block': block=True
    with pytest.raises(ValueError): direct_newton(g,d,b,eta,rr,cc,block)
