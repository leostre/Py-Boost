import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import budget_quality_worker
from adaptive_basis_core import risk_basis
from abl_budget_revision_core import fixed_support_projection


@settings(max_examples=30, deadline=None)
@given(st.integers(1,20), st.integers(1,10), st.integers(0,10000))
def test_identity_minimum_norm_against_diagonal_reference(c,n,seed):
    rng=np.random.default_rng(seed);v=rng.normal(size=(n,c));w=10**rng.uniform(-15,3,(n,c))
    keys=np.flatnonzero(rng.random((n,c)).ravel()<.2)
    a,rr,cc,values=fixed_support_projection(v,w,np.eye(c),keys)
    free=w.copy();free.ravel()[keys]=0
    expected=[]
    for row,weights in zip(v,free):
        # Independent generic Moore-Penrose solve, same relative eigen cutoff.
        expected.append(np.linalg.pinv(np.diag(weights),rcond=1e-12)@(weights*row))
    np.testing.assert_allclose(a,np.array(expected).astype(np.float32),atol=2e-7,rtol=1e-6)
    decoded=a.astype(float);decoded[rr,cc]+=values
    np.testing.assert_allclose(decoded.ravel()[keys],v.ravel()[keys],atol=2e-7,rtol=1e-6)


@pytest.mark.parametrize('c,r',[(6,3),(19,4),(71,16)])
def test_truncated_risk_basis_matches_full_prefix_subspaces(c,r):
    rng=np.random.default_rng(20+c);z=rng.normal(size=(c+80,c));w=np.exp(rng.normal(size=len(z))*.2);col=np.exp(rng.normal(size=c)*.4)
    full,_=risk_basis(z,'multilabel',w,col)
    part,meta=risk_basis(z,'multilabel',w,col,r)
    for k in [1,r]:
        np.testing.assert_allclose(full[:,:k]@full[:,:k].T,part[:,:k]@part[:,:k].T,atol=2e-6)
    assert meta['truncated_rank']==r


def test_exact_identity_avoids_eigensolver(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('unnecessary eigensolver')
    monkeypatch.setattr(np.linalg,'eigh',forbidden)
    a,*_=fixed_support_projection(np.ones((3,983)),np.ones((3,983)),np.eye(983))
    assert a.shape==(3,983)


def test_explicit_source_alias_never_drops_or_positionally_guesses_labels():
    from broad_data import label_columns
    names=['feature',"TAG_m'usica",'other'];labels=["TTAG_m\\'usica",'other']
    aliases={"TTAG_m\\'usica":"TAG_m'usica"}
    assert label_columns(names,labels,aliases)==[1,2]
    with pytest.raises(ValueError):label_columns(names,labels)
    with pytest.raises(ValueError):label_columns(names,labels,{labels[0]:'other'})
