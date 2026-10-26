import numpy as np
import pytest
from experiments.held_budget_event_core import spectral_view,audit_decision,hold_budget
from test_cumulative_budget_projection import dense
from experiments.shared_leaf_basis import functional_basis,compress
from experiments.uniform_threshold_core import ThresholdModel
from experiments.sparse_leaf_core import leaf_design


@pytest.mark.parametrize('seed',range(25))
def test_exactly_three_causal_revelations_under_noise_and_memory(seed):
    rng=np.random.default_rng(seed)
    ratios=rng.uniform(0,2,size=5)
    blocked=rng.random(5)<.3
    for policy in ('periodic','event'):
        used=False
        audits=[]
        for i in range(5):
            audit,_=audit_decision(i,policy,used,float(ratios[i]),bool(blocked[i]))
            audits.append(audit)
            if audit and i in (1,2,3): used=True
        assert sum(audits)==3 and audits[0] and audits[-1]


def test_warning_spectrum_is_scale_invariant_and_finite():
    z=np.random.default_rng(2).normal(size=(17,5))
    g,n=spectral_view(z,8)
    gg,nn=spectral_view(z*1e3,8)
    np.testing.assert_allclose(g,gg,atol=1e-14)
    assert abs(n-nn)<1e-14
    with pytest.raises(ValueError): audit_decision(2,'event',False,float('nan'),False)
    with pytest.raises(ValueError): audit_decision(True,'event',False,1.,False)


def test_held_projection_has_no_control_input_and_full_basis_replays():
    from dataclasses import replace
    d=dense(3)
    d=replace(d,coefficients=np.random.default_rng(3).normal(0,.1,d.coefficients.shape).astype(np.float32))
    x=(np.arange(40,dtype=np.float32)%2).reshape(-1,1)
    y=np.tile((np.arange(40)%2)[:,None],(1,4)).astype(np.float32)
    design=leaf_design(d.leaf_indices(x),len(d.coefficients))
    mean,b,_=functional_basis(design@d.coefficients,d.task)
    base=compress(d,mean,b)
    old=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(4,np.float32))
    held=hold_budget(d,x,y,old)
    assert held.base.coefficients.shape[1]==4 and not len(held.values)
    np.testing.assert_allclose(held.predict(x),d.predict(x),atol=2e-7,rtol=0)
