import numpy as np
import pytest
from budget_quality_core import probabilities, quality, fit_temperature, select, paired_bootstrap, variants


@pytest.mark.parametrize('classes,n,scale', [(2,7,.01),(3,21,1),(11,39,100)])
def test_temperature_argmax_and_identity(classes,n,scale):
    z = np.random.default_rng(classes).normal(size=(n,classes))*scale
    y = np.arange(n)%classes
    for t in (.1,.37,1.,3.,10.):
        assert np.array_equal(z.argmax(1), (z/t).argmax(1))
        assert quality(y,z,t)['f1_macro'] == quality(y,z)['f1_macro']
        np.testing.assert_allclose(probabilities(z,t).sum(1),1)
    fit=fit_temperature(z,y)
    assert .1-1e-7 <= fit['temperature'] <=10
    assert fit['calibration_loss'] <= fit['identity_loss']+1e-10


def test_shift_affects_decisions_not_probability_metrics():
    z=np.array([[2.,0.],[0.,2.],[1.,0.]])
    y=np.array([0,1,0])
    a,b=quality(y,z),quality(y,z,1.,[3.,0.])
    assert a['loss']==b['loss'] and a['brier']==b['brier']
    assert a['f1_macro'] != b['f1_macro']


def test_domain_includes_absent_class_and_budget_boundary():
    assert quality([0,0],np.array([[2.,0.,0.],[2.,0.,0.]]))['f1_macro']==pytest.approx(1/3)
    rows=[dict(uid='a',cost_bytes=100,selection=dict(loss=.2,f1_macro=.8)),
          dict(uid='b',cost_bytes=101,selection=dict(loss=.1,f1_macro=.7))]
    assert select(rows,'loss',99) is None
    assert select(rows,'loss',100)['uid']=='a'
    assert select(rows,'loss',101)['uid']=='b'
    assert select(rows[::-1],'f1_macro',101)['uid']=='a'


def test_bootstrap_pairing_identity_and_determinism():
    z=np.array([[2.,0.],[0.,2.],[1.,0.]])
    a=paired_bootstrap(np.array([0,1,0]),(z,1.,[0.,0.]),(z,1.,[0.,0.]),25)
    assert a['delta_loss_ci95']==[0.,0.]
    assert a['delta_f1_pp_ci95']==[0.,0.]
    assert a==paired_bootstrap(np.array([0,1,0]),(z,1.,[0.,0.]),(z,1.,[0.,0.]),25)


def test_roles_are_used_for_distinct_fits():
    logits={r:np.tile([[1.,0.]],(n,1)) for r,n in [('calibration',7),('control',5),('selection',9)]}
    targets={r:np.arange(len(z))%2 for r,z in logits.items()}
    calls=[]
    def fitter(z,y,task,grid):
        calls.append((len(z),len(y),task))
        return np.zeros(2,np.float32)
    out=variants(logits,targets,fitter,[0,.25],[.25,.5,1.])
    assert calls==[(5,5,'multiclass')]*2
    assert len(out)==10
    assert {r['policy'] for r in out}=={'raw','temperature','raw_shifts','temperature_shifts'}


@pytest.mark.parametrize('t',[0,-1,np.nan,np.inf])
def test_invalid_temperature(t):
    with pytest.raises(ValueError): probabilities([[1,0]],t)
