"""Conservation, leakage, numerical parity and independent metric oracles."""
import json
from dataclasses import replace

import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss

from budget_quality_core import fit_temperature, quality
from calibration_pilot_core import (Scheme, POLICIES, Control, make_plans, partition,
    validate_plan, fast_temperature, decision_shifts, evaluate_plan, choose_policy,
    choose_model, control_for, fit_controls, stratified_order)


@pytest.mark.parametrize('sizes',[(11,9,13),(2,3,4),(15,13,20)])
@pytest.mark.parametrize('repeat',[0,1,9])
def test_plans_conserve_rows_disjointness_and_nested_samples(sizes,repeat):
    n=sum(sizes);y=np.arange(n)%5
    plans=make_plans(y,sizes,repeat)
    assert plans==make_plans(y.copy(),sizes,repeat)
    for p in plans:
        assert validate_plan(p,n)==p
        for f in p.folds:
            assert not set(f.validation)&(set(f.temperature)|set(f.shifts))
        if p.scheme==Scheme.CROSSFIT:
            assert sorted(i for f in p.folds for i in f.validation)==list(range(n))
    for scheme in Scheme:
        a,b,c=[p for p in plans if p.scheme==scheme]
        for fa,fb,fc in zip(a.folds,b.folds,c.folds):
            assert set(fa.temperature)<=set(fb.temperature)<=set(fc.temperature)
            assert fa.validation==fb.validation==fc.validation
    for fraction in (.25,.5,1.):
        a,b=[p for p in plans if p.fraction==fraction and p.scheme!=Scheme.CROSSFIT]
        assert set(a.folds[0].temperature)|set(a.folds[0].shifts)==set(b.folds[0].temperature)
        assert a.folds[0].validation==b.folds[0].validation


def test_rare_singleton_is_not_duplicated_or_lost():
    y=np.array([0]*12+[1]*5+[2])
    groups=partition(y,[6,6,6],41)
    assert sorted(i for group in groups for i in group)==list(range(18))
    assert sum(17 in group for group in groups)==1
    for seed in (1,2,3):
        assert sorted(stratified_order(y,seed))==list(range(18))


def test_leakage_duplicate_and_incomplete_oof_rejected():
    p=next(p for p in make_plans(np.arange(30)%3,[8,10,12],0) if p.scheme==Scheme.CROSSFIT)
    f=p.folds[0]
    leaking=replace(f,temperature=(f.validation[0],))
    with pytest.raises(ValueError,match='leakage'):
        validate_plan(replace(p,folds=(leaking,*p.folds[1:])),30)
    with pytest.raises(ValueError,match='cover'):
        validate_plan(replace(p,folds=p.folds[:2]),30)
    with pytest.raises(ValueError,match='overlap'):
        validate_plan(replace(p,folds=(f,f,*p.folds[1:])),30)


@pytest.mark.parametrize('classes,n,scale',[(2,17,.1),(3,31,1),(11,80,10)])
def test_fast_temperature_matches_frozen_objective(classes,n,scale):
    z=np.random.default_rng(classes).normal(size=(n,classes))*scale;y=np.arange(n)%classes
    t=fast_temperature(z,y);old=fit_temperature(z,y)['temperature']
    assert t==pytest.approx(old,rel=2e-5,abs=2e-6)
    assert quality(y,z,t)['loss']<=quality(y,z)['loss']+1e-12
    assert quality(y,z,t)['f1_macro']==quality(y,z)['f1_macro']


def scalar_shift_oracle(z,y,grid):
    z=z.astype(float)-z.mean(axis=1,keepdims=True)
    output=[]
    for j in range(z.shape[1]):
        others=np.delete(z,j,axis=1).max(axis=1)
        truth=y==j
        if truth.sum() in (0,len(y)):
            output.append(0.);continue
        thresholds=sorted(set(grid),key=lambda t:(abs(t),-t))
        scores=[f1_score(truth,z[:,j]-others>=t,zero_division=0) for t in thresholds]
        output.append(thresholds[np.argmax(scores)])
    return np.array(output,np.float32)


@pytest.mark.parametrize('classes,n',[(2,17),(3,31),(7,55)])
def test_vector_decisions_match_independent_classwise_oracle(classes,n):
    z=np.random.default_rng(classes).normal(size=(n,classes));y=np.arange(n)%classes
    grid=[0,.25,-.25,.5,-.5,1,-1,2,-2]
    np.testing.assert_array_equal(decision_shifts(z,y,grid),scalar_shift_oracle(z,y,grid))
    y[:]=0
    np.testing.assert_array_equal(decision_shifts(z,y,grid),np.zeros(classes))


def test_oof_metrics_use_concatenation_not_fold_macro_average():
    rng=np.random.default_rng(2);z=rng.normal(size=(43,5));y=np.arange(43)%4
    p=next(p for p in make_plans(y,[11,13,19],1) if p.scheme==Scheme.CROSSFIT and p.fraction==1.)
    scores,_=evaluate_plan(z,y,p,[0,.25,-.25,1,-1])
    expected=f1_score(y,z.argmax(1),labels=np.arange(5),average='macro',zero_division=0)
    assert scores['raw']['f1_macro']==pytest.approx(expected)
    assert scores['temperature']['f1_macro']==scores['raw']['f1_macro']
    assert scores['raw']['loss']==pytest.approx(quality(y,z)['loss'])
    for name in scores:
        if 'shift' in name:
            probability='temperature' if name.startswith('temperature') else 'raw'
            assert scores[name]['loss']==scores[probability]['loss']
            assert scores[name]['brier']==scores[probability]['brier']


def test_calibrator_cannot_see_heldout_labels(monkeypatch):
    import calibration_pilot_core as core
    z=np.random.default_rng(8).normal(size=(30,3));y=np.arange(30)%3
    p=make_plans(y,[8,10,12],0)[0];calls=[]
    old=core.fit_controls
    def recording(raw,labels,temp,shift,grid):
        calls.append((set(temp),set(shift)))
        return old(raw,labels,temp,shift,grid)
    monkeypatch.setattr(core,'fit_controls',recording)
    core.evaluate_plan(z,y,p,[0,.25,-.25])
    assert len(calls)==1
    assert not (calls[0][0]|calls[0][1])&set(p.folds[0].validation)
    changed=y.copy();changed[list(p.folds[0].validation)]=(changed[list(p.folds[0].validation)]+1)%3
    a=old(z,y,p.refit_temperature,p.refit_shifts,[0,.25,-.25])
    b=old(z,changed,p.refit_temperature,p.refit_shifts,[0,.25,-.25])
    assert a[0]==b[0];np.testing.assert_array_equal(a[1],b[1])


def test_selection_budget_identity_and_serialization():
    scores={p.name:dict(loss=.5,f1_macro=.5) for p in POLICIES}
    assert choose_policy(scores,'loss')=='raw'
    assert choose_policy(scores,'f1_macro')=='raw'
    scores['temperature']['loss']=.6
    assert choose_policy(scores,'loss')=='raw'
    rows=[dict(model_id='a',cost_bytes=100,validation=dict(loss=.3,f1_macro=.6)),
          dict(model_id='b',cost_bytes=101,validation=dict(loss=.2,f1_macro=.7))]
    assert choose_model(rows,'loss',99) is None
    assert choose_model(rows,'loss',100)['model_id']=='a'
    assert choose_model(rows[::-1],'f1_macro',101)['model_id']=='b'
    control=Control(float(np.float32(.7)),(.0,float(np.float32(.25))))
    encoded=json.loads(json.dumps(dict(temperature=control.temperature,shifts=control.shifts)))
    assert Control(encoded['temperature'],tuple(encoded['shifts']))==control


def test_test_loader_requires_complete_immutable_seal(tmp_path):
    from calibration_pilot_worker import require_seal
    path=tmp_path/'seal.json';path.write_text(json.dumps(dict(status='incomplete')))
    with pytest.raises(RuntimeError,match='seal'):
        require_seal(path)
    development=tmp_path/'dev.json';development.write_text('{}')
    path.write_text(json.dumps(dict(status='calibration_choices_sealed_before_test',
        cases=[dict(path=str(development),sha256='invalid')])))
    with pytest.raises(RuntimeError,match='changed'):
        require_seal(path)
