import numpy as np
import pytest
from scipy.special import softmax, expit
from sklearn.metrics import log_loss, f1_score
from unified_frontier_core import (Fold,make_folds,validate_folds,sufficient,metrics,
    replay_calibration,frontier,quality_change,choose_policy)


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_fold_cover_is_exact_deterministic_and_disjoint(task):
    rng=np.random.default_rng(5)
    y=np.arange(63)%3 if task=='multiclass' else (rng.random((63,5))<.2).astype(int)
    a=make_folds(y,task,17);assert a==make_folds(y,task,17)
    validate_folds(a,len(y));assert make_folds(y,task,18)!=a
    assert sorted(i for f in a for i in f.validation)==list(range(len(y)))


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_sufficient_statistics_match_sklearn(task):
    rng=np.random.default_rng(19);z=rng.normal(size=(70,4));s=rng.normal(size=4)*.1
    y=np.arange(70)%4 if task=='multiclass' else (rng.random((70,4))<.25).astype(int)
    pred,loss,brier=sufficient(y,z,task,1.,s);q=metrics(y,pred,loss,brier,task,4)
    assert q['f1_macro']==pytest.approx(f1_score(y,pred,average='macro',zero_division=0))
    if task=='multiclass':
        assert q['loss']==pytest.approx(log_loss(y,softmax(z,axis=1)))
    else:
        expected=np.mean([-np.mean(y[:,j]*np.log(expit(z[:,j]))+(1-y[:,j])*np.log1p(-expit(z[:,j]))) for j in range(4)])
        assert q['loss']==pytest.approx(expected)


def test_calibration_cannot_see_its_validation_labels():
    rng=np.random.default_rng(9);z=rng.normal(size=(24,3));y=np.arange(24)%3
    fold=(Fold(tuple(range(16)),tuple(range(16,24))),)
    a=replay_calibration(z,y,'multiclass',fold,[-1,0,1],[0,.5,1],complete=False)
    changed=y.copy();changed[16:]=(changed[16:]+1)%3
    b=replay_calibration(z,changed,'multiclass',fold,[-1,0,1],[0,.5,1],complete=False)
    assert a['states']==b['states'];assert a['scores']!=b['scores']
    with pytest.raises(ValueError):validate_folds((Fold((0,1),(1,2)),),3)


def test_frontier_retains_large_and_low_quality_points_when_nondominated():
    rows=[dict(uid='a',cost_bytes=10,quality={'f1_macro':.1}),dict(uid='b',cost_bytes=20,quality={'f1_macro':.4}),
        dict(uid='c',cost_bytes=200,quality={'f1_macro':.9}),dict(uid='d',cost_bytes=30,quality={'f1_macro':.3})]
    assert frontier(rows)==['a','b','c']
    assert quality_change(.4,.8)==.5 and quality_change(.1,0) is None
    assert choose_policy({'raw':{'f1_macro':.3,'loss':1},'temperature':{'f1_macro':.3,'loss':1}},'f1_macro')=='raw'


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_saved_model_end_to_end_calibration_and_independent_audit(tmp_path,task):
    from dataclasses import replace
    from test_abl_quality_core import fixture
    from abl_protected_core import as_protected,block_cost
    from budget_quality_worker import digest
    from unified_recalibration import evaluate_model,audit_case
    leaf,_,_,_=fixture(9,4,5)
    if task=='multilabel':leaf=replace(leaf,base=replace(leaf.base,task=task))
    model=as_protected(leaf);x=np.linspace(-1,1,66,dtype=np.float32)[:,None]
    raw=model.raw(x);rng=np.random.default_rng(43)
    y=np.arange(len(x))%4 if task=='multiclass' else (rng.random(raw.shape)<.3).astype(int)
    path=tmp_path/'model.npz';model.save(path);cache=tmp_path/'raw.npz';np.savez(cache,raw=raw)
    evaluated=evaluate_model(raw,y,task,[22,22,22],19,[-1.,0,1.],[.5,1.],tmp_path,1)
    assert len(evaluated)==6
    item=dict(artifact=str(path),sha256=digest(path),raw=dict(path=str(cache),sha256=digest(cache)),
        evaluations=evaluated,**block_cost(model))
    audited=audit_case(dict(task=task,models=[item]),dict(dev_x=x,dev_y=y))
    assert audited['status']=='passed'
    assert audited['metric_policy_checks']==(32 if task=='multiclass' else 12)
