"""Exercise the saved-choice evaluator with independent metrics and real files."""
import json
from pathlib import Path

import numpy as np
import pytest

import budget_quality_worker
import abl_quality_worker as worker
from abl_quality_evaluation import evaluate_case
from audit_calibration_pilot import independent_quality


def test_postseal_evaluation_preserves_choices_and_missing_budget_comparator(tmp_path,monkeypatch):
    y=np.array([0,1,2,0,1,2,0,1,2])
    base=np.array([[2.,.1,-1.],[.1,1.5,-1.],[0.,.2,1.]])[y]
    raws={'pyboost/full':base,'compact/r1':base+np.array([-.4,.6,.1]),'tabpfn8/frozen8':base*.5}
    predictions={}
    for uid,z in raws.items():
        path=tmp_path/(uid.replace('/','-')+'.npz');np.savez(path,raw=z)
        predictions[uid]=dict(path=str(path),sha256=worker.digest(path))
    monkeypatch.setattr(worker,'case_test_predictions',lambda *a:(y,predictions,[]))
    metadata={
        'pyboost/full':dict(family='pyboost',rank=None,nnz=0,cost_bytes=100),
        'compact/r1':dict(family='compact',rank=1,nnz=1,cost_bytes=50,parent_uid='pyboost/full'),
        'tabpfn8/frozen8':dict(family='tabpfn8',rank=None,nnz=0,cost_bytes=None)}
    scores=dict(loss=.5,f1_macro=.8,f1_weighted=.8,brier=.2)
    def choice(group,uid,budget=True):
        value=dict(plan=0,scheme='pooled',repeat=0,mode='raw',objective='loss',group=group,
            chosen=None if uid is None else dict(uid=uid,policy='raw',validation=scores))
        if budget:value.update(budget_ratio=.5,budget_bytes=60)
        return value
    case=dict(case_id='tiny',record=dict(case=dict(outputs=3)),metas=metadata,
        rows=[dict(uid=uid,plan=0,refit=dict(temperature=1.,raw_shifts=[0.]*3,
            temperature_shifts=[0.]*3)) for uid in metadata],
        budget_choices=[choice('native',None),choice('ab',None),choice('abl','compact/r1')],
        external_choices=[choice('tabpfn8','tabpfn8/frozen8',False)])
    original=json.dumps(case,sort_keys=True);seal=tmp_path/'seal.json';seal.write_text('{}')
    cfg=dict(abl_quality=dict(bootstrap_repeats=12,bootstrap_repeat_indices=[0],bootstrap_modes=['raw']))
    result=evaluate_case(cfg,tmp_path,seal,case)
    evaluated=json.loads(Path(result['path']).read_text())
    hybrid=evaluated['budget_points'][-1]
    expected=independent_quality(y,raws['compact/r1'],dict(temperature=1.,shifts=[0.]*3))
    for name,value in expected.items():assert hybrid['test'][name]==pytest.approx(value,abs=1e-12)
    assert hybrid['uid']=='compact/r1'
    assert 'versus_native' not in hybrid
    assert hybrid['versus_tabpfn8']['reference_uid']=='tabpfn8/frozen8'
    assert len(evaluated['bootstrap_intervals'])==1
    assert evaluated['bootstrap_intervals'][0]['repeats']==12
    assert json.dumps(case,sort_keys=True)==original


def test_preseal_loader_returns_only_train_and_development_targets(tmp_path):
    data=tmp_path/'data.npz';roles=tmp_path/'roles.npz'
    target=np.array([0,1,2,0,1,2,0,1,2,9,9,9])
    np.savez(data,features=np.arange(12,dtype=np.float32)[:,None],target=target)
    np.savez(roles,train=np.arange(3),calibration=np.arange(3,5),control=np.arange(5,7),
        selection=np.arange(7,9),test=np.arange(9,12))
    record=dict(case=dict(split_file=str(roles),split_sha256=worker.digest(roles),outputs=3),
        dataset=dict(path=str(data),sha256=worker.digest(data)))
    loaded=worker.before_test_data(record,{})
    assert set(np.unique(loaded['train_y']))=={0,1,2}
    assert 9 not in loaded['dev_y']
    assert not any('test' in key for key in loaded if key!='ids')
    np.savez(roles,train=np.arange(3),calibration=np.arange(3,5),control=np.arange(5,7),
        selection=np.arange(7,9),test=np.array([8,10,11]))
    record['case']['split_sha256']=worker.digest(roles)
    with pytest.raises(RuntimeError,match='exact disjoint cohort'):
        worker.before_test_data(record,{})
