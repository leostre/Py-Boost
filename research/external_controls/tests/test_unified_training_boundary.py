from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from test_abl_quality_core import fixture
from budget_quality_worker import digest,write


@pytest.mark.parametrize("reclaim", [False,True])
def test_training_orchestration_reuses_only_matching_initial_functions(tmp_path,monkeypatch,reclaim):
    import unified_training as worker
    x=np.linspace(-1,1,120,dtype=np.float32)[:,None];y=np.arange(120)%4
    data=tmp_path/'data.npz';split=tmp_path/'split.npz';np.savez(data,features=x,target=y)
    np.savez(split,train=np.arange(72),calibration=np.arange(72,84),control=np.arange(84,96),selection=np.arange(96,108),test=np.arange(108,120))
    record=dict(dataset='toy',seed=7,task='multiclass',outputs=4,split_file=str(split),split_sha256=digest(split))
    manifest=tmp_path/'manifest.json';write(manifest,dict(cases=[record],datasets={'toy':dict(path=str(data),sha256=digest(data))}))
    prior=tmp_path/'old.json';write(prior,dict(cases=[]))
    u1=tmp_path/'u1.json';write(u1,dict(status='unified_recalibration_complete',cases=[dict(audit=dict(status='passed'))]))
    calls=[]
    def fake_fit(request):
        model,_,_,_=fixture(request['seed'],4,request['trees'])
        model=replace(model,base=replace(model.base,bias=np.zeros(4,np.float32)))
        p=Path(request['output']);model.save(p);calls.append(request)
        return dict(artifact=str(p),sha256=digest(p),trees=request['trees'],train_replay_error=0.,validation_replay_error=0.)
    def local_write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(worker,'write',local_write)
    monkeypatch.setattr(worker,'fit_training_increment',fake_fit)
    monkeypatch.setattr(worker,'check_native_environment',lambda cfg:dict(test_fixture=True))
    cfg=dict(absolute_roles_manifest=str(manifest),absolute_roles_manifest_sha256=digest(manifest),
        abl_protected_development_completed=dict(path=str(prior),sha256=digest(prior)),decision_grid=[-1.,0,1.],decision_shrinkage=[.5,1.],lr=.1,
        unified_training=dict(u1=dict(path=str(u1),sha256=digest(u1)),cases=[['toy',7]],memory_ratios=[.7],maximum_new_trees=1000,maximum_development_candidates=3))
    if reclaim:cfg['unified_training'].update(reclaim_training_fraction=.8,fresh_warmup=True)
    worker.run(cfg,tmp_path/'result')
    report=json.loads((tmp_path/'result/report.json').read_text());case=json.loads(Path(report['cases'][0]['path']).read_text())
    if reclaim:
        assert case['record']['data_reuse']['test_indices_unchanged']
        assert case['record']['data_reuse']['new_train_rows']>72
    assert report['status']=='unified_training_complete' and not report['test_evaluated']
    assert report['paid_new_trees']==sum(r['trees'] for r in calls)
    assert case['audit']['status']=='passed' and case['audit']['actual_checkpoint_initialization_checks']>0
    assert any(r['family']=='posthoc' for r in case['models']) and any(r['family']=='online' for r in case['models'])
    # Promoting a completed pilot must not fit or count it a second time.
    cfg['unified_training']['prior_completed']=dict(path=str(tmp_path/'result/report.json'),sha256=digest(tmp_path/'result/report.json'))
    before=len(calls);worker.run(cfg,tmp_path/'promoted')
    promoted=json.loads((tmp_path/'promoted/report.json').read_text())
    assert len(calls)==before and promoted['paid_new_trees']==report['paid_new_trees']
