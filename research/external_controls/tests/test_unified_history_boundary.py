from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from test_abl_quality_core import fixture
from budget_quality_worker import digest,write
from abl_protected_core import as_protected


@pytest.mark.parametrize("reclaim", [False,True])
def test_real_branch_orchestrator_boundary_and_moment_audit(tmp_path,monkeypatch,reclaim):
    import unified_history_training as worker
    x=np.linspace(-1,1,120,dtype=np.float32)[:,None];y=np.arange(120)%4
    data=tmp_path/'data.npz';split=tmp_path/'split.npz';np.savez(data,features=x,target=y)
    np.savez(split,train=np.arange(72),calibration=np.arange(72,84),control=np.arange(84,96),selection=np.arange(96,108),test=np.arange(108,120))
    case=dict(dataset='toy',seed=7,task='multiclass',outputs=4,split_file=str(split),split_sha256=digest(split))
    manifest=tmp_path/'manifest.json';write(manifest,dict(cases=[case],datasets={'toy':dict(path=str(data),sha256=digest(data))}))
    model,_,_,_=fixture(11,4,40);warm=tmp_path/'warm.npz';as_protected(model).save(warm)
    oldcase=tmp_path/'oldcase.json';write(oldcase,dict(metas={'full/b00':dict(artifact=str(warm),sha256=digest(warm))}))
    old=tmp_path/'old.json';write(old,dict(cases=[dict(case_id='toy_7',path=str(oldcase),sha256=digest(oldcase))]))
    u2=tmp_path/'u2.json';write(u2,dict(status='unified_training_complete'))
    calls=[]
    class Process:
        def __init__(self,args,**kwargs):
            request=json.loads(Path(args[-1]).read_text());model,_,_,_=fixture(request['seed'],4,request['trees'])
            model=replace(model,base=replace(model.base,bias=np.zeros(4,np.float32)))
            artifact=Path(request['output']);model.save(artifact);calls.append(request)
            self.pid=900+len(calls)
            write(request['result_path'],dict(artifact=str(artifact),sha256=digest(artifact),trees=request['trees'],
                fit_started_ns=1,fit_finished_ns=100,pid=self.pid,seed=request['seed']))
        def wait(self,timeout):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',Process)
    def fake_warm(request):
        model,_,_,_=fixture(request['seed'],4,request['trees'])
        model=replace(model,base=replace(model.base,bias=np.zeros(4,np.float32)))
        path=Path(request['output']);model.save(path);calls.append(request)
        return dict(artifact=str(path),sha256=digest(path),trees=request['trees'])
    monkeypatch.setattr(worker,'fit_training_increment',fake_warm)
    # The process boundary is mocked; isolate hot ledger writes from Windows
    # vault/antivirus replace locks. Production Linux atomic writes stay intact.
    def local_write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(worker,'write',local_write)
    monkeypatch.setattr(worker,'check_native_environment',lambda cfg:dict(test_fixture=True))
    monkeypatch.setattr(worker,'METHODS',tuple(m for m in worker.METHODS if m[0] in ('one_current','one_history','two_current_sum','two_history_sum')))
    env=tmp_path/'env.json';write(env,dict(executable='unused-fixture-runtime'))
    cfg=dict(frozen_hasboost_environment=str(env),absolute_roles_manifest=str(manifest),absolute_roles_manifest_sha256=digest(manifest),
        abl_protected_development_completed=dict(path=str(old),sha256=digest(old)),decision_grid=[-1.,0,1.],decision_shrinkage=[.5,1.],lr=.1,lambda_l2=1.,
        unified_history=dict(u2=dict(path=str(u2),sha256=digest(u2)),cases=[['toy',7]],maximum_new_trees=2000,residual_fraction=.02))
    if reclaim:cfg['unified_history']['reclaim_training_fraction']=.8
    worker.run(cfg,tmp_path/'result')
    report=json.loads((tmp_path/'result/report.json').read_text());c=json.loads(Path(report['cases'][0]['path']).read_text())
    if reclaim:assert c['record']['data_reuse']['new_train_rows']>72 and c['record']['data_reuse']['test_indices_unchanged']
    assert report['paid_new_trees']==sum(r['trees'] for r in calls)
    assert c['audit']['status']=='passed' and c['audit']['actual_checkpoint_initializations']>0
    assert all(a['frames']==9 for a in c['audit']['moment_audits'])
    assert c['audit']['overlapping_new_branch_epochs']>0

    # A promoted run may add a control method without refitting completed ones.
    paid_before=report['paid_new_trees'];calls_before=len(calls)
    cfg['unified_history']['prior_completed']=dict(path=str(tmp_path/'result/report.json'),sha256=digest(tmp_path/'result/report.json'))
    monkeypatch.setattr(worker,'METHODS',worker.METHODS+(('one_functional','current',1,1.),))
    worker.run(cfg,tmp_path/'promoted')
    promoted=json.loads((tmp_path/'promoted/report.json').read_text());promoted_case=json.loads(Path(promoted['cases'][0]['path']).read_text())
    assert promoted['paid_new_trees']==sum(r['trees'] for r in calls)
    assert 0<promoted['paid_new_trees']-paid_before<=160
    assert len(calls)>calls_before and len(promoted_case['models'])==len(c['models'])+4
    assert promoted_case['audit']['status']=='passed'
