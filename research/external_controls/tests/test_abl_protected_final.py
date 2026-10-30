from pathlib import Path
import sys,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from budget_quality_worker import write,digest
from abl_protected_final import run,evaluate
from abl_protected_core import as_protected,block_cost
from test_abl_quality_core import fixture


def test_pilot_cannot_open_test(tmp_path):
    source=tmp_path/'pilot.json';write(source,dict(status='protected_pilot_development_complete',test_accessed=False))
    cfg=dict(abl_protected_development_completed=dict(path=str(source),sha256=digest(source)))
    with pytest.raises(RuntimeError,match='completed protected development required'):run(cfg,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_protected_evaluation_uses_only_frozen_choices_and_marks_uncompressed_prefix(tmp_path):
    teacher,x,_,y=fixture(88,4,3);model=as_protected(teacher);artifact=tmp_path/'model.npz';model.save(artifact)
    data=tmp_path/'data.npz';np.savez(data,features=x,target=y)
    split=tmp_path/'roles.npz';np.savez(split,test=np.arange(17,37))
    trajectory=tmp_path/'trajectory.json';write(trajectory,dict(cycles=[],models=[{}],elapsed_seconds=0.,stop_reason='horizon'))
    uid='full/b00';meta=dict(uid=uid,family='full',artifact=str(artifact),sha256=digest(artifact),ranks=[None],
        compressed_blocks=0,**block_cost(model));choices=[]
    for mode in ('raw','temperature_shifts'):
        for objective in ('loss','f1_macro'):
            for group in ('full_unbounded','native_budget','functional','current','mean','history','deployed'):
                chosen=dict(uid=uid,policy='raw',validation=dict(loss=1.,f1_macro=.1)) if group in ('full_unbounded','native_budget','deployed') else None
                choices.append(dict(mode=mode,objective=objective,group=group,budget_bytes=None if group=='full_unbounded' else 100000,chosen=chosen))
    case=dict(case_id='synthetic',task='multiclass',record=dict(case=dict(split_file=str(split),split_sha256=digest(split)),
        dataset=dict(path=str(data),sha256=digest(data))),budget_bytes=100000,choices=choices,
        rows=[dict(uid=uid,refit=dict(temperature=1.,raw_shifts=[0.]*4,temperature_shifts=[0.]*4))],
        metas={uid:meta,'unselected':dict(artifact='must-not-be-read')},trajectories=[dict(mode='full',path=str(trajectory),sha256=digest(trajectory))])
    p=tmp_path/'case.json';write(p,case);entry=evaluate(dict(path=str(p),sha256=digest(p)),'toy-seal',tmp_path/'out')
    result=json.loads(Path(entry['path']).read_text())
    assert len(result['points'])==28 and set(result['predictions'])=={uid}
    assert all(not p['actual_compressed_increment'] for p in result['points'] if p['status']=='evaluated')
