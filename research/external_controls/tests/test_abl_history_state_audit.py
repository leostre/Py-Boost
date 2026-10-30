from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_abl_quality_core import fixture
from budget_quality_worker import write,digest
from experiments.parallel_leaf_core import prefix
from experiments.absolute_history_core import AbsoluteHistory
from abl_history_state_audit import audit_history_states


def test_independent_temporal_moments_replay_all_three_estimators(tmp_path):
    teacher,x,_,y=fixture(50,4,40);artifact=tmp_path/'teacher.npz';teacher.save(artifact)
    data=tmp_path/'data.npz';np.savez(data,features=x,target=y)
    split=tmp_path/'split.npz';np.savez(split,train=np.arange(20),calibration=np.arange(20,25),control=np.arange(25,30),
        selection=np.arange(30,34),test=np.arange(34,37))
    warm=tmp_path/'warm.json';write(warm,dict(warmup_threshold=dict(artifact=str(artifact),sha256=digest(artifact))))
    trajectories=[]
    for kind in ('current','mean','history'):
        history=AbsoluteHistory(kind,4,'multiclass',.8)
        for trees in (0,10,20,30,40):
            scores=np.broadcast_to(teacher.base.bias,(20,4)).copy() if not trees else prefix(teacher,trees).prepare_inference().raw(x[:20])
            estimate=history.observe(y[:20],scores,1.,1)
        s=history.state;path=tmp_path/(kind+'.npz');np.savez(path,mean=np.empty((0,0)) if s.mean is None else s.mean,
            gram=np.empty((0,0)) if s.gram is None else s.gram,mass=np.array(s.mass),observations=np.array(s.observations),basis=estimate.basis)
        p=tmp_path/(kind+'.json');write(p,dict(cycles=[dict(block=1,history_state_path=str(path),history_state_sha256=digest(path),
            algorithm_state_bytes=estimate.diagnostics['algorithm_state_bytes'])]))
        trajectories.append(dict(mode=kind,path=str(p),sha256=digest(p)))
    case=tmp_path/'case.json';write(case,dict(case_id='synthetic',task='multiclass',record=dict(
        case=dict(task='multiclass',split_file=str(split),split_sha256=digest(split),outputs=4),
        dataset=dict(path=str(data),sha256=digest(data))),warmup=dict(path=str(warm),sha256=digest(warm)),trajectories=trajectories))
    result=audit_history_states([dict(path=str(case),sha256=digest(case))])
    assert result['status']=='passed' and len(result['checks'])==3
