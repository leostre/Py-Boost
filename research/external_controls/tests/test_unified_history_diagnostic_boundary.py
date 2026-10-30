from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.special import softmax
from test_abl_quality_core import fixture
from budget_quality_worker import digest
from abl_protected_core import as_protected,append_protected
from abl_budget_revision_core import decode_model,blank
from unified_training_core import canonical
from unified_training import evaluate
from unified_frontier_core import make_folds
from experiments.parallel_leaf_core import prefix
from unified_history_diagnostic import diagnose_case


def test_timing_diagnostic_reuses_one_saved_proposal_and_audits_all_seven_bases(tmp_path):
    x=np.linspace(-1,1,120,dtype=np.float32)[:,None];y=np.arange(120)%4
    dataset=tmp_path/'data.npz';splits=tmp_path/'split.npz';np.savez(dataset,features=x,target=y)
    np.savez(splits,train=np.arange(72),calibration=np.arange(72,84),control=np.arange(84,96),selection=np.arange(96,108),test=np.arange(108,120))
    record=dict(case=dict(dataset='toy',seed=7,task='multiclass',outputs=4,split_file=str(splits),split_sha256=digest(splits)),dataset=dict(path=str(dataset),sha256=digest(dataset)))
    warm_model,_,_,_=fixture(5,4,40);warm=canonical(as_protected(warm_model));warm_path=tmp_path/'warm.npz';warm.save(warm_path)
    inc,_,_,_=fixture(31,4,40);inc=replace(inc,base=replace(inc.base,bias=np.zeros(4,np.float32)));inc_path=tmp_path/'increment.npz';inc.save(inc_path)
    full=canonical(append_protected(warm,inc));full_path=tmp_path/'full.npz';full.save(full_path)
    view,_=decode_model(warm);dense=blank(replace(view.base(),bias=view.bias.copy()));frames=[]
    for t in (0,10,20,30,40):
        raw=np.broadcast_to(warm.bias,(72,4)).copy() if t==0 else as_protected(prefix(dense,t)).raw(x[:72])
        p=tmp_path/f'warm{t}.npz';np.savez_compressed(p,scores=raw);frames.append(dict(path=str(p),sha256=digest(p)))
    p=tmp_path/'after.npz';np.savez_compressed(p,scores=full.raw(x[:72]));frames.append(dict(path=str(p),sha256=digest(p)))
    cfg=dict(lambda_l2=1.,decision_grid=[-1.,0.,1.],decision_shrinkage=[.5,1.]);evdir=tmp_path/'teacher_eval';evdir.mkdir()
    evaluations=evaluate(full.raw(x[72:108]),y[72:108],'multiclass',make_folds(y[72:108],'multiclass',7+20261028),cfg,evdir)
    source=dict(uid='one_full/0',family='one_full',block=0,artifact=str(full_path),sha256=digest(full_path),initial_checkpoint=str(warm_path),fits=[dict(artifact=str(inc_path),sha256=digest(inc_path))],evaluations=evaluations)
    case=dict(case='toy_7',task='multiclass',record=record,models=[source],paths=[dict(method='one_full',history_states=frames)])
    result=diagnose_case(cfg,case,tmp_path/'diagnostic')
    assert result['audit']['status']=='passed' and result['audit']['models']==7
    assert digest(inc_path)==source['fits'][0]['sha256'] and digest(full_path)==source['sha256']
