from pathlib import Path
import numpy as np
import pytest
from budget_quality_worker import digest
from unified_frontier_core import make_folds
from unified_training import evaluate
from unified_evaluation_audit import audit_evaluations


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_independent_audit_detects_corrupted_predictions_even_with_updated_hash(task,tmp_path):
    rng=np.random.default_rng(16);raw=rng.normal(size=(60,4));y=np.arange(60)%4 if task=='multiclass' else (rng.random((60,4))<.3).astype(int)
    cfg=dict(decision_grid=[-1.,0.,1.],decision_shrinkage=[.5,1.]);folds=make_folds(y,task,9)
    ev=evaluate(raw,y,task,folds,cfg,tmp_path)
    assert audit_evaluations(raw,y,task,ev)['status']=='passed'
    p=Path(ev[0]['evaluation']['path'])
    with np.load(p,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    name=next(k for k in arrays if k.endswith('__prediction'));a=arrays[name]
    if a.ndim==1:a[0]=(a[0]+1)%4
    else:a[0,0]=not a[0,0]
    np.savez_compressed(p,**arrays)
    for e in ev:e['evaluation']['sha256']=digest(p)
    with pytest.raises(AssertionError):audit_evaluations(raw,y,task,ev)
