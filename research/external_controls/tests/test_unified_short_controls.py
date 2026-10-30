from pathlib import Path
import numpy as np
from scipy.special import softmax
from test_abl_quality_core import fixture
from abl_protected_core import as_protected,load_protected
from budget_quality_worker import digest
from experiments.parallel_leaf_core import prefix
from unified_serving import add_short_controls


def test_saved_short_controls_retain_initial_bias_and_all_trees_are_prefixes(tmp_path):
    model,x,_,y=fixture(18,4,40);path=tmp_path/'warm.npz';as_protected(model).save(path)
    case=dict(warmup=dict(artifact=str(path),sha256=digest(path)),task='multiclass',record=dict(case=dict(seed=9)),models=[],audit=dict(status='passed'))
    cfg=dict(decision_grid=[-1.,0,1.],decision_shrinkage=[.5,1.])
    add_short_controls(cfg,case,dict(dev_y=y,dev_x=x),tmp_path/'controls')
    assert len(case['models'])==3 and case['audit']['short_controls']['status']=='passed'
    for item,count in zip(case['models'],(10,20,40)):
        actual=load_protected(item['artifact'])
        expected=as_protected(prefix(model,count)).raw(x)
        assert actual.n_trees==count and item['block']==-1
        np.testing.assert_allclose(softmax(actual.raw(x),axis=1),softmax(expected,axis=1),atol=2e-7,rtol=0)
