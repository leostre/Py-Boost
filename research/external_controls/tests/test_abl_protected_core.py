from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import budget_quality_worker
from test_abl_quality_core import fixture
from abl_protected_core import as_protected,append_protected,block_cost,project_new_increment,load_protected
from abl_quality_core import fingerprint
from budget_quality_core import probabilities


def test_protected_append_does_not_reencode_old_blocks_and_roundtrips(tmp_path):
    old,x,_,_=fixture(30,5,3);inc,_,_,_=fixture(40,5,2)
    checkpoint=as_protected(old);before=[fingerprint(b) for b in checkpoint.blocks]
    result=append_protected(checkpoint,inc)
    assert [fingerprint(b) for b in result.blocks[:len(before)]]==before
    expected=checkpoint.raw(x)+inc.prepare_inference().raw(x)
    np.testing.assert_allclose(result.raw(x),expected,atol=2e-7,rtol=0)
    p=tmp_path/'protected.npz';result.save(p)
    np.testing.assert_array_equal(load_protected(p).raw(x),result.raw(x))
    price=block_cost(result)
    assert price['cost_bytes']>=price['coo_resident_bytes']
    assert price['cost_bytes']>=price['prepared_blocks_bytes']


def test_projection_recovers_rank_one_increment_and_rejects_impossible_budget_without_mutation():
    old,x,_,_=fixture(11,4,3);inc,_,_,_=fixture(21,4,2)
    coefficients=np.arange(len(inc.base.coefficients),dtype=np.float32)[:,None]*np.array([[.1,0,0,-.1]],np.float32)
    inc=replace(inc,base=replace(inc.base,coefficients=coefficients,bias=np.zeros(4,np.float32)))
    checkpoint=as_protected(old);before=[fingerprint(b) for b in checkpoint.blocks]
    raw=checkpoint.raw(x)+inc.prepare_inference().raw(x);y=raw.argmax(1)
    development=dict(train=(x[:20],y[:20]),calibration=(x[20:28],y[20:28]),control=(x[28:],y[28:]))
    cfg=dict(margin_scale=.5,decision_grid=[0,.5,-.5,1,-1,2,-2],decision_shrinkage=[.25,.5,1.])
    model,info=project_new_increment(checkpoint,inc,development,100000,cfg)
    assert model is not None and info['chosen']['rank']==1
    np.testing.assert_allclose(probabilities(model.raw(x)),probabilities(raw),atol=2e-7,rtol=0)
    rejected,why=project_new_increment(checkpoint,inc,development,1,cfg)
    assert rejected is None and why['status']=='no_memory_or_local_quality_candidate'
    assert [fingerprint(b) for b in checkpoint.blocks]==before
