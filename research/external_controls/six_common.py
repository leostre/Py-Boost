"""Shared sealed-data, exact-byte and GPU prediction shell for R1-R6."""
from pathlib import Path
import gc,json
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_protected_core import load_protected,block_cost
from abl_transfer_study import data_before_test
from packed_serving_core import prepare_snapshot,PackedPlan
from broad_core import assess,grouped_folds

def price(model):
    return prepare_snapshot(model,np.zeros(len(model.bias),np.float32)).resident_bytes+8

def gpu_raw(model,x,batch=1024):
    import cupy as cp
    from packed_serving_gpu import PackedGpu
    c=len(model.bias);width=max(b.base.coefficients.shape[1] for b in model.blocks);has=any(len(b.values) for b in model.blocks)
    groups=next(g for g in (4,2,1) if 8*(c+g*width+g*c*int(has)+32*g)<=49152)
    device=PackedGpu(model,1.,np.zeros(c,np.float32),groups,'aos');out=np.empty((len(x),c),float)
    for start in range(0,len(x),batch):
        dx=cp.asarray(np.ascontiguousarray(x[start:start+batch]));buffers=device.buffers(len(dx));device.launch_into(dx,*buffers,raw=True)
        out[start:start+len(dx)]=buffers[0].get()
        del dx,buffers
    del device;gc.collect();cp.get_default_memory_pool().free_all_blocks();return out

def load_cases(cfg,first=False):
    source=locked(cfg['six']['training']['path'],cfg['six']['training']['sha256']);seen=set();cases=[]
    for ref in source['cases']:
        case=locked(ref['path'],ref['sha256']);name=case['record']['case']['dataset']
        if first and name in seen:continue
        seen.add(name);cases.append(case)
    return source,cases

def load_model(m):
    assert digest(m['artifact'])==m['sha256'];return load_protected(m['artifact'])

def store_model(model,uid,family,data,cfg,case,out,**meta):
    out.mkdir(parents=True,exist_ok=True);raw=gpu_raw(model,data['dev_x'])
    np.testing.assert_allclose(raw[:13],model.raw(data['dev_x'][:13]),atol=1e-8,rtol=1e-10)
    folds=grouped_folds(data['dev_x'],case['record']['case']['seed']+20261029)
    assessment,arrays=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
    artifact=out/'model.npz';model.save(artifact);cache=out/'raw.npz';np.savez_compressed(cache,raw=raw)
    m=dict(uid=uid,family=family,artifact=str(artifact),sha256=digest(artifact),assessment=assessment,packed_bytes=price(model),raw=dict(path=str(cache),sha256=digest(cache)),**block_cost(model),**meta)
    write(out/'model.json',m);return m

def test_case(case,candidates,out):
    from residual_gate_core import sufficient,quality
    record=case['record'];cc,dd=record['case'],record['dataset']
    assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
    with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
    results=[]
    for m in candidates:
        model=load_model(m);raw=gpu_raw(model,x);control=m['assessment']['refit_control']
        q=quality(y,*sufficient(y,raw,'multilabel',control),'multilabel',len(model.bias))
        results.append(dict(uid=m['uid'],family=m['family'],quality=q,D_quality=m['assessment']['quality'],packed_bytes=m['packed_bytes'],trees=model.n_trees,sha256=m['sha256']))
        del model,raw
    write(out,dict(case=case['case'],models=results));return results
