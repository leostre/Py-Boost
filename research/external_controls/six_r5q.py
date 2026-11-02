"""Limited FQTree-inspired quantized-forward continuation, not NN STE training."""
from pathlib import Path
import gc,json,sys,time
import numpy as np
from budget_quality_worker import write,locked,digest,event
from abl_quality_worker import check_native_environment
from six_common import load_cases,load_model
from six_r5 import quant_raw
from six_quant_core import prepare_quant,QuantSnapshot
from abl_transfer_study import data_before_test
from broad_core import model_prefix,assess,grouped_folds
from abl_protected_core import append_protected
from residual_gate_core import sufficient,quality


def fit(case,data,initial,basis,importance,codec,scales,qat,cfg,path):
    import cupy as cp
    from direct_leaf_gpu import DirectLeafBoosting
    from quantized_direct_coordinate_gpu import QuantizedDirectLeafBoosting
    from experiments.a100_runner import build_model
    from experiments.inference_gpu_guard import gpu_probe
    ready,reason,probe=gpu_probe('shared',8192)
    if not ready:raise RuntimeError(reason)
    initial_raw=quant_raw(initial,data['train_x']).astype(np.float32);val=quant_raw(initial,data['dev_x'])
    n=80;seed=case['record']['case']['seed']+105000
    native=build_model('full','multilabel',len(basis),cfg,seed,3,cfg['lambda_l2'],trees=n)
    args=dict(native.params);args.update(verbose=200,use_hess=True,gd_steps=1)
    cls=QuantizedDirectLeafBoosting if qat else DirectLeafBoosting
    extra=dict(forward_codec=codec,fixed_a=scales) if qat else {}
    trainer=cls(**args,initial_train=initial_raw,initial_validation=[val],basis=basis,task='multilabel',residual_fraction=.02,
                class_importance=importance,full_coordinates=False,chunk_rows=1024,pool_limit_bytes=12*1024**3,cumulative_quota=True,**extra)
    cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
    trainer.fit(data['train_x'],data['train_y'],eval_sets=[dict(X=data['dev_x'],y=data['dev_y'])]);cp.cuda.get_current_stream().synchronize()
    proxy=trainer.export_increment();proxy.save(path);increment_quant=prepare_quant(append_protected_placeholder(proxy),np.zeros(len(basis),np.float32),codec,fixed_a=scales)
    replay=val+quant_raw(increment_quant,data['dev_x']);
    if qat:
        from scipy.special import expit
        err=float(np.max(np.abs(expit(replay)-expit(trainer.validation_final_raw[0]))));assert err<2e-5,err
    else:err=None
    record=dict(trees=n,seconds=time.perf_counter()-start,qat=qat,codec=codec,seed=seed,proxy_sha256=digest(path),
                replay_probability_error=err,clipped=int(getattr(trainer,'quant_clipped',0)),coefficients=int(getattr(trainer,'quant_total',0)),gpu_probe=probe)
    del trainer,native;gc.collect();cp.get_default_memory_pool().free_all_blocks();return proxy,record


def append_protected_placeholder(inc):
    from abl_protected_core import as_protected
    return as_protected(inc)


def run(cfg,out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3);out.mkdir(parents=True,exist_ok=True);env=check_native_environment(cfg)
    _,cases=load_cases(cfg,first=True);r2=locked(cfg['six']['R2']['path'],cfg['six']['R2']['sha256']);by={r['case']:r for r in r2['cases']}
    ptq=locked(cfg['six']['R5']['path'],cfg['six']['R5']['sha256']);assert ptq['status']=='R5_PTQ_complete'
    refs=[];paid=0;ledger=[];event('R5Q_protocol',limit_new_trees=3840,initial_trees=120,new_per_fit=80,codecs=['fp16','int8'],same_quantized_initial=True)
    for case in cases:
        data=data_before_test(case['record']);directory=out/case['case'];directory.mkdir(exist_ok=True);source=locked(by[case['case']]['path'],by[case['case']]['sha256'])
        m=next(m for m in source['models'] if m['uid']=='compressed40_ABL/t200');checkpoint=model_prefix(load_model(m),120)
        last=checkpoint.blocks[-1];basis=last.base.basis.copy();assert basis.size
        scales=np.maximum(2*np.max(np.abs(last.base.coefficients),axis=0)/127.,np.finfo(np.float32).tiny).astype(np.float32)
        from adaptive_basis_core import risk_weights
        importance=risk_weights(data['train_y'],checkpoint.raw(data['train_x']),'multilabel')[1];models=[]
        for codec in ('fp16','int8'):
            initial=prepare_quant(checkpoint,np.zeros(len(basis),np.float32),codec,fixed_a=scales)
            for qat in (False,True):
                uid=codec+('/QAT' if qat else '/PTQ');paid+=80;assert paid<=3840
                attempt=dict(case=case['case'],uid=uid,trees=80,status='reserved');ledger.append(attempt);write(out/'tree_ledger.json',dict(new_trees=paid,attempts=ledger))
                path=directory/(uid.replace('/','-')+'-factor-proxy.npz');increment,trace=fit(case,data,initial,basis,importance,codec,scales,qat,cfg,path)
                attempt.update(status='complete',fit=trace);write(out/'tree_ledger.json',dict(new_trees=paid,attempts=ledger))
                combined=append_protected(checkpoint,increment);snapshot=prepare_quant(combined,np.zeros(len(basis),np.float32),codec,fixed_a=scales)
                raw=quant_raw(snapshot,data['dev_x']);np.testing.assert_allclose(raw[:13],snapshot.raw(data['dev_x'][:13]),atol=1e-8)
                folds=grouped_folds(data['dev_x'],case['record']['case']['seed']+20261029);assessment,_=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
                path=directory/(uid.replace('/','-')+'.npz');snapshot.save(path)
                models.append(dict(uid=uid,artifact=str(path),sha256=digest(path),packed_bytes=snapshot.resident_bytes+8,assessment=assessment,fit=trace))
                event('R5Q_fit',case=case['case'],uid=uid,new_trees=paid,fit=trace)
        p=directory/'development.json';write(p,dict(case=case['case'],record=case['record'],models=models));refs.append(dict(case=case['case'],path=str(p),sha256=digest(p)))
        write(out/'progress.json',dict(cases=refs,new_trees=paid,test_opened=False));del data,checkpoint;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    seal=out/'pretest_seal.json';write(seal,dict(cases=refs,new_trees=paid,test_opened=False));event('R5Q_seal',sha256=digest(seal));results=[]
    for ref in refs:
        case=locked(ref['path'],ref['sha256']);cc,dd=case['record']['case'],case['record']['dataset']
        assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        rows=[]
        for m in case['models']:
            snapshot=QuantSnapshot.load(m['artifact']);raw=quant_raw(snapshot,x)
            q=quality(y,*sufficient(y,raw,'multilabel',m['assessment']['refit_control']),'multilabel',snapshot.outputs)
            rows.append(dict(uid=m['uid'],quality=q,packed_bytes=m['packed_bytes'],fit=m['fit']))
        p=out/case['case']/'test.json';write(p,dict(case=case['case'],models=rows));results.append(dict(case=case['case'],models=rows))
    write(out/'report.json',dict(status='R5Q_complete',cases=refs,results=results,new_trees=paid,environment=env,pretest_seal=dict(path=str(seal),sha256=digest(seal))))
    event('R5Q_complete',sha256=digest(out/'report.json'),new_trees=paid)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
