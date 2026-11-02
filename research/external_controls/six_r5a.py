"""R5 PTQ quality and physical storage. Timing is delegated to R4/R6."""
from pathlib import Path
import gc,json,sys,time
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from six_common import load_cases,load_model,gpu_raw,price
from six_mixed_quant import prepare_mixed as prepare_quant,MixedSnapshot as QuantSnapshot
from six_mixed_quant import MixedGpu as QuantPackedGpu
from broad_core import assess,grouped_folds
from residual_gate_core import sufficient,quality


def quant_raw(snapshot,x,batch=1024):
    import cupy as cp
    groups=next(g for g in (4,2,1) if 8*(snapshot.outputs+g*snapshot.max_width+g*snapshot.outputs*int(snapshot.has_residual)+32*g)<=49152)
    dev=QuantPackedGpu(snapshot,groups=groups);result=np.empty((len(x),snapshot.outputs),float)
    assert dev.resident_device_array_bytes==snapshot.resident_bytes
    for a in range(0,len(x),batch):
        dx=cp.asarray(np.ascontiguousarray(x[a:a+batch]));buf=dev.buffers(len(dx));dev.launch_into(dx,*buf,raw=True);result[a:a+len(dx)]=buf[0].get();del dx,buf
    del dev;gc.collect();cp.get_default_memory_pool().free_all_blocks();return result


def run(cfg,out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    env=check_native_environment(cfg);_,cases=load_cases(cfg,first=True)
    r2=locked(cfg['six']['R2']['path'],cfg['six']['R2']['sha256']);r2by={r['case']:r for r in r2['cases']};refs=[]
    event('R5_protocol',new_trees=0,codecs=['fp16','int8'],quantized=['A','B','dense warm leaves'],retained_fp32=['sparse L','bias','shifts','thresholds'],physical_quantized_GPU_reads=True)
    for case in cases:
        data=data_before_test(case['record']);directory=out/case['case'];directory.mkdir(exist_ok=True);models=[]
        extra=locked(r2by[case['case']]['path'],r2by[case['case']]['sha256'])
        selected=[next(m for m in case['models'] if m['uid']==uid) for uid in ('direct_AB/t200','direct_ABL/t200')]
        selected.append(next(m for m in extra['models'] if m['uid']=='compressed40_ABL/t200'))
        for original in selected:
            model=load_model(original);name=original['uid'].replace('/','-');reference=gpu_raw(model,data['dev_x'])
            folds=grouped_folds(data['dev_x'],case['record']['case']['seed']+20261029)
            original_assessment,_=assess(reference,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
            for codec in (precision+'-'+scope for precision in ('fp16','int8') for scope in ('a_only','b_only','warm_only','ab_only')):
                snapshot=prepare_quant(model,np.zeros(len(model.bias),np.float32),codec);raw=quant_raw(snapshot,data['dev_x'])
                sample=data['dev_x'][:13];np.testing.assert_allclose(raw[:13],snapshot.raw(sample),atol=1e-8,rtol=1e-10)
                assessment,_=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
                path=directory/(name+'-'+codec+'.npz');snapshot.save(path)
                row=dict(uid=name+'/'+codec,family=original['family'],codec=codec,artifact=str(path),sha256=digest(path),
                         source=original,assessment=assessment,original_assessment=original_assessment,packed_bytes=snapshot.resident_bytes+8,
                         fp32_bytes=price(model),D_raw_error=float(np.max(np.abs(raw-reference))),
                         payload_arrays={k:dict(dtype=str(v.dtype),bytes=v.nbytes) for k,v in snapshot.arrays.items()})
                models.append(row);event('R5_quantized',case=case['case'],uid=row['uid'],bytes_ratio=row['packed_bytes']/price(model),raw_error=row['D_raw_error'])
            del model,reference;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        item=dict(case=case['case'],record=case['record'],models=models);p=directory/'development.json';write(p,item);refs.append(dict(case=case['case'],path=str(p),sha256=digest(p)))
        write(out/'progress.json',dict(cases=refs,test_opened=False))
    seal=out/'pretest_seal.json';write(seal,dict(cases=refs,test_opened=False));event('R5_seal',sha256=digest(seal));results=[]
    for ref in refs:
        case=locked(ref['path'],ref['sha256']);record=case['record'];cc,dd=record['case'],record['dataset']
        assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        rows=[];originals={}
        for m in case['models']:
            key=m['source']['uid']
            if key not in originals:
                model=load_model(m['source']);raw=gpu_raw(model,x);control=m['original_assessment']['refit_control']
                originals[key]=quality(y,*sufficient(y,raw,'multilabel',control),'multilabel',len(model.bias))
            s=QuantSnapshot.load(m['artifact']);raw=quant_raw(s,x);q={}
            for policy,assessment in [('fixed',m['original_assessment']),('recalibrated',m['assessment'])]:
                q[policy]=quality(y,*sufficient(y,raw,'multilabel',assessment['refit_control']),'multilabel',s.outputs)
            rows.append(dict(uid=m['uid'],codec=m['codec'],source_uid=key,quality=q,original_quality=originals[key],packed_bytes=m['packed_bytes'],fp32_bytes=m['fp32_bytes']))
        p=out/case['case']/'test.json';write(p,dict(case=case['case'],models=rows));results.append(dict(case=case['case'],models=rows));event('R5_test',case=case['case'],models=len(rows))
    write(out/'report.json',dict(status='R5A_complete',cases=refs,results=results,new_trees=0,environment=env,
                               pretest_seal=dict(path=str(seal),sha256=digest(seal)),seconds=time.perf_counter()-started))
    event('R5_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
