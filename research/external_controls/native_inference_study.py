"""Paired native / legacy / packed timing of sealed saved models, via orx only."""
from pathlib import Path
from time import perf_counter_ns
import gc,json,subprocess,sys
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from abl_protected_core import load_protected
from abl_transfer_study import data_before_test
from native_pyboost_adapter import original_classes,OriginalGpu
from wide_probe import legacy_device


def run(cfg,out):
    import cupy as cp
    from scipy.special import expit
    from packed_serving_gpu import PackedGpu
    from unified_serving_gpu import BlockGpu
    out.mkdir(parents=True,exist_ok=True);environment=check_native_environment(cfg)
    spec=cfg['native_inference'];seal=locked(spec['tuning']['path'],spec['tuning']['sha256'])
    classes=original_classes();assert classes[3]['commit']==spec['source_commit']
    pool=cp.get_default_memory_pool();pool.set_limit(size=spec['own_pool_limit_bytes'])
    before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free=int(before.split(',')[0])*1024**2
    if free<spec['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory')
    entries=[e for e in seal['entries'] if any(r in spec['roles'] for r in e['roles'])]
    datasets=list(dict.fromkeys(e['dataset'] for e in entries));rows=[];checks=[]
    for index,name in enumerate(datasets):
        group=[e for e in entries if e['dataset']==name];full=next(e for e in group if 'full200' in e['roles'])
        data=data_before_test(full['record']);features=data['dev_x'];del data
        models={}
        for e in group:
            assert digest(e['model']['artifact'])==e['model']['sha256']
            role=next(r for r in spec['roles'] if r in e['roles']);models[role]=(e,load_protected(e['model']['artifact']))
        for batch in spec['batches']:
            devices={};buffers={};model_meta={}
            x=np.ascontiguousarray(features[np.arange(batch)%len(features)]);dx=cp.asarray(x)
            for role,(e,model) in models.items():
                control=e['model']['assessment']['refit_control'];t=control['temperature'];shifts=control['shifts']
                plan=next(p for p in e['plans'] if p['batch']==batch)
                for backend in ('legacy','packed'):
                    key=backend+':'+role
                    device=BlockGpu(model,t,shifts) if backend=='legacy' else PackedGpu(model,t,shifts,plan['groups'],plan['layout'])
                    devices[key]=device
                    if backend=='packed':buffers[key]=device.buffers(batch)
                    else:
                        a=cp.empty((batch,len(model.bias)),cp.float64);buffers[key]=(a,cp.empty_like(a) if len(device.blocks)>1 else a)
                if role=='full200':devices['native:full200']=OriginalGpu(model,t,shifts,classes)
                model_meta[role]=dict(uid=e['uid'],artifact=e['model']['artifact'],sha256=e['model']['sha256'],oof_f1=e['model']['assessment']['quality']['f1_macro'],plan=plan)
                # Validate on full D-prefix and NaN edge input; no labels/fitting.
                test=x[:min(batch,37)].copy();test[-1,0]=np.nan
                ref_raw=model.raw(test);expected=np.clip(expit(ref_raw/t),1e-7,1-1e-7);labels=ref_raw/t>=np.asarray(shifts)
                for backend in ('legacy','packed')+ (('native',) if role=='full200' else ()):
                    p,l=devices[backend+':'+role].request(test);err=float(np.max(np.abs(p-expected)))
                    tolerance=3e-5 if backend=='native' else 2e-9
                    assert err<tolerance,(name,role,backend,err)
                    ties=np.abs(expected-expit(np.asarray(shifts)))<tolerance
                    assert not np.any((l!=labels)&~ties),(name,role,backend,'non-tie label difference')
                    checks.append(dict(dataset=name,role=role,backend=backend,batch=batch,probability_max_abs_error=err,label_disagreements=int(np.sum(l!=labels)),near_threshold_disagreements=int(np.sum((l!=labels)&ties)),tolerance=tolerance,nan_edge=True))
            names=list(devices);rng=np.random.default_rng(spec['seed']+index*10+spec['batches'].index(batch))
            def call(key):
                device=devices[key]
                if key.startswith('native'):return device.device(dx)
                if key.startswith('packed'):return device.launch_into(dx,*buffers[key])
                return legacy_device(device,dx,*buffers[key])
            for key in names:
                for _ in range(spec['warmup']):call(key);devices[key].request(x)
            cp.cuda.get_current_stream().synchronize()
            for session in range(spec['sessions']):
                samples={k:dict(gpu_ms=[],request_ms=[]) for k in names}
                for iteration in range(spec['requests']):
                    for key in rng.permutation(names):
                        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();call(key);b.record();b.synchronize()
                        samples[key]['gpu_ms'].append(float(cp.cuda.get_elapsed_time(a,b)))
                for iteration in range(spec['requests']):
                    xx=np.ascontiguousarray(features[(np.arange(batch)+17*iteration+53*session)%len(features)])
                    for key in rng.permutation(names):
                        start=perf_counter_ns();p,l=devices[key].request(xx)
                        samples[key]['request_ms'].append((perf_counter_ns()-start)/1e6)
                rows.append(dict(dataset=name,case=full['case'],batch=batch,session=session,samples=samples,models=model_meta))
                write(out/'progress.json',dict(rows=rows,checks=checks));event('native_session',dataset=name,batch=batch,session=session)
            del devices,buffers,dx,call,device;gc.collect();pool.free_all_blocks()
        del models,features;gc.collect();pool.free_all_blocks()
    result=dict(status='native_comparison_complete',rows=rows,checks=checks,source=spec,environment=environment,gpu_before=before,
        gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),
        new_trees=0,test_labels_used=False,precision='native unchanged float32; legacy/packed float32 inputs and parameters, float64 sums/probabilities',
        request_scope='H2D + original Ensemble.predict(CuPy) or optimized request + frozen thresholds + D2H probabilities and labels; warm JIT; native original allocations and stream synchronization retained',
        gpu_scope='CUDA event elapsed resident input; native original allocations/synchronization retained; complete probabilities and labels; excludes H2D/D2H',
        quality_scope='no new quality evaluation; OOF F1 carried from sealed model assessment; distinct compact models are not the same-function comparison',
        memory_scope='No new isolated memory benchmark; existing legacy/packed memory evidence is separate')
    write(out/'report.json',result);event('native_complete',rows=len(rows),checks=len(checks),new_trees=0)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
