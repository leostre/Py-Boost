"""D-tuned dispatcher, then paired final serving and active workspace peak."""
from pathlib import Path
import gc,json,sys,time,subprocess
import numpy as np
from budget_quality_worker import write,digest,event
from abl_quality_worker import check_native_environment
from six_common import load_cases,load_model
from abl_transfer_study import data_before_test
from packed_serving_gpu import PackedGpu
from packed_serving_core import prepare_snapshot
from six_gather_gpu import GatherGemmGpu


def run(cfg,out):
    import cupy as cp
    out.mkdir(parents=True,exist_ok=True);pool=cp.get_default_memory_pool();pool.set_limit(size=12*1024**3)
    from native_pyboost_adapter import original_classes,OriginalGpu
    classes=original_classes()
    env=check_native_environment(cfg);_,cases=load_cases(cfg,first=True);rows=[];rng=np.random.default_rng(202610305)
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    event('R4_protocol',batches=[1,32,256,1024,4096],tuning_queries=16,final_sessions=3,requests=32,new_trees=0,gpu_before=gpu_before)
    for case in cases:
        data=data_before_test(case['record']);x=data['dev_x'];del data
        for uid in ('ordinary/d3/t200','direct_AB/t200','direct_ABL/t200'):
            m=next(m for m in case['models'] if m['uid']==uid);model=load_model(m);control=m['assessment']['refit_control'];c=len(model.bias)
            snap=prepare_snapshot(model,control['shifts']);groups=next(g for g in (4,2,1) if 8*(c+g*snap.max_width+g*c*int(snap.has_residual)+32*g)<=49152)
            devices={'packed':PackedGpu(model,control['temperature'],control['shifts'],groups),'gather_gemm':GatherGemmGpu(model,control['temperature'],control['shifts'])}
            if uid=='ordinary/d3/t200':devices['native']=OriginalGpu(model,control['temperature'],control['shifts'],classes)
            check=x[:13].copy();check=np.concatenate((check,np.full_like(check[:1],np.nan)));expected=model.raw(check);errors={}
            for name,dev in devices.items():
                if name=='native':
                    from scipy.special import expit
                    actual,labels=dev.request(check);errors[name]=float(np.max(np.abs(actual-expit(expected))))
                    assert errors[name]<3e-5
                    continue
                xx=cp.asarray(check);buf=dev.buffers(len(xx));dev.launch_into(xx,*buf,raw=True);errors[name]=float(np.max(np.abs(buf[0].get()-expected)))
                assert errors[name]<1e-8,(case['case'],uid,name,errors[name]);del xx,buf
            tuning=[];measurements=[]
            for batch in (1,32,256,1024,4096):
                features=np.ascontiguousarray(x[np.arange(batch)%len(x)]);xx=cp.asarray(features);buffers={n:d.buffers(batch) for n,d in devices.items() if n!='native'}
                def call(name):return devices[name].device(xx) if name=='native' else devices[name].launch_into(xx,*buffers[name])
                for name in devices:
                    for _ in range(4):call(name)
                cp.cuda.get_current_stream().synchronize();times={n:[] for n in devices}
                for _ in range(16):
                    for name in rng.permutation(list(devices)):
                        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();call(name);b.record();b.synchronize();times[name].append(float(cp.cuda.get_elapsed_time(a,b)))
                chosen=min(('packed','gather_gemm'),key=lambda n:(np.median(times[n]),n));tuning.append(dict(batch=batch,chosen=chosen,samples=times))
                # Decision is recorded before any final draw at this batch.
                directory=out/case['case']/uid.replace('/','-');directory.mkdir(parents=True,exist_ok=True)
                write(directory/'routing.json',dict(tuning=tuning,final_used_for_selection=False))
                for session in range(3):
                    samples={n:dict(gpu_ms=[],request_ms=[]) for n in devices}
                    for iteration in range(32):
                        for name in rng.permutation(list(devices)):
                            a,b=cp.cuda.Event(),cp.cuda.Event();a.record();call(name);b.record();b.synchronize();samples[name]['gpu_ms'].append(float(cp.cuda.get_elapsed_time(a,b)))
                    for iteration in range(32):
                        features=np.ascontiguousarray(x[(np.arange(batch)+iteration*17+session*53+409)%len(x)])
                        for name in rng.permutation(list(devices)):
                            begin=time.perf_counter_ns();devices[name].request(features);samples[name]['request_ms'].append((time.perf_counter_ns()-begin)/1e6)
                    measurements.append(dict(batch=batch,session=session,chosen=chosen,samples=samples))
                del xx,buffers,call
            memory=[]
            for name,dev in devices.items():
                for batch in (1,32,256,1024,4096):
                    features=np.ascontiguousarray(x[np.arange(batch)%len(x)]);dev.request(features);cp.cuda.get_current_stream().synchronize();gc.collect();pool.free_all_blocks()
                    baseline=pool.used_bytes();peak=[baseline];allocator=cp.cuda.get_allocator()
                    def tracked(size):
                        ptr=allocator(size);peak[0]=max(peak[0],pool.used_bytes());return ptr
                    with cp.cuda.using_allocator(tracked):dev.request(features)
                    cp.cuda.get_current_stream().synchronize()
                    memory.append(dict(name=name,batch=batch,model_arrays=dev.resident_device_array_bytes,workspace_peak=peak[0]-baseline,
                                       model_plus_active_workspace=dev.resident_device_array_bytes+peak[0]-baseline))
            item=dict(case=case['case'],uid=uid,model_sha256=m['sha256'],tuning=tuning,measurements=measurements,memory=memory,raw_errors=errors,
                      bases=len(model.blocks),rank_widths=[b.base.coefficients.shape[1] for b in model.blocks])
            write(directory/'report.json',item);rows.append(dict(case=case['case'],uid=uid,path=str(directory/'report.json'),sha256=digest(directory/'report.json')))
            write(out/'progress.json',dict(rows=rows));event('R4_model',case=case['case'],uid=uid,routes=[r['chosen'] for r in tuning])
            del devices,dev,model;gc.collect();pool.free_all_blocks()
    write(out/'report.json',dict(status='R4_complete',rows=rows,environment=env,new_trees=0,gpu_before=gpu_before,
          memory_scope='actual unique parameter arrays plus allocation-tracked active request workspace; excludes context and driver; GEMM decoder FP64 counted',
          native_source_commit=classes[3]['commit'],precision='native original FP32; packed/gather FP32 parameters and FP64 sums; native numerical check is probability error'))
    event('R4_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
