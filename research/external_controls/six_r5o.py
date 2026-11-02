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
from six_quant_core import prepare_quant
from fast_quant_gpu import FastQuantPackedGpu as QuantPackedGpu,half_preflight
from budget_quality_worker import locked


def run(cfg,out):
    import cupy as cp
    out.mkdir(parents=True,exist_ok=True);pool=cp.get_default_memory_pool();pool.set_limit(size=12*1024**3)
    env=check_native_environment(cfg);_,cases=load_cases(cfg,first=True);rows=[];rng=np.random.default_rng(202610305)
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    preflight=half_preflight();write(out/'half-preflight.json',preflight)
    event('R5O_protocol',batches=[1,32,256,1024,4096],tuning_queries=16,final_sessions=3,requests=32,new_trees=0,gpu_before=gpu_before)
    r2=locked(cfg['six']['R2']['path'],cfg['six']['R2']['sha256']);by={r['case']:r for r in r2['cases']}
    for case in cases:
        data=data_before_test(case['record']);x=data['dev_x'];del data
        extra=locked(by[case['case']]['path'],by[case['case']]['sha256'])
        selected=[next(m for m in case['models'] if m['uid']==uid) for uid in ('ordinary/d3/t200','direct_AB/t200','direct_ABL/t200')]
        selected.append(next(m for m in extra['models'] if m['uid']=='compressed40_ABL/t200'))
        for m in selected:
            uid=m['uid'];model=load_model(m);control=m['assessment']['refit_control'];c=len(model.bias)
            snap=prepare_snapshot(model,control['shifts']);groups=next(g for g in (4,2,1) if 8*(c+g*snap.max_width+g*c*int(snap.has_residual)+32*g)<=49152)
            quantized={codec:prepare_quant(model,control['shifts'],codec) for codec in ('fp16','int8')}
            devices={'fp32':PackedGpu(model,control['temperature'],control['shifts'],groups)}
            devices.update({codec:QuantPackedGpu(q,control['temperature'],groups) for codec,q in quantized.items()})
            check=x[:13].copy();check=np.concatenate((check,np.full_like(check[:1],np.nan)));errors={};fp32_reference=model.raw(check)
            quantization_errors={}
            for name,dev in devices.items():
                expected=fp32_reference if name=='fp32' else quantized[name].raw(check)
                xx=cp.asarray(check);buf=dev.buffers(len(xx));dev.launch_into(xx,*buf,raw=True);actual=buf[0].get()
                errors[name]=float(np.max(np.abs(actual-expected)));assert errors[name]<1e-8,(case['case'],uid,name,errors[name])
                quantization_errors[name]=float(np.max(np.abs(actual-fp32_reference)));del xx,buf
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
                chosen='fp32';tuning.append(dict(batch=batch,chosen=chosen,samples=times))
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
            item=dict(case=case['case'],uid=uid,model_sha256=m['sha256'],tuning=tuning,measurements=measurements,memory=memory,raw_errors=errors,quantization_raw_errors=quantization_errors,
                      bases=len(model.blocks),rank_widths=[b.base.coefficients.shape[1] for b in model.blocks])
            write(directory/'report.json',item);rows.append(dict(case=case['case'],uid=uid,path=str(directory/'report.json'),sha256=digest(directory/'report.json')))
            write(out/'progress.json',dict(rows=rows));event('R5O_model',case=case['case'],uid=uid,routes=[r['chosen'] for r in tuning])
            del devices,dev,model;gc.collect();pool.free_all_blocks()
    write(out/'report.json',dict(status='R5O_complete',rows=rows,environment=env,new_trees=0,gpu_before=gpu_before,
          memory_scope='actual physical FP32/FP16/INT8 parameter arrays and scales plus allocation-tracked active request workspace; excludes context and driver',
          codec_selected=False,measurement_reference='fp32',half_preflight=preflight,
          precision='same packed algorithm and FP64 sums; physical FP32 FP16 INT8 parameters; quantization changes model function; common fixed controls'))
    event('R5O_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
