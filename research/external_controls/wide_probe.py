"""Isolated latency or memory process, launched only by an orx-owned worker."""
from pathlib import Path
from time import perf_counter_ns
import json,sys,gc,resource,subprocess
import numpy as np
from budget_quality_worker import digest,write
from abl_protected_core import load_protected
from abl_transfer_study import data_before_test
from packed_serving_gpu import PackedGpu
from unified_serving_gpu import BlockGpu


def legacy_device(device,x,out,work):
    import cupy as cp
    device.raw_into(x,out,work);out/=device.temperature
    labels=out>=device.shifts if device.task=='multilabel' else (out-device.shifts).argmax(1)
    if device.task=='multilabel':
        e=cp.exp(-cp.abs(out));out[:]=cp.where(out>=0,1/(1+e),e/(1+e))
    else:
        out-=out.max(1,keepdims=True);cp.exp(out,out=out);out/=out.sum(1,keepdims=True)
    cp.clip(out,1e-7,1-1e-7,out=out);return out,labels


def run(request):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    entry=request['entry'];m=entry['model'];assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact'])
    data=data_before_test(entry['record']);features=data['dev_x'];control=m['assessment']['refit_control'];t=control['temperature'];shifts=control['shifts']
    del data
    outdir=Path(request['output']);rows=[];rng=np.random.default_rng(request['seed']);mode=request['mode'];pool=cp.get_default_memory_pool()
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    for batch in request['batches']:
        plan=next(p for p in entry['plans'] if p['batch']==batch);kwargs=dict(groups=plan['groups'],layout=plan['layout'])
        if mode=='latency':
            packed=PackedGpu(model,t,shifts,**kwargs);legacy=BlockGpu(model,t,shifts);dx=cp.asarray(np.ascontiguousarray(features[np.arange(batch)%len(features)]))
            buffers=packed.buffers(batch);old_out=cp.empty((batch,len(model.bias)),cp.float64);work=cp.empty_like(old_out) if len(legacy.blocks)>1 else old_out
            def device_call(name):return packed.launch_into(dx,*buffers) if name=='packed' else legacy_device(legacy,dx,old_out,work)
            for name in ('packed','legacy'):
                for _ in range(4):device_call(name)
            cp.cuda.get_current_stream().synchronize()
            for session in range(request['sessions']):
                samples={name:dict(gpu_ms=[],request_ms=[]) for name in ('packed','legacy')};orders=[]
                for iteration in range(request['requests']):
                    order=list(rng.permutation(['packed','legacy']));orders.append(order)
                    for name in order:
                        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();device_call(name);b.record();b.synchronize()
                        samples[name]['gpu_ms'].append(float(cp.cuda.get_elapsed_time(a,b)))
                for iteration in range(request['requests']):
                    ids=(np.arange(batch)+iteration*17+session*53)%len(features);x=np.ascontiguousarray(features[ids])
                    for name in rng.permutation(['packed','legacy']):
                        device=packed if name=='packed' else legacy;begin=perf_counter_ns();prob,labels=device.request(x)
                        elapsed=(perf_counter_ns()-begin)/1e6;samples[name]['request_ms'].append(elapsed)
                row=dict(batch=batch,session=session,samples=samples,gpu_orders=orders,plan=plan)
                rows.append(row);write(outdir/'progress.json',dict(rows=rows,case=entry['case'],uid=entry['uid']))
            del packed,legacy,dx,buffers,old_out,work,device_call,device
        else:
            gc.collect();pool.free_all_blocks();device=PackedGpu(model,t,shifts,**kwargs) if mode=='memory_packed' else BlockGpu(model,t,shifts)
            x=np.ascontiguousarray(features[np.arange(batch)%len(features)])
            device.request(x);cp.cuda.get_current_stream().synchronize();gc.collect();pool.free_all_blocks()
            baseline=pool.used_bytes();peak=[baseline];allocator=cp.cuda.get_allocator()
            def tracked(size):
                ptr=allocator(size);peak[0]=max(peak[0],pool.used_bytes());return ptr
            with cp.cuda.using_allocator(tracked):
                for _ in range(3):prob,labels=device.request(x)
            cp.cuda.get_current_stream().synchronize()
            rows.append(dict(batch=batch,plan=plan,resident_device_array_bytes=device.resident_device_array_bytes,
                baseline_pool_used_bytes=baseline,peak_pool_used_bytes=peak[0],incremental_peak_pool_bytes=peak[0]-baseline,
                end_pool_used_bytes=pool.used_bytes(),rss_process_highwater_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024))
            write(outdir/'progress.json',dict(rows=rows,case=entry['case'],uid=entry['uid']));del device,tracked
        gc.collect();pool.free_all_blocks()
    result=dict(status='wide_probe_complete',mode=mode,case=entry['case'],uid=entry['uid'],roles=entry['roles'],rows=rows,
        gpu_before=gpu_before,gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),
        feature_scope='fixed D features; no labels used; benchmark repeats extend beyond tuning16, tuning and final draws separated',
        precision='float32 parameters/features; float64 sums/probabilities; same calibration',
        memory_scope='own CuPy used-byte allocator peak, excludes foreign contexts and driver/compiler allocations; RSS is process highwater')
    write(outdir/'report.json',result)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()))
