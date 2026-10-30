"""Real simultaneous bank residency, isolated backend, reproducible requests."""
from pathlib import Path
from time import perf_counter_ns
import json,sys,gc,resource,subprocess
import numpy as np
from budget_quality_worker import digest,write
from abl_protected_core import load_protected
from abl_transfer_study import data_before_test
from packed_serving_gpu import PackedGpu
from unified_serving_gpu import BlockGpu


def run(spec):
    import cupy as cp
    pool=cp.get_default_memory_pool();pool.set_limit(size=12*1024**3);devices=[];features=[];resident=[]
    for entry in spec['entries']:
        m=entry['model'];assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact']);control=m['assessment']['refit_control']
        plan=next(p for p in entry['plans'] if p['batch']==32)
        device=PackedGpu(model,control['temperature'],control['shifts'],plan['groups'],plan['layout']) if spec['backend']=='packed' else BlockGpu(model,control['temperature'],control['shifts'])
        devices.append(device);features.append(data_before_test(entry['record'])['dev_x']);resident.append(device.resident_device_array_bytes)
    # All task models remain resident through every request, with no reloads.
    for device,x in zip(devices,features):device.request(x[:1])
    cp.cuda.get_current_stream().synchronize();gc.collect();pool.free_all_blocks();baseline=pool.used_bytes();rows=[]
    ledger=dict(actual_device_array_bytes=sum(resident),per_model_array_bytes=resident,baseline_pool_used_bytes=baseline,
                models=len(devices),plan='one permanent D-tuned batch32 snapshot per task, used for every request batch; no copies for other layouts')
    write(Path(spec['output'])/'residency.json',ledger)
    wide=int(np.argmax([e['record']['case']['outputs'] for e in spec['entries']]))
    n=len(devices);rng=np.random.default_rng(spec['seed']);plans={}
    for distribution in ('uniform','hot_wide'):
        if distribution=='uniform':tasks=np.arange(spec['sessions']*spec['requests'])%n
        else:
            weights=np.full(n,.5/(n-1));weights[wide]=.5;tasks=rng.choice(n,spec['sessions']*spec['requests'],p=weights)
        plans[distribution]=tasks.tolist()
        for batch in spec['batches']:
            for device,x in zip(devices,features):device.request(np.ascontiguousarray(x[np.arange(batch)%len(x)]))
            cp.cuda.get_current_stream().synchronize();gc.collect();pool.free_all_blocks()
            for session in range(spec['sessions']):
                times=[];peak=[pool.used_bytes()];allocator=cp.cuda.get_allocator()
                def tracked(size):
                    ptr=allocator(size);peak[0]=max(peak[0],pool.used_bytes());return ptr
                with cp.cuda.using_allocator(tracked):
                    for iteration in range(spec['requests']):
                        task=int(tasks[session*spec['requests']+iteration]);x=features[task]
                        ids=(np.arange(batch)+iteration*17+session*53)%len(x);request=np.ascontiguousarray(x[ids])
                        begin=perf_counter_ns();prob,labels=devices[task].request(request);times.append((perf_counter_ns()-begin)/1e6)
                cp.cuda.get_current_stream().synchronize()
                rows.append(dict(distribution=distribution,batch=batch,session=session,request_ms=times,
                                 peak_pool_used_bytes=peak[0],incremental_peak_pool_bytes=peak[0]-baseline))
                write(Path(spec['output'])/'progress.json',dict(rows=rows,residency=ledger))
    result=dict(status='wide_bank_probe_complete',bank=spec['bank'],backend=spec['backend'],residency=ledger,request_plans=plans,rows=rows,
                hot_wide_case=spec['entries'][wide]['case'],rss_process_highwater_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),
                memory_scope='own CuPy model arrays and allocator peak; foreign contexts and driver/compiler allocations excluded; all twelve models physically resident',
                limitation='fixed12tasks; fitting both banks on A100 does not prove maximum model count or100task transfer')
    write(Path(spec['output'])/'report.json',result)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()))
