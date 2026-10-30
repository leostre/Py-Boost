"""Already-sealed native UIDs, exactly the same two-task traffic as A4X."""
from pathlib import Path
from time import perf_counter_ns
import json,sys,gc,os,resource
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from unified_training import quality_row
from unified_serving_core import quality_key,coalesce,summary
from unified_serving_gpu import BlockGpu
from tfm_serving_core import request_plan


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['adaptive_matched'];environment=check_native_environment(cfg)
    final=locked(spec['final']['path'],spec['final']['sha256']);assert final['status']=='adaptive_final_complete'
    seal=locked(final['seal']['path'],final['seal']['sha256']);external=locked(spec['external']['path'],spec['external']['sha256']);assert external['status']=='adaptive_external_complete'
    selected={};records={};choices=[];cases=spec['cases']
    for cid in cases:
        case=next(c for c in seal['inventory'] if c['case']==cid);records[cid]=case['record'];models=case['models'];task='multiclass'
        frozen=[next(m for m in models if m['uid']==s['uid']) for s in seal['selected'] if s['case']==cid and s['ratio']==.7]
        pools={'full':[m for m in models if m['family']=='ordinary'],'ordinary07':[m for m in frozen if m['family']=='ordinary'],
            'compact07':[m for m in frozen if m['family']!='ordinary'],'union07':frozen}
        for mode,pool in pools.items():
            m=min(pool,key=lambda m:quality_key(m,task));selected[mode,cid]=m
            choices.append(dict(mode=mode,case=cid,uid=m['uid'],source_artifact=m['artifact'],sha256=m['sha256'],cost_bytes=m['cost_bytes'],control=quality_row(m['evaluations'],task)['refit_control']))
    write(out/'measurement_seal.json',dict(choices=choices,source_final=spec['final'],source_external=spec['external'],new_test_selection=False))
    import cupy as cp
    from tfm_bank_worker import gpu_processes
    from unified_gpu_ownership import cupy_owner
    namespaces={os.getpid()}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):namespaces.update(map(int,line.split()[1:]))
    owner=cupy_owner(gpu_processes,namespaces);timings=[];inventory=[];parity=[]
    for mode in ('full','ordinary07','compact07','union07'):
        models={};prepared={};gpu={};inputs={};controls={};resident=0
        for cid in cases:
            m=selected[mode,cid];assert digest(m['artifact'])==m['sha256'];model=coalesce(load_protected(m['artifact']));models[cid]=model
            prepared[cid]=[b.prepare_inference() for b in model.blocks];inputs[cid]=data_before_test(records[cid])['dev_x']
            q=quality_row(m['evaluations'],'multiclass')['refit_control'];controls[cid]=q;g=BlockGpu(model,q['temperature'],q['shifts']);gpu[cid]=g;resident+=g.resident_device_array_bytes
            from scipy.special import softmax
            x=inputs[cid][:17];p,labels=g.request(x);z=model.raw(x)/q['temperature'];expected=np.clip(softmax(z,axis=1),1e-7,1-1e-7)
            err=float(np.max(np.abs(p-expected)));assert err<2e-5 and np.array_equal(labels,(z-q['shifts']).argmax(1))
            parity.append(dict(mode=mode,case=cid,error=err,decision_mismatches=0))
            inventory.append(dict(mode=mode,case=cid,uid=m['uid'],resident_array_bytes=g.resident_device_array_bytes,model_cost_bytes=block_cost(model)['cost_bytes'],backend=g.backend))
        for hardware in ('GPU','CPU_prepared_reference'):
            for batch in (1,32,256,1024):
                def request(cid,x):
                    if hardware=='GPU':return gpu[cid].request(x)
                    model=models[cid];z=np.broadcast_to(model.bias,(len(x),len(model.bias))).astype(float,copy=True)
                    for b in prepared[cid]:z+=b.raw(x)
                    q=controls[cid];z/=q['temperature'];return np.clip(softmax(z,axis=1),1e-7,1-1e-7),(z-q['shifts']).argmax(1)
                for cid in cases:request(cid,inputs[cid][np.arange(batch)%len(inputs[cid])])
                for traffic in ('uniform','skew'):
                    rows=[];peak=cp.get_default_memory_pool().used_bytes();allocator=cp.cuda.get_allocator()
                    def tracked(size):
                        nonlocal peak
                        p=allocator(size);peak=max(peak,cp.get_default_memory_pool().used_bytes());return p
                    with cp.cuda.using_allocator(tracked):
                        for i,j in enumerate(request_plan(2,32,traffic)):
                            cid=cases[j];x=inputs[cid][np.arange(batch)%len(inputs[cid])];t=perf_counter_ns();request(cid,x);ms=(perf_counter_ns()-t)/1e6
                            rows.append(dict(request=i,case=cid,ms=ms));write(out/'latest_requests.json',dict(mode=mode,hardware=hardware,batch=batch,traffic=traffic,requests=rows))
                    timing=dict(mode=mode,hardware=hardware,batch=batch,traffic=traffic,requests=rows,**summary([r['ms'] for r in rows]),device_resident_array_bytes=resident,
                        own_pool_peak_used_bytes=peak,cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    timings.append(timing);write(out/'timings.json',timings)
        del gpu,g,models,prepared;gc.collect();cp.get_default_memory_pool().free_all_blocks();event('adaptive_matched_bank',mode=mode)
    result=dict(status='adaptive_matched_serving_complete',timings=timings,inventory=inventory,parity=parity,gpu_owner=owner,environment=environment,
        seal=dict(path=str(out/'measurement_seal.json'),sha256=digest(out/'measurement_seal.json')),new_trees=0,
        limitations=['same frozen D choices and input/query schedules; separate sequential processes','CPU prepared Python is a reference path, not native C++ library implementation'])
    write(out/'report.json',result);event('adaptive_matched_serving_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
