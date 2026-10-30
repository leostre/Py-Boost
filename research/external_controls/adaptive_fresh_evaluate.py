"""A4 frozen D choices, official test replay and equal-path resident banks."""
from pathlib import Path
from time import perf_counter,perf_counter_ns
import json,sys,gc,os,resource
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from unified_training import quality_row
from unified_frontier_core import sufficient,metrics
from unified_serving_core import bank_frontier,quality_key,summary,coalesce
from unified_serving_gpu import BlockGpu
from tfm_serving_core import request_plan


def selections(cases):
    """Pure D-only choice; test arrays are not arguments."""
    selected=[];options={};bank_cases=[]
    for case in cases:
        cid=case['case'];task=case['task'];reference=case['reference_bytes']
        ordinary=[m for m in case['models'] if m['family']=='ordinary'];anchor=min(ordinary,key=lambda m:quality_key(m,task))
        aq=quality_row(anchor['evaluations'],task)['quality']
        for family in ('ordinary','posthoc','controller','fixed_risk','escape_risk'):
            for ratio in (.25,.5,.7,1.,1.5):
                pool=[m for m in case['models'] if m['family']==family and m['cost_bytes']<=int(ratio*reference)]
                if pool:
                    m=min(pool,key=lambda m:quality_key(m,task));selected.append(dict(case=cid,family=family,ratio=ratio,uid=m['uid']))
        if cid.endswith(('_6104','_6204','_6304','_6404')):
            bank_cases.append(cid);options[cid]=[]
            for m in case['models']:
                q=quality_row(m['evaluations'],task)['quality'];cost=block_cost(coalesce(load_protected(m['artifact'])))['cost_bytes']
                utility=1-q['loss']/aq['loss'] if task=='multiclass' else q['f1_macro']/max(aq['f1_macro'],1e-12)-1
                options[cid].append(dict(uid=m['uid'],family=m['family'],cost_bytes=cost,utility=utility,quality=q))
    total=sum(next(c for c in cases if c['case']==cid)['reference_bytes'] for cid in bank_cases)
    budgets=sorted(set(int(r*total) for r in (.25,.5,.7,1.)))
    banks={mode:bank_frontier([[o for o in options[cid] if mode=='all' or o['family']=='ordinary'] for cid in bank_cases],budgets) for mode in ('all','ordinary')}
    return dict(selected=selected,bank_cases=bank_cases,options=options,banks=banks,budgets=budgets,test_evaluated=False)


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();environment=check_native_environment(cfg);spec=cfg['adaptive_final']
    report=locked(spec['training']['path'],spec['training']['sha256']);assert report['status']=='adaptive_fresh_training_complete'
    cases=[locked(r['path'],r['sha256']) for r in report['cases']];assert all(c['audit']['status']=='passed' for c in cases)
    # Every decision and all points are fixed before opening target[test].
    seal=selections(cases);seal['source']=spec['training'];seal['inventory']=[dict(case=c['case'],models=c['models'],record=c['record']) for c in cases]
    write(out/'pretest_seal.json',seal);sealed=digest(out/'pretest_seal.json');points=[];prediction_files=[]
    for case in cases:
        cid=case['case'];record=case['record'];task=case['task'];ids=None
        with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(record['dataset']['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        directory=out/cid;directory.mkdir(exist_ok=True);target=directory/'targets.npz';np.savez_compressed(target,y=y,rows=ids)
        for index,m in enumerate(case['models']):
            assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact']);z=model.raw(x)
            control=quality_row(m['evaluations'],task)['refit_control'];pred,loss,brier=sufficient(y,z,task,control['temperature'],control['shifts'])
            q=metrics(y,pred,loss,brier,task,z.shape[1]);raw=directory/f'prediction{index}.npz'
            np.savez_compressed(raw,raw=z,prediction=pred,loss=loss,brier=brier)
            points.append(dict(case=cid,task=task,uid=m['uid'],family=m['family'],cost_bytes=m['cost_bytes'],quality=q,
                development=quality_row(m['evaluations'],task)['quality'],control=control,artifact=m['artifact'],artifact_sha256=m['sha256'],
                raw=dict(path=str(raw),sha256=digest(raw)),targets=dict(path=str(target),sha256=digest(target)),outputs=z.shape[1]))
        write(out/'test_points.json',points);event('adaptive_final_case',case=cid,models=len(case['models']))
    # No reselection here: both bank families use the exact D-sealed choices.
    bank_test=[]
    for mode,allocations in seal['banks'].items():
        for allocation in allocations:
            if allocation['status']!='feasible':continue
            rows=[next(p for p in points if p['case']==cid and p['uid']==uid) for cid,uid in zip(seal['bank_cases'],allocation['choices'])]
            bank_test.append(dict(mode=mode,allocation=allocation,points=rows))
    write(out/'bank_test.json',bank_test)
    import cupy as cp
    from tfm_bank_worker import gpu_processes
    from unified_gpu_ownership import cupy_owner
    cp.get_default_memory_pool().set_limit(size=16*1024**3)
    namespaces={os.getpid()}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):namespaces.update(map(int,line.split()[1:]))
    owner=cupy_owner(gpu_processes,namespaces);timings=[];parity=[]
    for mode,allocations in seal['banks'].items():
        for allocation in allocations:
            if allocation['status']!='feasible':continue
            cpu={};gpu={};inputs={};controls={};resident=0
            for cid,uid in zip(seal['bank_cases'],allocation['choices']):
                case=next(c for c in cases if c['case']==cid);m=next(m for m in case['models'] if m['uid']==uid)
                data=data_before_test(case['record']);inputs[cid]=data['dev_x'];model=coalesce(load_protected(m['artifact']))
                control=quality_row(m['evaluations'],case['task'])['refit_control'];controls[cid]=control
                cpu[cid]=(model,[b.prepare_inference() for b in model.blocks]);g=BlockGpu(model,control['temperature'],control['shifts']);gpu[cid]=g;resident+=g.resident_device_array_bytes
                actual,labels=g.request(inputs[cid][:17]);z=model.raw(inputs[cid][:17]);from experiments.shared_leaf_basis import probabilities
                expected=probabilities(z/control['temperature'],model.task);expected=np.clip(expected,1e-7,1-1e-7)
                decisions=(z/control['temperature']-control['shifts']).argmax(1) if model.task=='multiclass' else z>=control['shifts']
                err=float(np.max(np.abs(actual-expected)));assert err<2e-5 and np.array_equal(labels,decisions)
                parity.append(dict(mode=mode,budget=allocation['budget'],case=cid,error=err,decision_mismatches=0))
            for hardware in ('GPU','CPU_prepared_reference'):
                for batch in spec['batches']:
                    rows=[];schedule=request_plan(len(inputs),32,'uniform');peak=cp.get_default_memory_pool().used_bytes();allocator=cp.cuda.get_allocator()
                    def tracked(size):
                        nonlocal peak
                        pointer=allocator(size);peak=max(peak,cp.get_default_memory_pool().used_bytes());return pointer
                    def request(cid,x):
                        if hardware=='GPU':return gpu[cid].request(x)
                        model,prepared=cpu[cid];z=np.broadcast_to(model.bias,(len(x),len(model.bias))).astype(float,copy=True)
                        for block in prepared:z+=block.raw(x)
                        from experiments.shared_leaf_basis import probabilities
                        q=controls[cid];return probabilities(z/q['temperature'],model.task),(z/q['temperature']-q['shifts']).argmax(1) if model.task=='multiclass' else z>=q['shifts']
                    for cid in inputs:request(cid,inputs[cid][np.arange(batch)%len(inputs[cid])])
                    with cp.cuda.using_allocator(tracked):
                        for i,j in enumerate(schedule):
                            cid=seal['bank_cases'][j];x=inputs[cid][np.arange(batch)%len(inputs[cid])];t=perf_counter_ns();request(cid,x)
                            ms=(perf_counter_ns()-t)/1e6;rows.append(dict(request=i,case=cid,ms=ms))
                            write(out/'latest_requests.json',dict(mode=mode,budget=allocation['budget'],hardware=hardware,batch=batch,requests=rows))
                    result=dict(mode=mode,budget=allocation['budget'],hardware=hardware,batch=batch,requests=rows,**summary([r['ms'] for r in rows]),
                        device_resident_array_bytes=resident,own_pool_peak_used_bytes=peak,cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        scope='host inputs to host probabilities and decisions; sequential32; CPU prepared Python reference, not optimized C++')
                    timings.append(result);write(out/'timings.json',timings)
            del cpu,gpu,g;gc.collect();cp.get_default_memory_pool().free_all_blocks()
            event('adaptive_bank_complete',mode=mode,budget=allocation['budget'])
    assert digest(out/'pretest_seal.json')==sealed
    result=dict(status='adaptive_final_complete',points=points,bank_test=bank_test,timings=timings,parity=parity,environment=environment,
        seal=dict(path=str(out/'pretest_seal.json'),sha256=sealed),gpu_owner=owner,elapsed_seconds=perf_counter()-started,
        limitations=['four distinct official test sets; three fits per same test are not independent tests','D adaptive reuse remains',
        'small banks do not establish physical A100 capacity gain','model arrays, pool peak and process RSS are separate quantities'])
    write(out/'report.json',result);event('adaptive_final_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
