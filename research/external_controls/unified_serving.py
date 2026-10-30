"""Frozen frontier models: equal GPU paths, actual banks and paired controls."""
from pathlib import Path
from time import perf_counter,perf_counter_ns,sleep
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc,json,os,sys
import numpy as np
from scipy.special import expit,softmax
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,as_protected,block_cost
from unified_training import evaluate,quality_row,serialize
from unified_frontier_core import make_folds,sufficient,metrics
from unified_serving_core import coalesce,bank_frontier,summary,quality_key
from unified_serving_gpu import BlockGpu
from unified_training_core import candidate_grid,shortlist
from abl_budget_revision_core import decode_model,blank
from dataclasses import replace
from unified_recalibration import audit_case
from tfm_serving_core import request_plan,gpu_owner
from experiments.parallel_leaf_core import prefix
from unified_training_core import canonical
from unified_gpu_ownership import cupy_owner


def stream_request(gpu,features):
    import cupy as cp
    with cp.cuda.Device(0),cp.cuda.Stream(non_blocking=True) as stream:
        result=gpu.request(features);stream.synchronize();return result



def native_matched(cfg,out):
    """Re-evaluate every saved ordinary/sketch topology using all three old D roles."""
    ref=cfg['bank_sources']['native'];source=locked(ref['path'],ref['sha256']);entries=[]
    for record in source['records']:
        cid=record['case_id'];directory=out/cid/record['family'];directory.mkdir(parents=True,exist_ok=True)
        data=data_before_test(record);task=record['case']['task'];folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028)
        models=[]
        for uid,ref in record['artifacts'].items():
            assert digest(ref['artifact'])==ref['sha256'];model=as_protected(native_load(ref['artifact']))
            p=directory/uid;p.mkdir(exist_ok=True);raw=model.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,p)
            models.append(serialize(model,raw,ev,p,record['family']+'/'+uid,dict(family=record['family'],source=ref)))
        entry=dict(case=cid,task=task,record=record,models=models);file=directory/'case.json';write(file,entry)
        entries.append(dict(case=cid,family=record['family'],path=str(file),sha256=digest(file)))
        event('unified_matched_native',case=cid,family=record['family'],models=len(models))
    return entries


def matched_compression(cfg,entries,out):
    cases=[]
    for entry in entries:
        if entry['family']!='full':continue
        source=locked(entry['path'],entry['sha256']);record=source['record'];data=data_before_test(record)
        task=source['task'];cid=source['case'];directory=out/cid;directory.mkdir(parents=True,exist_ok=True)
        models=[dict(m,family='ordinary') for m in source['models']];full=min(models,key=lambda m:quality_key(m,task))
        model=load_protected(full['artifact']);view,_=decode_model(model);dense=blank(replace(view.base(),bias=view.bias.copy()))
        folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028)
        # Even when the selected ordinary model is the shortest old checkpoint,
        # include true smaller prefixes rather than inventing a feasible option.
        for count in sorted({max(1,model.n_trees//4),max(1,model.n_trees//2)}):
            short=canonical(as_protected(prefix(dense,count)));p=directory/f'prefix{count}';p.mkdir(exist_ok=True)
            raw=short.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,p)
            models.append(serialize(short,raw,ev,p,f'ordinary_prefix/{count}',dict(family='ordinary',source=full['uid'])))
        candidates=list(candidate_grid(None,dense,{'train':data['train_x'],'development':data['dev_x']},
            {'train':data['train_y']},int(.7*full['cost_bytes']),caps=(4,8,32,128)))
        for candidate in shortlist(candidates,6):
            p=directory/candidate['uid'];p.mkdir(exist_ok=True)
            ev=evaluate(candidate['raw']['development'],data['dev_y'],task,folds,cfg,p)
            models.append(serialize(candidate['model'],candidate['raw']['development'],ev,p,'posthoc/'+candidate['uid'],
                dict(family='posthoc',rank=candidate['rank'],leaf_cap=candidate['leaf_cap'],residual_fraction=candidate['residual_fraction'],source=full['uid'])))
        case=dict(case=cid,record=record,task=task,models=models,reference_bytes=full['cost_bytes'],full_uid=full['uid'],reference=quality_row(full['evaluations'],task))
        case['audit']=audit_case(case,data);write(directory/'case.json',case);cases.append(case)
    return cases


def add_short_controls(cfg,case,data,out):
    ref=case['warmup'];assert digest(ref['artifact'])==ref['sha256'];warm=load_protected(ref['artifact'])
    view,_=decode_model(warm);dense=blank(replace(view.base(),bias=view.bias.copy()))
    folds=make_folds(data['dev_y'],case['task'],case['record']['case']['seed']+20261028);extra=[]
    for count in (10,20,40):
        model=canonical(as_protected(prefix(dense,count)));p=out/f't{count}';p.mkdir(parents=True,exist_ok=True)
        raw=model.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],case['task'],folds,cfg,p)
        extra.append(serialize(model,raw,ev,p,f'ordinary_savedprefix/{count}',dict(family='ordinary',block=-1,source_warmup=ref)))
    check=audit_case(dict(task=case['task'],models=extra),data)
    case['models']=case['models']+extra;case['audit']=dict(status='passed',original=case['audit'],short_controls=check)


def run(cfg,out):
    import resource
    from tfm_bank_worker import gpu_processes
    started=perf_counter();environment=check_native_environment(cfg);spec=cfg['unified_serving'];out.mkdir(parents=True,exist_ok=True)
    u2=locked(spec['u2']['path'],spec['u2']['sha256']);assert all(r['audit']['status']=='passed' for r in u2['cases'])
    if spec.get('u3'):
        u3=locked(spec['u3']['path'],spec['u3']['sha256']);assert all(r['audit']['status']=='passed' for r in u3['cases'])
    cases=[locked(r['path'],r['sha256']) for r in u2['cases'] if r['case'] in spec['cases']]
    if spec.get('matched_recalibration'):
        ref=spec['matched_recalibration'];prior=locked(ref['path'],ref['sha256']);matched=prior['matched_native']
    else:matched=native_matched(cfg,out/'matched') if spec.get('matched_native',True) else []
    if spec.get('source_kind')=='matched':
        cases=matched_compression(cfg,matched,out/'matched_compression')
    cases.sort(key=lambda c:spec['cases'].index(c['case']))
    assert len(cases)==len(spec['cases'])
    selected={};inventory=[];options={};inputs={}
    for case in cases:
        cid=case['case'];task=case['task'];reference=case['reference']['quality'];options[cid]=[];data=data_before_test(case['record'])
        if spec.get('short_controls') and spec.get('source_kind')!='matched':add_short_controls(cfg,case,data,out/'short_controls'/cid)
        for m in case['models']:
            q=quality_row(m['evaluations'],task)['quality']
            utility=1-q['loss']/reference['loss'] if task=='multiclass' else q['f1_macro']/max(reference['f1_macro'],1e-30)-1
            compact=coalesce(load_protected(m['artifact']));cost=block_cost(compact)['cost_bytes']
            m['leaf_count']=sum(len(b.base.coefficients) for b in compact.blocks);m['node_count']=sum(len(b.base.nodes) for b in compact.blocks);m['coalesced_cost_bytes']=cost
            options[cid].append(dict(uid=m['uid'],cost_bytes=cost,utility=utility,family=m['family'],quality=q))
        pools=dict(full=[m for m in case['models'] if m['family']=='ordinary' and (m['uid']==case['full_uid'] if 'full_uid' in case else m['block']==3)],
            posthoc=[m for m in case['models'] if m['family']=='posthoc' and m['cost_bytes']<=.7*case['reference_bytes']],
            online=[m for m in case['models'] if m['family']=='online' and m['cost_bytes']<=.7*case['reference_bytes']],
            small=[m for m in case['models'] if m['family']=='ordinary' and m['cost_bytes']<=.7*case['reference_bytes']])
        for family,pool in pools.items():
            if family not in spec.get('families',('full','posthoc','online')):continue
            assert pool,(cid,family);m=min(pool,key=lambda m:quality_key(m,task));selected[cid,family]=m
            inventory.append(dict(case=cid,bank=family,uid=m['uid'],artifact=m['artifact'],sha256=m['sha256'],
                control=quality_row(m['evaluations'],task)['refit_control'],cost_bytes=m['cost_bytes'],quality=quality_row(m['evaluations'],task)['quality'],
                leaf_count=m['leaf_count'],node_count=m['node_count'],trees=m['trees'],coalesced_cost_bytes=m['coalesced_cost_bytes']))
        inputs[cid]=data['dev_x'].copy()
        p=out/'frontier_cases'/cid/'case.json';write(p,case)
    budgets=spec['bank_budgets_bytes'];banks={}
    for mode in ('all','ordinary_only','posthoc_only',*(() if spec.get('source_kind')=='matched' else ('online_only',))):
        allowed=lambda o: mode=='all' or o['family']==mode.split('_only')[0]
        banks[mode]=bank_frontier([[o for o in options[c['case']] if allowed(o)] for c in cases],budgets)
    # Seal all choices and allocation before any GPU request and before test use.
    seal=dict(inventory=inventory,bank_frontiers=banks,options=options,selection_scope='adaptive D',test_evaluated=False,
        source_cases=[dict(case=c['case'],record=c['record']) for c in cases])
    write(out/'pretest_seal.json',seal)
    training_gate=None
    if spec.get('wait_for_owned_training'):
        gate=spec['wait_for_owned_training'];path=Path(gate['report_path']);waiting=perf_counter()
        while not path.is_file():
            if perf_counter()-waiting>gate['maximum_wait_seconds']:raise RuntimeError('owned training completion gate timed out')
            event('waiting_for_owned_training',report=str(path),elapsed_seconds=perf_counter()-waiting);sleep(30)
        completed=json.loads(path.read_text());assert completed['status']=='unified_history_complete'
        assert len(completed['cases'])==gate['expected_cases'] and all(r['audit']['status']=='passed' for r in completed['cases'])
        training_gate=dict(path=str(path),sha256=digest(path),status='audited_training_complete',role='resource readiness only; no metrics used in serving choices')
        write(out/'training_completion_gate.json',training_gate)
    before_gpu=gpu_processes()
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=16*1024**3)
    results=[];parity=[];bank_measurements=[];prefix_memory=[];ownership=None
    for family in spec.get('families',('full','posthoc','online')):
        cp.get_default_memory_pool().free_all_blocks();gpu={};resident=0
        for case in cases:
            cid=case['case'];m=selected[cid,family];assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact'])
            control=quality_row(m['evaluations'],case['task'])['refit_control'];g=BlockGpu(model,control['temperature'],control['shifts'])
            x=inputs[cid][:17];actual,labels=g.request(x);z=model.raw(x)/control['temperature']
            expected=np.clip(softmax(z,axis=1) if case['task']=='multiclass' else expit(z),1e-7,1-1e-7)
            expected_labels=(z-control['shifts']).argmax(1) if case['task']=='multiclass' else z>=control['shifts']
            error=float(np.max(np.abs(actual-expected)));mismatches=int(np.count_nonzero(labels!=expected_labels))
            assert error<=2e-5 and mismatches==0,(cid,family,error,mismatches)
            parity.append(dict(case=cid,bank=family,probability_max_abs=error,decision_mismatches=mismatches,
                blocks_before=len(model.blocks),blocks_after=len(g.blocks),backend=g.backend,resident_array_bytes=g.resident_device_array_bytes))
            if spec.get('correctness_only'):
                for batch in spec['batches']:
                    xx=inputs[cid][np.arange(batch)%len(inputs[cid])]
                    zz=model.raw(xx)/control['temperature'];pp=np.clip(softmax(zz,axis=1) if case['task']=='multiclass' else expit(zz),1e-7,1-1e-7)
                    yy=(zz-control['shifts']).argmax(1) if case['task']=='multiclass' else zz>=control['shifts']
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        pending=[executor.submit(stream_request,g,xx) for _ in range(4)]
                        for future in pending:
                            actual,observed=future.result();np.testing.assert_allclose(actual,pp,atol=2e-5,rtol=0);np.testing.assert_array_equal(observed,yy)
                    parity.append(dict(case=cid,bank=family,batch=batch,concurrent_requests=4,status='passed'))
            gpu[cid]=g;resident+=g.resident_device_array_bytes
            cp.get_default_memory_pool().free_all_blocks()
            namespace_pids={os.getpid()}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('NSpid:'):namespace_pids.update(map(int,line.split()[1:]))
            if ownership is None:ownership=cupy_owner(gpu_processes,namespace_pids)
            owner=ownership
            prefix_memory.append(dict(bank=family,tasks=len(gpu),last_task=cid,resident_model_arrays_bytes=resident,
                pool_used_bytes=cp.get_default_memory_pool().used_bytes(),pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
                driver_bytes=gpu_processes().get(owner['pid']) if owner['verified'] else None,gpu_ownership=owner,
                interpretation='actual fixed-order resident prefix of distinct tasks, not synthetic independent replicas'))
            write(out/'resident_prefix_memory.json',prefix_memory)
        resident_pool=dict(used_bytes=cp.get_default_memory_pool().used_bytes(),reserved_bytes=cp.get_default_memory_pool().total_bytes())
        if spec.get('correctness_only'):
            del gpu,g;gc.collect();cp.get_default_memory_pool().free_all_blocks();continue
        namespace_pids={os.getpid()}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('NSpid:'):namespace_pids.update(map(int,line.split()[1:]))
        assert ownership is not None
        driver_snapshot=gpu_processes().get(ownership['pid']) if ownership['verified'] else None
        for case in cases:
            cid=case['case'];g=gpu[cid]
            for batch in spec['batches']:
                x=inputs[cid][np.arange(batch)%len(inputs[cid])].copy();g.request(x);gc.collect();cp.get_default_memory_pool().free_all_blocks();times=[];pool_peak=0
                for request in range(spec['requests']):
                    t=perf_counter_ns();g.request(x);elapsed=(perf_counter_ns()-t)/1e6;times.append(elapsed)
                    pool_peak=max(pool_peak,cp.get_default_memory_pool().total_bytes())
                    write(out/'latest_request.json',dict(bank=family,case=cid,batch=batch,request=request,ms=elapsed))
                row=dict(bank=family,case=cid,batch=batch,concurrency=1,raw_ms=times,**summary(times),
                    resident_bank_arrays_bytes=resident,resident_pool=resident_pool,pool_peak_reserved_bytes=pool_peak,
                    gpu_ownership=ownership,driver_bytes_after_bank_load=driver_snapshot,
                    input_bytes=x.nbytes,output_bytes=batch*g.outputs*8,scope='host-to-device + calibrated probability and label output to host')
                results.append(row);write(out/'latencies.json',results);event('unified_serving_latency',**{k:v for k,v in row.items() if k not in ('raw_ms','resident_pool')})
        # Real eight-task resident bank, same uniform traffic for each family.
        for batch in spec['batches']:
            for concurrency in spec['concurrency']:
              for traffic in spec.get('traffic',('uniform',)):
                requests=[cases[i]['case'] for i in request_plan(len(cases),spec['requests'],traffic)];rows=[]
                gc.collect();cp.get_default_memory_pool().free_all_blocks()
                def invoke(item):
                    i,cid=item;start=perf_counter_ns()
                    stream_request(gpu[cid],inputs[cid][np.arange(batch)%len(inputs[cid])])
                    return dict(request=i,case=cid,start_ns=start,end_ns=perf_counter_ns())
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    # Measure steady requests, excluding fresh thread/CUDA setup.
                    list(executor.map(invoke,[(-1,cid) for cid in requests[:max(concurrency,len(cases))]]))
                    begin=perf_counter_ns()
                    pending=[executor.submit(invoke,item) for item in enumerate(requests)]
                    for future in as_completed(pending):
                        row=future.result()
                        rows.append(row);write(out/'bank_requests_latest.json',dict(bank=family,batch=batch,concurrency=concurrency,traffic=traffic,requests=rows))
                seconds=(perf_counter_ns()-begin)/1e9
                bank_measurements.append(dict(bank=family,batch=batch,concurrency=concurrency,traffic=traffic,requests=rows,seconds=seconds,
                    samples_per_second=len(requests)*batch/seconds,resident_device_array_bytes=resident,
                    pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),gpu_ownership=ownership,
                    driver_bytes_sample=gpu_processes().get(ownership['pid']) if ownership['verified'] else None,
                    **summary([(r['end_ns']-r['start_ns'])/1e6 for r in rows])))
                write(out/'bank_measurements.json',bank_measurements)
        del gpu,g;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    case_representations=[dict(case=c['case'],path=str(out/'frontier_cases'/c['case']/'case.json'),sha256=digest(out/'frontier_cases'/c['case']/'case.json')) for c in cases]
    result=dict(status='unified_serving_correctness_complete' if spec.get('correctness_only') else 'unified_serving_complete',environment=environment,seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),case_representations=case_representations,
        cases=spec['cases'],matched_native=matched,parity=parity,latencies=results,banks=bank_measurements,prefix_memory=prefix_memory,training_gate=training_gate,elapsed_seconds=perf_counter()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,test_evaluated=False,
        limitations=['32 requests are not a precise tail-latency estimate','model-bank quotas are assigned operational limits, not physical A100 exhaustion',
        'pool-reserved bytes include reusable workspace; CUDA context and library costs separate from logical arrays','Bibtex uses labelled wide fallback',
        'D-selected quality is exploratory; no independent generalization claim',
        'process RSS includes scientific manifests and analysis, not a minimal production daemon'])
    write(out/'report.json',result);event('unified_serving_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
