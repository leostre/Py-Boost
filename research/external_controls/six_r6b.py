"""Resident task banks with explicit workspace reservation and real concurrency."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading,gc,json,sys,time,subprocess
import numpy as np
from budget_quality_worker import locked,write,digest,event
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from six_quant_core import QuantSnapshot
from six_quant_gpu import QuantPackedGpu
from six_common import load_cases,load_model
from packed_serving_gpu import PackedGpu
from packed_serving_core import prepare_snapshot
from unified_serving_core import bank_frontier


def aligned(size):return ((int(size)+511)//512)*512
def pool_price(s):return sum(aligned(a.nbytes) for a in s.arrays.values())
def make_device(s,t=1.):
    g=next(g for g in (4,2,1) if 8*(s.outputs+g*s.max_width+g*s.outputs*int(s.has_residual)+32*g)<=49152)
    return QuantPackedGpu(s,t,groups=g)


def run(cfg,out):
    import cupy as cp
    out.mkdir(parents=True,exist_ok=True);pool=cp.get_default_memory_pool();pool.set_limit(size=12*1024**3);env=check_native_environment(cfg)
    source=locked(cfg['six']['R6D']['path'],cfg['six']['R6D']['sha256']);cases=[locked(r['path'],r['sha256']) for r in source['cases']]
    sealed_test={c['case']:{m['uid']:m['quality'] for m in c['models']} for c in source['results']}
    features={c['case']:data_before_test(c['record'])['dev_x'] for c in cases};snapshots={};prices={}
    for case in cases:
        for m in case['models']:
            assert digest(m['artifact'])==m['sha256'];s=QuantSnapshot.load(m['artifact']);snapshots[case['case'],m['uid']]=s;prices[case['case'],m['uid']]=pool_price(s)
    rows=[];rng=np.random.default_rng(202610308);capacity_templates={}
    for traffic in ('uniform','skew_priority'):
        weights=np.asarray(next(b for b in source['banks'] if b['traffic']==traffic)['task_weights'])
        frequency=np.asarray(next(b for b in source['banks'] if b['traffic']==traffic)['frequency'])
        for batch in (32,1024):
            for concurrency in (1,4,8):
                workspace_per_stream=max(aligned(batch*4*features[c['case']].shape[1])+
                    aligned(batch*9*c['record']['case']['outputs'])+2048 for c in cases)
                reserve=concurrency*workspace_per_stream
                for family in rng.permutation(['ordinary_int8','ordinary_refit','compact','mixed','mixed_union']).tolist():
                    options=[[dict(uid=m['uid'],cost_bytes=prices[c['case'],m['uid']],utility=float(len(cases)*weights[i]*m['assessment']['quality']['f1_macro']))
                              for m in c['models'] if family in m['catalogues'] or (family=='mixed_union' and any(k in m['catalogues'] for k in ('ordinary_refit','compact'))) ] for i,c in enumerate(cases)]
                    model_budgets=[int(source['full_bank_bytes']*r) for r in (.125,.25,.5,1.)]
                    for ratio,bank in zip((.125,.25,.5,1.),bank_frontier(options,model_budgets)):
                        identity=dict(family=family,traffic=traffic,batch=batch,concurrency=concurrency,ratio=ratio,total_budget=bank['budget']+reserve,workspace_reserved=reserve,
                                      catalogue_sizes=[len(o) for o in options],larger_product_catalogue=family=='mixed_union')
                        if bank['status']!='feasible':rows.append(dict(identity,status='infeasible'));continue
                        choices=bank['choices'];selected=[next(m for m in c['models'] if m['uid']==uid) for c,uid in zip(cases,choices)]
                        devices=[make_device(snapshots[c['case'],m['uid']],m['assessment']['refit_control']['temperature']) for c,m in zip(cases,selected)]
                        quality_values=[sealed_test[c['case']][m['uid']]['f1_macro'] for c,m in zip(cases,selected)]
                        if traffic=='uniform' and batch==32 and concurrency==1 and ratio==.25 and family in ('ordinary_refit','mixed','mixed_union'):
                            capacity_templates[family]=[(snapshots[c['case'],m['uid']],m['assessment']['refit_control']['temperature']) for c,m in zip(cases,selected)]
                        local=threading.local();peak=[pool.used_bytes()];allocator=cp.cuda.get_allocator();lock=threading.Lock()
                        def tracked(size):
                            ptr=allocator(size)
                            with lock:peak[0]=max(peak[0],pool.used_bytes())
                            return ptr
                        def request(item):
                            task,offset,submitted=item;begin=time.perf_counter_ns()
                            if not hasattr(local,'stream'):local.stream=cp.cuda.Stream(non_blocking=True)
                            x=features[cases[task]['case']];xx=np.ascontiguousarray(x[(np.arange(batch)+offset)%len(x)])
                            with cp.cuda.using_allocator(tracked),local.stream:devices[task].request(xx)
                            end=time.perf_counter_ns()
                            return dict(service_ms=(end-begin)/1e6,queued_request_ms=(end-submitted)/1e6)
                        sessions=[]
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            list(executor.map(request,[(i,0,time.perf_counter_ns()) for i in range(len(cases))]));gc.collect();pool.free_all_blocks();baseline=pool.used_bytes();peak[0]=baseline
                            paired_rng=np.random.default_rng(202610308+batch*10+concurrency+(100000 if traffic=='skew_priority' else 0))
                            for session in range(3):
                                schedule=paired_rng.choice(len(cases),32,p=frequency).tolist()
                                start=time.perf_counter_ns();samples=list(executor.map(request,[(task,409+17*j+53*session,start) for j,task in enumerate(schedule)]));seconds=(time.perf_counter_ns()-start)/1e9
                                sessions.append(dict(session=session,request_ms=[s['service_ms'] for s in samples],queued_request_ms=[s['queued_request_ms'] for s in samples],task_schedule=schedule,wall_seconds=seconds,requests_per_second=32/seconds))
                        actual_peak=peak[0];budget_pass=actual_peak<=identity['total_budget'];assert budget_pass,(identity,actual_peak,baseline)
                        item=dict(identity,status='complete',choices=choices,sessions=sessions,model_array_bytes=sum(d.resident_device_array_bytes for d in devices),
                                  retained_pool_bytes=baseline,own_pool_peak_bytes=actual_peak,total_budget_pass=budget_pass,
                                  test_equal_task_macro_f1=float(np.mean(quality_values)),test_weighted_macro_f1=float(np.asarray(quality_values)@weights))
                        rows.append(item);write(out/'serving_progress.json',dict(rows=rows));event('R6B_bank',**identity,peak=actual_peak)
                        del devices,request,tracked,executor;gc.collect();pool.free_all_blocks()
    # Capacity uses copies of twelve real tasks; it does not create new tasks.
    capacities=[]
    for family,template in capacity_templates.items():
        per_bank=sum(pool_price(s) for s,_ in template);reserve=4*max(aligned(32*4*s.features)+aligned(32*9*s.outputs)+2048 for s,_ in template)
        for total in (64*1024**2,256*1024**2,1024*1024**2):
            allocated=[];gc.collect();pool.free_all_blocks();before=pool.used_bytes();copies=0
            while copies<512 and pool.used_bytes()-before+per_bank+reserve<=total:
                allocated.extend(make_device(s,t) for s,t in template);copies+=1
            actual=pool.used_bytes()-before;assert actual+reserve<=total
            capacities.append(dict(family=family,budget=total,bank_copies=copies,model_copies=copies*len(cases),distinct_tasks=len(cases),
                owned_pool_bytes=actual,workspace_reserved=reserve,censored_at512=copies==512,replica_stress_only=True))
            event('R6B_capacity',**capacities[-1]);del allocated;gc.collect();pool.free_all_blocks()
    write(out/'report.json',dict(status='R6B_complete',rows=rows,capacity=capacities,new_trees=0,environment=env,
          budget_scope='actual allocator-rounded model arrays plus reserved concurrent request workspaces; context/driver excluded; physical peak checked',
          tail_scope='three32-request bursts, exploratory throughput and service/queue latency, not SLA certification; paired schedules across families and budgets; randomized bank-family order'))
    event('R6B_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
