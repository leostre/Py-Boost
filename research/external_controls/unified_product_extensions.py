"""Bounded budgeted ensembles and explicitly compositional mixed-bank estimates."""
from pathlib import Path
from dataclasses import replace
from itertools import combinations
from time import perf_counter_ns
import gc,json,sys
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from experiments.block_leaf_model import BlockLeafModel
from unified_training import evaluate,serialize,quality_row
from unified_frontier_core import make_folds
from unified_serving_core import coalesce,summary,quality_key
from unified_serving_gpu import BlockGpu
from unified_recalibration import audit_case
from storage_graph import BufferSpan,union_bytes


def weighted_pair(first,second,weight):
    if not 0<weight<1:raise ValueError('strict convex weight required')
    if first.task!=second.task or first.n_features!=second.n_features or first.bias.shape!=second.bias.shape:
        raise ValueError('ensemble domains differ')
    blocks=[]
    for model,w in ((first,1-weight),(second,weight)):
        for block in model.blocks:
            blocks.append(replace(block,base=replace(block.base,coefficients=(block.base.coefficients*w).astype(np.float32)),values=(block.values*w).astype(np.float32)))
    bias=((1-weight)*first.bias+weight*second.bias).astype(np.float32)
    return coalesce(BlockLeafModel(tuple(blocks),bias,np.zeros_like(bias)))


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['unified_product'];native=locked(spec['native']['path'],spec['native']['sha256']);external=locked(spec['external']['path'],spec['external']['sha256'])
    assert native['status']=='unified_serving_complete' and external['status']=='unified_external_serving_complete'
    original=locked(native['seal']['path'],native['seal']['sha256']);all_cases={r['case']:locked(r['path'],r['sha256']) for r in native['case_representations']}
    old_native={}
    for ref in native['matched_native']:
        item=locked(ref['path'],ref['sha256']);old_native[item['case'],ref['family']]=item
    inventory=[];cases=[];ensemble_checks=[];ensembles={};controls={};features={}
    for cid in native['cases']:
        case=all_cases[cid];record=case['record'];data=data_before_test(record);directory=out/cid;directory.mkdir(exist_ok=True)
        models=list(case['models'])
        for (other,family),item in old_native.items():
            if other==cid and family!='full':models.extend(item['models'])
        cap=int(.7*case['reference_bytes']);base=[]
        for family in sorted({m['family'] for m in models}):
            pool=[m for m in models if m['family']==family and m['cost_bytes']<=cap/2]
            if pool:base.append(min(pool,key=lambda m:quality_key(m,'multiclass')))
        folds=make_folds(data['dev_y'],'multiclass',record['case']['seed']+20261028);candidates=[]
        for index,(a,b) in enumerate(combinations(base,2)):
            if index>=6:break
            for weight in (.25,.5,.75):
                first,second=load_protected(a['artifact']),load_protected(b['artifact'])
                model=weighted_pair(first,second,weight);cost=block_cost(model)['cost_bytes']
                if cost>cap:continue
                uid=f'pair{index}-w{weight:g}';p=directory/uid;p.mkdir(exist_ok=True);raw=model.raw(data['dev_x'])
                expected=(1-weight)*first.raw(data['dev_x'][:17])+weight*second.raw(data['dev_x'][:17])
                assert np.max(np.abs(raw[:17]-expected))<2e-5
                ev=evaluate(raw,data['dev_y'],'multiclass',folds,cfg,p)
                candidates.append(serialize(model,raw,ev,p,uid,dict(family='ensemble',components=[a['uid'],b['uid']],weight=weight,budget=cap)))
        if not candidates:
            ensemble_checks.append(dict(case=cid,status='no_distinct_family_pair_in_this_budget_grid',base_options=len(base)));continue
        check=audit_case(dict(task='multiclass',models=candidates),data);chosen=min(candidates,key=lambda m:quality_key(m,'multiclass'))
        single_pool=[m for m in models if m['cost_bytes']<=cap];best_single=min(single_pool,key=lambda m:quality_key(m,'multiclass'))
        q=quality_row(chosen['evaluations'],'multiclass');single_q=quality_row(best_single['evaluations'],'multiclass')['quality']
        justified=q['quality']['loss']<single_q['loss']
        ensemble_checks.append(dict(case=cid,status='evaluated',variants=len(candidates),chosen=chosen['uid'],quality=q['quality'],best_single=best_single['uid'],best_single_quality=single_q,improves_single_D_nll=justified,audit=check))
        inventory.append(dict(case=cid,bank='ensemble',uid=chosen['uid'],artifact=chosen['artifact'],sha256=chosen['sha256'],control=q['refit_control'],quality=q['quality'],cost_bytes=chosen['cost_bytes']))
        cases.append(dict(case=cid,record=record));ensembles[cid]=chosen;controls[cid]=q['refit_control'];features[cid]=data['dev_x']
    seal=dict(inventory=inventory,source_cases=cases,checks=ensemble_checks,selection_scope='D only; no test labels',test_evaluated=False)
    write(out/'pretest_seal.json',seal)
    from quality_worker import gpu_guard
    gpu_guard(cfg)  # no latency run when approved free-memory margin is absent
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=16*1024**3);latencies=[];parity=[]
    for cid,item in ensembles.items():
        model=load_protected(item['artifact']);control=controls[cid];gpu=BlockGpu(model,control['temperature'],control['shifts'])
        for batch in spec['batches']:
            x=features[cid][np.arange(batch)%len(features[cid])];actual,labels=gpu.request(x)
            from scipy.special import softmax
            expected=np.clip(softmax(model.raw(x)/control['temperature'],axis=1),1e-7,1-1e-7)
            error=float(np.max(np.abs(actual-expected)));assert error<2e-5
            np.testing.assert_array_equal(labels,(model.raw(x)/control['temperature']-control['shifts']).argmax(1))
            parity.append(dict(case=cid,batch=batch,probability_error=error))
            gc.collect();cp.get_default_memory_pool().free_all_blocks();times=[]
            for request in range(spec['requests']):
                begin=perf_counter_ns();gpu.request(x);times.append((perf_counter_ns()-begin)/1e6)
                write(out/'latest_request.json',dict(case=cid,batch=batch,raw_ms=times))
            latencies.append(dict(case=cid,bank='ensemble',batch=batch,raw_ms=times,**summary(times),resident_arrays_bytes=gpu.resident_device_array_bytes,pool_reserved_bytes=cp.get_default_memory_pool().total_bytes()))
            write(out/'latencies.json',latencies)
        del gpu;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    # Task-level routes use D quality and previously measured latencies only.
    # This is a TRACE REPLAY, not a newly measured mixed Torch/CuPy service.
    routes=[];native_inventory=original['inventory']+inventory
    for batch in spec['batches']:
        for slo in (1.,10.,1000.,None):
            selected=[]
            for cid in native['cases']:
                point=next(r for r in external['tfm_points'] if r['case']==cid);q=quality_row(point['evaluations'],'multiclass')['quality']
                tfm_trace=next(r for r in external['timings'] if r['family']=='tabpfn_auto' and r['batch']==batch and r['traffic']=='uniform')
                samples=[r['ms'] for r in tfm_trace['requests'] if r['case']==cid]
                options=[dict(case=cid,source='tabpfn_auto',quality=q,latency_ms=float(np.median(samples)))]
                for m in native_inventory:
                    if m['case']!=cid:continue
                    if m['bank']=='ensemble' and not next(r for r in ensemble_checks if r['case']==cid).get('improves_single_D_nll',False):continue
                    timing=next(r for r in (latencies if m['bank']=='ensemble' else native['latencies']) if r['case']==cid and r['bank']==m['bank'] and r['batch']==batch)
                    options.append(dict(case=cid,source=m['bank'],quality=m['quality'],latency_ms=timing['p50_ms']))
                allowed=[o for o in options if slo is None or o['latency_ms']<=slo]
                chosen=min(allowed,key=lambda o:(o['quality']['loss'],o['latency_ms'])) if allowed else min(options,key=lambda o:o['latency_ms'])
                selected.append(dict(chosen,slo_satisfied=bool(allowed)))
            tfm_ids={r['case'] for r in selected if r['source']=='tabpfn_auto'}
            spans=[BufferSpan(**s) for cid in tfm_ids for s in external['context_spans'][cid]]
            gpu_bytes=union_bytes([s for s in spans if s.device.startswith('cuda')]);cpu_bytes=union_bytes([s for s in spans if s.device=='cpu'])
            for r in selected:
                if r['source']=='tabpfn_auto':continue
                timing=next(t for t in (latencies if r['source']=='ensemble' else native['parity']) if t['case']==r['case'] and t['bank']==r['source'] and (t.get('batch',batch)==batch))
                gpu_bytes+=timing.get('resident_arrays_bytes',timing.get('resident_array_bytes',0))
            by_case={r['case']:r for r in selected};stitched=[]
            for request in tfm_trace['requests']:
                choice=by_case[request['case']]
                if choice['source']=='tabpfn_auto':ms=request['ms']
                elif choice['source']=='ensemble':
                    row=next(r for r in latencies if r['case']==request['case'] and r['batch']==batch);ms=row['raw_ms'][request['request']%len(row['raw_ms'])]
                else:
                    row=next(r for r in native['banks'] if r['bank']==choice['source'] and r['batch']==batch and r['traffic']=='uniform' and r['concurrency']==1)
                    timing=next(r for r in row['requests'] if r['request']==request['request']);assert timing['case']==request['case']
                    ms=(timing['end_ns']-timing['start_ns'])/1e6
                stitched.append(dict(request=request['request'],case=request['case'],source=choice['source'],estimated_ms=ms))
            routes.append(dict(batch=batch,slo_ms=slo,choices=selected,logical_gpu_array_bytes=gpu_bytes,tfm_reachable_cpu_bytes=cpu_bytes,
                estimated_trace=stitched,estimated_trace_summary=summary([r['estimated_ms'] for r in stitched]),
                interpretation='compositional estimate only; no measured mixed-service latency, driver context, allocator workspace or process-RSS prediction'))
    write(out/'routes_sealed_without_test.json',routes)
    result=dict(status='unified_product_extensions_complete',checks=ensemble_checks,latencies=latencies,parity=parity,routes=routes,
        seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),new_tree_fits=0,test_evaluated=False,
        limitations=['ensemble search bounded by six distinct-family pairs and three logit weights','one deployed weighted model still pays every retained tree traversal',
            'mixed-bank routes are D/latency-selected trace estimates; a combined live service was not built or benchmarked'])
    write(out/'report.json',result);event('unified_product_extensions_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
