"""Post-seal evaluation of frozen T1 choices; no fit and no GPU execution."""
from pathlib import Path
from dataclasses import asdict
import gc,json,subprocess,sys
import numpy as np
from budget_quality_worker import digest,locked,write,event,native_load
from budget_quality_core import quality,paired_bootstrap
from abl_quality_core import array_ledger,fitted_control
from abl_quality_evaluation import table,difference


def evaluate_case(cfg,entry,seal_sha,output):
    case=locked(entry['path'],entry['sha256']);record=case['record'];cid=case['case_id']
    for src,key in ((record['case'],'split_file'),(record['dataset'],'path')):
        expected=src['split_sha256'] if key=='split_file' else src['sha256']
        if digest(src[key])!=expected:raise RuntimeError('frozen input changed')
    with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(record['dataset']['path'],allow_pickle=False) as f:
        x=f['features'][ids].astype(np.float32);y=f['target'][ids].astype(np.int64)
    directory=output/'evaluated'/cid;directory.mkdir(parents=True,exist_ok=True)
    target=directory/'target.npz';np.savez_compressed(target,target=y,indices=ids)
    wanted=sorted({c['chosen']['uid'] for c in case['choices'] if c['chosen'] is not None})
    raw={};prediction_records={}
    for uid in wanted:
        meta=case['metas'][uid]
        if digest(meta['artifact'])!=meta['sha256']:raise RuntimeError('selected model changed')
        model=native_load(meta['artifact']);ledger=array_ledger(model)
        if ledger!=meta['ledger']:raise RuntimeError('selected model ledger changed')
        z=model.prepare_inference().raw(x)
        if z.shape!=(len(y),record['case']['outputs']) or not np.isfinite(z).all():
            raise RuntimeError('invalid test output domain')
        path=directory/(uid.replace('/','-')+'.npz');np.savez_compressed(path,raw=z)
        prediction_records[uid]=dict(path=str(path),sha256=digest(path));raw[uid]=z
        del model
    fit_rows={r['uid']:r for r in case['rows']};points=[];metric_cache={}
    for choice in case['choices']:
        point={k:v for k,v in choice.items() if k!='chosen'};chosen=choice['chosen']
        if chosen is None:
            points.append(dict(point,status='no_feasible_candidate'));continue
        uid=chosen['uid'];meta=case['metas'][uid];control=fitted_control(fit_rows[uid],chosen['policy'])
        key=(uid,control.temperature,tuple(control.shifts))
        if key not in metric_cache:metric_cache[key]=quality(y,raw[uid],control.temperature,control.shifts)
        family=meta['family'];trajectory=locked(next(t['path'] for t in case['trajectories']
            if t['mode']==('full' if family=='posthoc' else family)),next(t['sha256'] for t in case['trajectories']
            if t['mode']==('full' if family=='posthoc' else family)))
        cycles=[c for c in trajectory['cycles'] if c['block']<=meta['block']]
        projections=sum(c['action']=='projection' for c in cycles)
        compact_blocks=sum(c['initial_from_representation']=='low_rank' for c in cycles)
        point.update(status='evaluated',uid=uid,family=family,policy=chosen['policy'],control=asdict(control),
            test=metric_cache[key],validation=chosen['validation'],cost_bytes=meta['cost_bytes'],
            parameters=meta['ledger']['parameters'],coefficient_values=sum(int(np.prod(a['shape']))
                for a in meta['ledger']['arrays'] if a['name'] in ('coefficients','basis','values','bias','decision_shifts')),
            rank=meta['rank'],nnz=meta['nnz'],trees=meta['rounds'],leaves=meta['leaves'],nodes=meta['internal_nodes'],
            projection_events_before_snapshot=projections,compact_gradient_blocks_before_snapshot=compact_blocks,
            actual_compact_gradient_training=bool(compact_blocks),
            training_prefix_projection_seconds=sum(c['projection_seconds'] for c in cycles),
            trajectory_elapsed_seconds=trajectory['elapsed_seconds'],
            training_peak_pool_bytes=max((c['fit']['training_pool_peak_bytes'] for c in trajectory['cycles'] if c['fit']),default=0),
            training_peak_cpu_rss_bytes=max((c['cpu_process_peak_rss_bytes'] or 0 for c in trajectory['cycles']),default=0),
            training_error_state_bytes=max((c['error_state_bytes'] for c in cycles),default=0),
            compressed_posthoc=bool(family=='posthoc' and not meta.get('ordinary_fallback',False)))
        if point['group']!='full_unbounded' and point['cost_bytes']>case['budget_bytes']:
            raise RuntimeError('selected deployment exceeds cap')
        points.append(point)
    by={(p['mode'],p['objective'],p['group']):p for p in points};intervals=[];seen={}
    from audit_abl_quality import guard
    for p in points:
        if p['status']!='evaluated':continue
        for opponent in ('full_unbounded','native_budget','posthoc'):
            other=by[p['mode'],p['objective'],opponent]
            if other['status']!='evaluated':continue
            p['versus_'+opponent]=dict(reference_uid=other['uid'],**difference(p['test'],other['test']))
            if opponent=='full_unbounded':p['full_reference_guard_5percent']=guard(p['test'],other['test'])
            if p['mode'] not in ('raw','temperature_shifts') or p['group'] not in ('posthoc','fixed','joint'):continue
            a=(p['uid'],p['control']['temperature'],tuple(p['control']['shifts']))
            b=(other['uid'],other['control']['temperature'],tuple(other['control']['shifts']))
            pair=(a,b)
            if pair not in seen:
                item=dict(id=len(intervals),left_uid=a[0],right_uid=b[0],left_control=p['control'],right_control=other['control'],
                    **paired_bootstrap(y,(raw[a[0]],a[1],a[2]),(raw[b[0]],b[1],b[2]),repeats=500,seed=20261026))
                seen[pair]=item['id'];intervals.append(item)
            p['versus_'+opponent]['interval_id']=seen[pair]
    result=dict(case=cid,points=points,bootstrap_intervals=intervals,predictions=prediction_records,
        target=dict(path=str(target),sha256=digest(target)),source_development=entry,
        pretest_seal_sha256=seal_sha,test_used_for_selection=False,new_tree_fits=0)
    path=directory/'evaluation.json';write(path,result)
    flat=[]
    for p in points:
        row={k:v for k,v in p.items() if isinstance(v,(str,int,float,bool)) or v is None}
        row.update({'test_'+k:v for k,v in p.get('test',{}).items()});flat.append(row)
    table(directory/'points.csv',flat)
    event('abl_training_case_test_evaluated',case=cid,points=len(points),path=str(path),sha256=digest(path))
    del raw;gc.collect()
    return dict(case=cid,path=str(path),sha256=digest(path),points=len(points),bootstrap_pairs=len(intervals))


def run(cfg,output):
    source=cfg['abl_training_development_completed'];development=locked(source['path'],source['sha256'])
    if development['status']!='training_development_complete' or development['test_accessed']:
        raise RuntimeError('sealed development required')
    root=Path(source['path']).parent;seal=locked(root/'pretest_choices.json',development['pretest_seal_sha256'])
    audit=locked(root/'development_audit.json',seal['development_audit_sha256'])
    if seal['status']!='abl_training_choices_sealed_before_test' or audit['status']!='passed' or len(seal['cases'])!=9:
        raise RuntimeError('complete nine-case pretest audit required')
    if seal['cases']!=development['cases']:raise RuntimeError('sealed cases differ')
    from abl_quality_worker import check_native_environment
    check_native_environment(cfg)
    results=[evaluate_case(cfg,e,development['pretest_seal_sha256'],output) for e in seal['cases']]
    result=dict(status='complete',source=source,evaluations=results,training_ledger=development['training_ledger'],
        pretest_seal_sha256=development['pretest_seal_sha256'],new_tree_fits=0,gpu_execution=False,
        uncertainty_scope='500 paired row repeats per frozen pair; three descriptive outer seeds per task; reused exploratory test',
        guard='5percent jointly NLL macro-F1 weighted-F1 versus separately chosen full control',
        training_memory_scope='observed owned worker pool and RSS; not a claim of r-only temporary G/H',
        physical_bank_capacity_measured=False)
    from audit_abl_training_final import audit as independent_audit
    audit_path=output/'audit_report.json';write(audit_path,independent_audit(result))
    result['independent_audit']=dict(path=str(audit_path),sha256=digest(audit_path),status='passed')
    path=output/'report.json';write(path,result)
    event('abl_training_final_complete',path=str(path),sha256=digest(path),cases=len(results))


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));output=Path(sys.argv[2])
    native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    if '--native' not in sys.argv:
        subprocess.run([native,__file__,sys.argv[1],sys.argv[2],'--native'],check=True,timeout=6000);return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(cfg,output)


if __name__=='__main__':main()
