"""Post-seal test of protected-increment bases; no new optimization."""
from pathlib import Path
from dataclasses import asdict
import json,gc,subprocess,sys
import numpy as np
from budget_quality_worker import locked,digest,write,event
from budget_quality_core import quality
from abl_quality_core import fitted_control
from abl_protected_core import load_protected,block_cost
from experiments.output_calibration_core import calibrated_quality
from audit_abl_quality import guard
from abl_transfer_final import paired_interval


def evaluate(entry,seal_sha,out):
    case=locked(entry['path'],entry['sha256']);cid=case['case_id'];directory=out/'evaluated'/cid;directory.mkdir(parents=True,exist_ok=True)
    record=case['record']
    if digest(record['case']['split_file'])!=record['case']['split_sha256'] or digest(record['dataset']['path'])!=record['dataset']['sha256']:
        raise RuntimeError('protected test roles changed')
    with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(record['dataset']['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
    target=directory/'targets.npz';np.savez_compressed(target,target=y,indices=ids)
    wanted={c['chosen']['uid'] for c in case['choices'] if c['chosen']};raw={};predictions={}
    for uid in sorted(wanted):
        meta=case['metas'][uid]
        if digest(meta['artifact'])!=meta['sha256']:raise RuntimeError('protected model changed')
        m=load_protected(meta['artifact']);assert block_cost(m)['cost_bytes']==meta['cost_bytes']
        z=m.raw(x);path=directory/(uid.replace('/','-')+'.npz');np.savez_compressed(path,raw=z)
        predictions[uid]=dict(path=str(path),sha256=digest(path));raw[uid]=z
    rows={r['uid']:r for r in case['rows']};points=[];cache={}
    for c in case['choices']:
        p={k:v for k,v in c.items() if k!='chosen'};chosen=c['chosen']
        if chosen is None:points.append(dict(p,status='no_feasible_candidate'));continue
        uid=chosen['uid'];meta=case['metas'][uid];cp=asdict(fitted_control(rows[uid],chosen['policy']));key=(uid,chosen['policy'])
        if key not in cache:
            cache[key]=(quality(y,raw[uid],cp['temperature'],cp['shifts']) if case['task']=='multiclass'
                else calibrated_quality(y,raw[uid],'multilabel',np.asarray(cp['shifts'])))
        p.update(status='evaluated',uid=uid,family=meta['family'],policy=chosen['policy'],control=cp,validation=chosen['validation'],
            test=cache[key],cost_bytes=meta['cost_bytes'],parameters=meta['parameters'],ranks=meta['ranks'],nnz=meta['nnz'],
            blocks=meta['blocks'],trees=meta['trees'],coefficient_values=meta['coefficient_values'],basis_values=meta['basis_values'],
            compressed_blocks=meta['compressed_blocks'],actual_compressed_increment=meta['compressed_blocks']>0)
        if c['group']!='full_unbounded' and p['cost_bytes']>case['budget_bytes']:raise RuntimeError('protected selection exceeds cap')
        points.append(p)
    by={(p['mode'],p['objective'],p['group']):p for p in points};intervals=[];seen={}
    for p in points:
        if p['status']!='evaluated':continue
        full=by[p['mode'],p['objective'],'full_unbounded'];p['full_reference_guard_5percent']=guard(p['test'],full['test'])
        p['full_reference_uid']=full['uid']
        for opponent in ('native_budget','functional','current','mean'):
            q=by[p['mode'],p['objective'],opponent]
            if q['status']!='evaluated':continue
            delta=dict(reference_uid=q['uid'],delta_loss=p['test']['loss']-q['test']['loss'],
                delta_f1_macro_pp=100*(p['test']['f1_macro']-q['test']['f1_macro']))
            if p['group']=='history' and p['mode']!='raw' and p['objective']=='f1_macro':
                key=(p['uid'],p['policy'],q['uid'],q['policy'])
                if key not in seen:
                    a,b=p['control'],q['control'];value=paired_interval(y,(raw[p['uid']],a['temperature'],a['shifts']),
                        (raw[q['uid']],b['temperature'],b['shifts']),case['task'])
                    seen[key]=len(intervals);intervals.append(dict(id=len(intervals),left_uid=p['uid'],right_uid=q['uid'],
                        left_control=a,right_control=b,**value))
                delta['interval_id']=seen[key]
            p['versus_'+opponent]=delta
    training=[]
    for ref in case['trajectories']:
        t=locked(ref['path'],ref['sha256']);cycles=t['cycles']
        training.append(dict(mode=ref['mode'],elapsed_seconds=t['elapsed_seconds'],stop_reason=t['stop_reason'],
            paid_new_rounds=40*len(cycles),accepted_checkpoints=len(t['models']),
            peak_algorithm_state_bytes=max((c['algorithm_state_bytes'] for c in cycles),default=0),
            peak_diagnostic_state_bytes=max((c['diagnostic_state_bytes'] for c in cycles),default=0),
            peak_previous_basis_bytes=max((c['previous_basis_bytes'] for c in cycles),default=0),
            training_pool_peak_bytes=max((c['fit']['training_pool_peak_bytes'] for c in cycles),default=0),
            cpu_peak_rss_bytes=max((c['cpu_process_peak_rss_bytes'] or 0 for c in cycles),default=0)))
    result=dict(case=cid,task=case['task'],points=points,predictions=predictions,training=training,
        target=dict(path=str(target),sha256=digest(target)),development=entry,bootstrap_intervals=intervals,
        pretest_seal_sha256=seal_sha,test_used_for_selection=False,new_tree_fits=0)
    path=directory/'evaluation.json';write(path,result);event('abl_protected_case_test_complete',case=cid,path=str(path),sha256=digest(path))
    del raw;gc.collect();return dict(case=cid,path=str(path),sha256=digest(path),points=len(points))


def run(cfg,out):
    source=cfg['abl_protected_development_completed'];r=locked(source['path'],source['sha256']);root=Path(source['path']).parent
    if r['status']!='protected_development_complete' or r['test_accessed']:raise RuntimeError('completed protected development required')
    seal=locked(root/'pretest_choices.json',r['pretest_seal_sha256'])
    if seal['status']!='protected_choices_sealed_before_test' or len(seal['cases'])!=9:raise RuntimeError('all nine protected cases required')
    for name,key in (('development_audit.json','development_audit_sha256'),('history_state_audit.json','history_state_audit_sha256')):
        if locked(root/name,seal[key])['status']!='passed':raise RuntimeError('both independent protected audits required')
    from abl_quality_worker import check_native_environment
    check_native_environment(cfg)
    result=dict(status='complete',source=source,evaluations=[evaluate(e,r['pretest_seal_sha256'],out) for e in seal['cases']],
        training_ledger=r['training_ledger'],new_tree_fits=0,gpu_execution=False,
        scope='three reused tasks x three splits; protected old blocks and projected new increments; fixed-model paired bootstrap',
        inference_latency_measured=False)
    from audit_abl_protected_final import audit
    a=out/'audit_report.json';write(a,audit(result));result['audit']=dict(path=str(a),sha256=digest(a),status='passed')
    path=out/'report.json';write(path,result);event('abl_protected_final_complete',path=str(path),sha256=digest(path))


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2]);native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    if '--native' not in sys.argv:
        subprocess.run([native,__file__,sys.argv[1],sys.argv[2],'--native'],check=True,timeout=6000);return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(cfg,out)


if __name__=='__main__':main()
