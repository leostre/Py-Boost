"""Post-seal evaluation only. No model, rank, policy or budget is chosen here."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import csv
import gc
import json
from pathlib import Path

import numpy as np

from budget_quality_worker import digest,write,event
from budget_quality_core import quality,paired_bootstrap
from abl_quality_core import fitted_control
from experiments.shared_leaf_basis import quality_guard


def table(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def difference(left,right):
    return dict(delta_loss=left['loss']-right['loss'],
        delta_f1_macro_pp=100*(left['f1_macro']-right['f1_macro']),
        delta_f1_weighted_pp=100*(left['f1_weighted']-right['f1_weighted']),
        delta_brier=left['brier']-right['brier'])


def control_key(uid,control):
    return uid,float(control.temperature),tuple(control.shifts)


def summary_rows(points):
    groups=defaultdict(list)
    for p in points:
        key=tuple(p[k] for k in ('case','scheme','mode','objective','group','budget_ratio'))
        groups[key].append(p)
    rows=[]
    for key,items in groups.items():
        row=dict(zip(('case','scheme','mode','objective','group','budget_ratio'),key))
        good=[p for p in items if p['status']=='evaluated']
        row.update(internal_repeats=len(items),feasible_repeats=len(good),
            uncertainty_scope='internal selection repartitions of one trained source split; no independent training-seed interval')
        if good:
            for metric in ('loss','f1_macro','f1_weighted','brier'):
                values=np.array([p['test'][metric] for p in good])
                row.update({metric+'_mean':float(values.mean()),metric+'_min':float(values.min()),
                    metric+'_max':float(values.max())})
            row.update(cost_bytes_mean=float(np.mean([p['cost_bytes'] for p in good])),
                nonzero_residual_fraction=float(np.mean([p['nnz']>0 for p in good])),
                own_parent_quality_pass_fraction=float(np.mean([p['own_parent_guard_5percent'] for p in good])))
            for opponent in ('native','tabpfn8'):
                compared=[p['versus_'+opponent] for p in good if 'versus_'+opponent in p]
                if compared:
                    for metric in ('delta_loss','delta_f1_macro_pp'):
                        row[opponent+'_'+metric+'_mean']=float(np.mean([x[metric] for x in compared]))
        rows.append(row)
    return rows


def flat_point(point):
    row={k:point.get(k) for k in ('case','plan','scheme','repeat','mode','objective','group',
        'budget_ratio','budget_bytes','status','uid','policy','rank','nnz','cost_bytes',
        'parent_uid','own_parent_guard_5percent')}
    for section in ('test','validation','versus_native','versus_tabpfn8'):
        row.update({section+'_'+k:v for k,v in point.get(section,{}).items() if isinstance(v,(int,float,bool))})
    if 'control' in point:row['temperature']=point['control']['temperature']
    return row


def evaluate_case(cfg,output,seal_path,case):
    # This is the sole reader of test targets; it verifies the whole-study seal.
    from abl_quality_worker import case_test_predictions,memory_guard
    target,predictions,prediction_audit=case_test_predictions(cfg,case,seal_path,output)
    raw={}
    for uid,item in predictions.items():
        if digest(item['path'])!=item['sha256']:raise RuntimeError('new test prediction cache changed')
        with np.load(item['path'],allow_pickle=False) as f:raw[uid]=f['raw'].copy()
    if any(z.shape!=(len(target),case['record']['case']['outputs']) or not np.isfinite(z).all() for z in raw.values()):
        raise RuntimeError('post-seal test logits are not finite canonical NxC arrays')
    metas=case['metas'];fit_rows={(r['uid'],r['plan']):r for r in case['rows']}
    metric_cache={};interval_cache={};intervals=[]

    def measured(uid,policy,plan):
        control=fitted_control(fit_rows[uid,plan],policy)
        key=control_key(uid,control)
        if key not in metric_cache:
            metric_cache[key]=quality(target,raw[uid],control.temperature,control.shifts)
        return metric_cache[key],control

    def result(choice):
        item={k:v for k,v in choice.items() if k!='chosen'}
        item['case']=case['case_id'];chosen=choice['chosen']
        if chosen is None:return dict(item,status='no_feasible_candidate')
        uid,policy=chosen['uid'],chosen['policy'];meta=metas[uid]
        q,control=measured(uid,policy,choice['plan'])
        parent=meta.get('parent_uid',uid)
        parent_q,_=measured(parent,policy,choice['plan'])
        return dict(item,status='evaluated',uid=uid,policy=policy,rank=meta['rank'],nnz=meta['nnz'],
            cost_bytes=meta['cost_bytes'],parent_uid=parent,control=asdict(control),
            validation=chosen['validation'],test=q,own_parent_test=parent_q,
            own_parent_guard_5percent=quality_guard(q,parent_q,.05))

    external=[result(c) for c in case['external_choices']]
    external_by={(p['plan'],p['mode'],p['objective'],p['group']):p for p in external}
    points=[result(c) for c in case['budget_choices']]
    by_key={(p['plan'],p['mode'],p['objective'],p['group'],p['budget_ratio']):p for p in points}

    def interval(point,other):
        a=control_key(point['uid'],fitted_control(fit_rows[point['uid'],point['plan']],point['policy']))
        b=control_key(other['uid'],fitted_control(fit_rows[other['uid'],other['plan']],other['policy']))
        key=(a,b)
        if key not in interval_cache:
            index=len(intervals)
            value=paired_bootstrap(target,(raw[a[0]],a[1],a[2]),(raw[b[0]],b[1],b[2]),
                repeats=cfg['abl_quality']['bootstrap_repeats'],seed=20261026)
            item=dict(id=index,left_uid=a[0],right_uid=b[0],left_control=point['control'],
                right_control=other['control'],**value)
            intervals.append(item);interval_cache[key]=index
        return interval_cache[key]

    for point in points:
        if point['status']!='evaluated':continue
        plan,mode,objective=(point[k] for k in ('plan','mode','objective'))
        native=by_key[plan,mode,objective,'native',point['budget_ratio']]
        tfm=external_by[plan,mode,objective,'tabpfn8']
        for name,other in (('native',native),('tabpfn8',tfm)):
            if other['status']!='evaluated':continue
            point['versus_'+name]=dict(reference_uid=other['uid'],**difference(point['test'],other['test']))
            if (point['group'] in ('ab','abl') and point['repeat'] in cfg['abl_quality']['bootstrap_repeat_indices']
                    and mode in cfg['abl_quality']['bootstrap_modes']):
                point['versus_'+name]['interval_id']=interval(point,other)
    for point in external:
        if point['status']=='evaluated':
            tfm=external_by[point['plan'],point['mode'],point['objective'],'tabpfn8']
            point['versus_tabpfn8']=dict(reference_uid=tfm['uid'],**difference(point['test'],tfm['test']))

    raw_diagnostics=[]
    for uid,meta in metas.items():
        if meta['family']!='compact':continue
        parent=meta['parent_uid'];q=quality(target,raw[uid]);parent_q=quality(target,raw[parent])
        raw_diagnostics.append(dict(case=case['case_id'],uid=uid,parent_uid=parent,
            rank=meta['rank'],nnz=meta['nnz'],cost_bytes=meta['cost_bytes'],
            parent_cost_bytes=metas[parent]['cost_bytes'],
            memory_fraction=meta['cost_bytes']/metas[parent]['cost_bytes'],
            **{'test_'+k:v for k,v in q.items()},**difference(q,parent_q),
            own_parent_guard_5percent=quality_guard(q,parent_q,.05),
            scope='preregistered raw-grid diagnostic, never test-selected frontier'))
    directory=output/'evaluated'/case['case_id'];directory.mkdir(parents=True,exist_ok=True)
    report=dict(case=case['case_id'],budget_points=points,external_points=external,
        raw_compression=raw_diagnostics,bootstrap_intervals=intervals,
        summary=summary_rows(points),prediction_audit=prediction_audit,
        pretest_seal_sha256=digest(seal_path),test_used_for_selection=False,
        peak_cpu_rss_bytes=memory_guard())
    path=directory/'evaluation.json';write(path,report)
    table(directory/'budget_points.csv',[flat_point(p) for p in points])
    table(directory/'external_points.csv',[flat_point(p) for p in external])
    table(directory/'raw_compression.csv',raw_diagnostics)
    table(directory/'summary.csv',report['summary'])
    event('abl_case_test_evaluated',case=case['case_id'],points=len(points),
        unique_bootstrap_pairs=len(intervals),report_path=str(path),report_sha256=digest(path),
        new_tree_fits=0,test_used_for_selection=False)
    del raw,metric_cache;gc.collect()
    return dict(case=case['case_id'],path=str(path),sha256=digest(path),
        summary=report['summary'],points=len(points),bootstrap_pairs=len(intervals))


def evaluate_study(cfg,output,seal_path):
    from abl_quality_worker import require_seal
    seal=require_seal(seal_path);cases=[]
    for entry in seal['cases']:
        case=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
        cases.append(evaluate_case(cfg,output,seal_path,case))
    return dict(evaluations=cases,test_is_reused_exploratory=True,
        uncertainty_scope='paired fixed-model row bootstrap plus descriptive internal selection-repeat spread; not independent training seeds',
        inference_latency_measured=False,physical_bank_capacity_measured=False)
