"""Expand unchanged history methods using sealed H-VAL controls and pilot."""
import json
from pathlib import Path
import numpy as np
from experiments.a100_runner import sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.absolute_history_study import absolute_history_training


def case_cost(record):
    return (sum(c['fit']['trees'] for t in record['trajectories'] for c in t['cycles'] if 'fit' in c)
            +sum(f['trees'] for f in record['full_fits']))


def match_reused_control(record,control):
    if (record['full']!=control['full'] or record['reference_validation']!=control['reference_validation']
            or [d for d in record['decisions'] if not d['method'].startswith('gh_')]!=control['decisions']):
        raise RuntimeError('cannot reuse history under changed full/baseline controls')


def expanded_history_training(config,report,output):
    rp=Path(config['absolute_roles_manifest'])
    cp=Path(config['cumulative_development_manifest'])
    pp=Path(config['history_pilot_manifest'])
    for path,key in ((rp,'absolute_roles_manifest_sha256'),(cp,'cumulative_development_manifest_sha256'),
                     (pp,'history_pilot_manifest_sha256')):
        if sha256(path)!=config[key]: raise RuntimeError('frozen expanded history dependency changed')
    roles=json.loads(rp.read_text(encoding='utf-8'))
    controls=json.loads(cp.read_text(encoding='utf-8'))
    pilot=json.loads(pp.read_text(encoding='utf-8'))
    for rows in (roles['cases'], controls['records'], pilot['records']):
        if len({(r['dataset'],r['seed']) for r in rows})!=len(rows):
            raise RuntimeError('duplicate expanded history case')
    if {(r['dataset'],r['seed']) for r in controls['records']}!={(c['dataset'],c['seed']) for c in roles['cases']}:
        raise RuntimeError('expanded controls incomplete')
    if not {(r['dataset'],r['seed']) for r in pilot['records']} <= {(c['dataset'],c['seed']) for c in roles['cases']}:
        raise RuntimeError('history pilot contains unregistered case')
    reused,remaining=[],[]
    for case in roles['cases']:
        old=next((r for r in pilot['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed']),None)
        if old is None:
            remaining.append(case)
            continue
        control=next(r for r in controls['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed'])
        match_reused_control(old,control)
        x,_=load_expanded(case['dataset'],roles)
        expected,actual=load_roles(old,len(x)),load_roles(case,len(x))
        for key in expected:
            if not np.array_equal(expected[key],actual[key]): raise RuntimeError('history reused roles changed')
        reused.append(dict(old,reused=True))
    reused_trees=sum(case_cost(r) for r in reused)
    if reused_trees!=config['expected_reused_history_trees']:
        raise RuntimeError('history pilot price changed')
    path=output/'new_roles_manifest.json'
    write_report(path,dict(roles,cases=remaining,planned_cases=len(remaining)))
    new_output=output/'new'
    new_output.mkdir()
    cfg=dict(config,cumulative_roles_manifest=str(path),cumulative_roles_manifest_sha256=sha256(path))
    fresh={}
    if remaining: absolute_history_training(cfg,fresh,new_output)
    else: fresh=dict(records=[],tree_fits=0,gpu_observations=[])
    write_report(output/'new_training_report.json',fresh)
    by_id={(r['dataset'],r['seed']):r for r in [*reused,*fresh['records']]}
    records=[by_id[(c['dataset'],c['seed'])] for c in roles['cases']]
    manifest=dict(records=records,tree_fits=fresh['tree_fits'],reused_tree_fits=reused_trees,
        source_roles_manifest=str(rp),source_roles_sha256=sha256(rp),
        reused_records_manifest=str(pp),reused_records_sha256=sha256(pp),
        source_controls_manifest=str(cp),source_controls_sha256=sha256(cp),
        baseline_training_cost=sum(case_cost(r) for r in controls['records']),test_evaluated=False)
    write_report(output/'cumulative_development_manifest.json',manifest)
    report.update(status='development_complete',cases=len(records),new_tree_fits=fresh['tree_fits'],
        reused_tree_fits=reused_trees,test_evaluated=False,gpu_observations=fresh['gpu_observations'],
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('expanded_history_development_complete',**{k:report[k] for k in
        ('cases','new_tree_fits','reused_tree_fits','development_manifest_sha256')})
