"""Transfer one frozen reserve policy; retain exact old pilot costs and results."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments.a100_runner import sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.absolute_history_study import absolute_history_training


def keyed(rows):
    keys=[(r['dataset'],r['seed']) for r in rows]
    if len(set(keys))!=len(keys): raise RuntimeError('duplicate reserve case')
    return dict(zip(keys,rows))


def baseline_equal(record,control):
    methods={d['method'] for d in control['decisions']}
    if (not methods or len(methods)!=len(control['decisions'])
            or record['full']!=control['full'] or record['reference_validation']!=control['reference_validation']
            or [d for d in record['decisions'] if d['method'] in methods]!=control['decisions']):
        raise RuntimeError('frozen reserve baselines changed')


def tree_cost(record):
    return (sum(c['fit']['trees'] for t in record['trajectories'] for c in t['cycles'] if 'fit' in c)
            +sum(f['trees'] for f in record['full_fits']))


def reserve_validation_training(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_reserve_validation.py'],check=True,timeout=300)
    rp=Path(config['reserve_roles_manifest'])
    cp=Path(config['cumulative_development_manifest'])
    pp=Path(config['reserve_pilot_manifest'])
    for path,key in ((rp,'reserve_roles_manifest_sha256'),(cp,'cumulative_development_manifest_sha256'),
                     (pp,'reserve_pilot_manifest_sha256')):
        if sha256(path)!=config[key]: raise RuntimeError('reserve dependency changed')
    roles=json.loads(rp.read_text(encoding='utf-8'))
    controls=json.loads(cp.read_text(encoding='utf-8'))
    pilot=json.loads(pp.read_text(encoding='utf-8'))
    rkey,ckey,pkey=keyed(roles['cases']),keyed(controls['records']),keyed(pilot['records'])
    if rkey.keys()!=ckey.keys() or not pkey.keys()<=rkey.keys(): raise RuntimeError('reserve cohorts inconsistent')
    reused,remaining=[],[]
    for case in roles['cases']:
        key=case['dataset'],case['seed']
        old=pkey.get(key)
        if old is None:
            remaining.append(case)
            continue
        baseline_equal(old,ckey[key])
        x,_=load_expanded(case['dataset'],roles)
        a,b=load_roles(old,len(x)),load_roles(case,len(x))
        for role in a: np.testing.assert_array_equal(a[role],b[role])
        reused.append(dict(old,reused=True))
    price=sum(tree_cost(r) for r in reused)
    if price!=config['reserve_expected_reused_trees']: raise RuntimeError('reserve pilot cost changed')
    path=output/'new_roles_manifest.json'
    write_report(path,dict(roles,cases=remaining,planned_cases=len(remaining)))
    new=output/'new'
    new.mkdir()
    cfg=dict(config,cumulative_roles_manifest=str(path),cumulative_roles_manifest_sha256=sha256(path))
    fresh={}
    absolute_history_training(cfg,fresh,new)
    by_key=keyed([*reused,*fresh['records']])
    records=[by_key[k] for k in rkey]
    write_report(output/'new_training_report.json',fresh)
    write_report(output/'cumulative_development_manifest.json',dict(records=records,tree_fits=fresh['tree_fits'],
        reused_tree_fits=price,reused_records_manifest=str(pp),reused_records_sha256=sha256(pp),
        source_roles_manifest=str(rp),source_roles_sha256=sha256(rp),
        source_controls_manifest=str(cp),source_controls_sha256=sha256(cp),
        baseline_training_cost=sum(tree_cost(r) for r in controls['records']),test_evaluated=False))
    report.update(status='development_complete',cases=len(records),new_tree_fits=fresh['tree_fits'],
        reused_tree_fits=price,test_evaluated=False,gpu_observations=fresh['gpu_observations'],
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('reserve_validation_development_complete',**{k:report[k] for k in
        ('cases','new_tree_fits','reused_tree_fits','development_manifest_sha256')})
