"""Freeze/reuse audited pilot; train only the other expanded V3 cases."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments.a100_runner import ROOT,write_report,sha256,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.cumulative_budget_study import cumulative_training


def absolute_validation_training(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_absolute_validation_shell.py'],
                   cwd=ROOT,check=True,timeout=600)
    roles_path=Path(config['absolute_roles_manifest'])
    parent_path=Path(config['cumulative_development_manifest'])
    if (sha256(roles_path)!=config['absolute_roles_manifest_sha256']
            or sha256(parent_path)!=config['cumulative_development_manifest_sha256']):
        raise RuntimeError('frozen expanded roles or pilot changed')
    source=json.loads(roles_path.read_text(encoding='utf-8'))
    parent=json.loads(parent_path.read_text(encoding='utf-8'))
    reused,remaining=[],[]
    reused_trees=0
    for case in source['cases']:
        old=next((r for r in parent['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed']),None)
        if old is None:
            remaining.append(case)
            continue
        x,_=load_expanded(case['dataset'],source)
        expected=load_roles(old,len(x))
        actual=load_roles(case,len(x))
        for role in expected:
            if not np.array_equal(expected[role],actual[role]):
                raise RuntimeError('cannot reuse pilot with changed development roles')
        if old['budget_bytes']!=case['absolute_budget_bytes']:
            raise RuntimeError('reused pilot budget changed')
        for row in [old['full'],*old['decisions']]:
            if row.get('artifact') and sha256(row['artifact'])!=row['sha256']:
                raise RuntimeError('reused pilot model changed')
        reused.append(dict(old,reused=True))
        reused_trees+=sum(t['trees_built'] for t in old['trajectories'])+sum(f['trees'] for f in old['full_fits'])
        emit('absolute_case_reused',dataset=case['dataset'],seed=case['seed'],roles_equal=True,new_trees=0)
    if reused_trees!=config['expected_reused_tree_fits']:
        raise RuntimeError('unexpected original training cost')
    path=output/'new_roles_manifest.json'
    write_report(path,dict(source,cases=remaining,planned_cases=len(remaining)))
    new_output=output/'new'
    new_output.mkdir()
    cfg=dict(config,cumulative_roles_manifest=str(path),cumulative_roles_manifest_sha256=sha256(path))
    new_report={}
    if remaining: cumulative_training(cfg,new_report,new_output)
    else: new_report=dict(records=[],tree_fits=0,gpu_observations=[])
    write_report(output/'new_training_report.json',new_report)
    by_id={(r['dataset'],r['seed']):r for r in [*reused,*new_report['records']]}
    records=[by_id[(c['dataset'],c['seed'])] for c in source['cases']]
    manifest=dict(records=records,tree_fits=new_report['tree_fits'],reused_tree_fits=reused_trees,
        source_roles_manifest=str(roles_path),source_roles_sha256=sha256(roles_path),
        reused_records_manifest=str(parent_path),reused_records_sha256=sha256(parent_path),
        test_evaluated=False)
    write_report(output/'cumulative_development_manifest.json',manifest)
    report.update(status='development_complete',cases=len(records),new_tree_fits=new_report['tree_fits'],
        reused_tree_fits=reused_trees,total_tree_fits_including_reused=new_report['tree_fits']+reused_trees,
        test_evaluated=False,gpu_observations=new_report['gpu_observations'],
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('absolute_validation_development_complete',**{k:report[k] for k in
        ('cases','new_tree_fits','reused_tree_fits','development_manifest_sha256')})
