"""Longer training at unchanged serving bytes, with strong longer native controls."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles,fit_dense
from experiments.cumulative_budget_study import train_cumulative_path,choose_absolute_models
from experiments.cumulative_budget_projection import cumulative_proposal
from experiments.cumulative_budget_core import CumulativeError
from experiments.closed_budget_study import empty_model
from experiments.shared_leaf_basis import reference_class_model
from experiments.parallel_leaf_core import prefix
from experiments.inference_gpu_guard import SharedGpuObserver


def reinvest_budget_training(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_reinvest_budget.py','tests/test_horizon_reserve.py'],
                   cwd=ROOT,check=True,timeout=300)
    rp=Path(config['reinvest_source_roles_manifest'])
    cp=Path(config['reinvest_controls_manifest'])
    if sha256(rp)!=config['reinvest_source_roles_manifest_sha256'] or sha256(cp)!=config['reinvest_controls_manifest_sha256']:
        raise RuntimeError('frozen reinvestment sources changed')
    source=json.loads(rp.read_text(encoding='utf-8'))
    controls=json.loads(cp.read_text(encoding='utf-8'))
    wanted=set(config['reinvest_cases'])
    cases=[c for c in source['cases'] if f'{c["dataset"]}_{c["seed"]}' in wanted]
    if len(cases)!=len(wanted): raise RuntimeError('incomplete reinvestment cohort')
    rolepath=output/'roles_manifest.json'
    write_report(rolepath,dict(source,cases=cases,planned_cases=len(cases)))
    records,ledger=[],dict(trees=0)
    horizon=config['serving_tree_horizon']
    with threadpool_limits(limits=1),SharedGpuObserver() as observer:
        for case in cases:
            old=next(r for r in controls['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed'])
            if old['split_sha256']!=case['split_sha256']: raise RuntimeError('reinvestment control roles differ')
            if sha256(old['full']['artifact'])!=old['full']['sha256']: raise RuntimeError('reinvestment reference changed')
            x,y=load_expanded(case['dataset'],source)
            roles=load_roles(case,len(x))
            development={k:(x[roles[k]].copy(),y[roles[k]].copy()) for k in ('train','calibration','control')}
            selection=x[roles['selection']].copy(),y[roles['selection']].copy()
            directory=output/f'{case["dataset"]}_{case["seed"]}'
            directory.mkdir()
            snapshots,trajectory=train_cumulative_path('cumulative_reserve',case,development,config,directory/'compact400',ledger)
            trajectory['method']='cumulative400'
            decisions=[choose_absolute_models('cumulative400',[(f'block{b}',m) for b,m in snapshots],
                development['calibration'],selection,old['reference_validation'],case['absolute_budget_bytes'],config,directory)]
            full_fits=[]
            native,posthoc=[],[]
            fitting_y=np.zeros_like(y)
            for role in ('train','calibration'): fitting_y[roles[role]]=y[roles[role]]
            for depth in (3,4,5):
                if ledger['trees']+horizon>config['cumulative_pilot_tree_cap']:
                    raise RuntimeError('reinvestment tree cap exceeded before full fit')
                dense,fit=fit_dense(case,depth,config,x,fitting_y,roles,trees=horizon)
                ledger['trees']+=horizon
                path=directory/f'dense400-depth{depth}.npz'
                dense.save(path)
                fit.update(artifact=str(path),sha256=sha256(path))
                full_fits.append(fit)
                full=empty_model(reference_class_model(dense) if case['task']=='multiclass' else dense)
                native.extend((f'd{depth}-t{t}',prefix(full,t)) for t in range(40,horizon+1,40))
                error=CumulativeError(np.zeros((len(development['train'][0]),case['outputs'])),case['task'])
                model,info,_=cumulative_proposal(dense,development,case['absolute_budget_bytes'],error,config)
                write_report(directory/f'posthoc400-depth{depth}.json',info)
                if model is not None: posthoc.append((f'd{depth}',model))
                emit('reinvestment_full_control',dataset=case['dataset'],depth=depth,tree_fits=ledger['trees'])
            for name,models in (('native400',native),('posthoc400',posthoc)):
                decisions.append(choose_absolute_models(name,models,development['calibration'],selection,
                    old['reference_validation'],case['absolute_budget_bytes'],config,directory))
            record=dict(dataset=case['dataset'],seed=case['seed'],task=case['task'],outputs=case['outputs'],
                split_file=case['split_file'],split_sha256=case['split_sha256'],protocol_version=3,
                budget_bytes=case['absolute_budget_bytes'],reference_validation=old['reference_validation'],full=old['full'],
                trajectories=[trajectory],full_fits=full_fits,decisions=[*old['decisions'],*decisions],test_evaluated=False)
            records.append(record)
            write_report(directory/'case.json',record)
            write_report(output/'cumulative_development_manifest.json',dict(records=records,tree_fits=ledger['trees'],
                source_roles_manifest=str(rolepath),source_roles_sha256=sha256(rolepath),
                source_controls_manifest=str(cp),source_controls_sha256=sha256(cp),test_evaluated=False))
            emit('reinvestment_case_sealed',dataset=case['dataset'],seed=case['seed'],
                decisions=[dict(method=d['method'],state=d['state']) for d in decisions],test_evaluated=False)
    report.update(status='development_complete',tree_fits=ledger['trees'],test_evaluated=False,
        roles_manifest=str(rolepath),roles_manifest_sha256=sha256(rolepath),gpu_observations=observer.samples,
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('reinvestment_development_complete',trees=ledger['trees'],development_manifest_sha256=report['development_manifest_sha256'])
