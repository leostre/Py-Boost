"""Expanded V3 roles and separate history pilot; never use test for selection."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments.a100_runner import ROOT,emit,write_report,sha256,make_partitions
from experiments.leaf_budget_runtime import expanded_manifest,load_expanded,load_roles
from experiments.cumulative_budget_core import widen_roles
from experiments.expanded_budget_data import control_selection_partition,validate_roles
from experiments.tiny_residual_study import calibration_partition


def prepare_absolute_roles(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_absolute_history_core.py',
        'tests/test_cumulative_budget_projection.py','tests/test_cumulative_budget_core.py'],cwd=ROOT,check=True,timeout=600)
    source=expanded_manifest(config)
    registered=json.loads((ROOT/'experiments/a100_data_manifest.json').read_text(encoding='utf-8'))
    datasets=dict(source['datasets'])
    cases,unsupported=[],[]
    for name,seeds in config['absolute_seeds'].items():
        if name not in datasets:
            entry=registered[name]
            datasets[name]=dict(entry,path=str(Path(config['data_dir'])/(name+'.npz')))
        x,y=load_expanded(name,dict(datasets=datasets))
        entry=datasets[name]
        task=entry['task']
        c=entry['target_shape'][1] if task=='multilabel' else int(y.max())+1
        for seed in seeds:
            directory=output/f'{name}_{seed}'
            directory.mkdir()
            old=next((p for p in source['cases'] if p['dataset']==name and p['seed']==seed),None)
            if old is None:
                train,validation,test=make_partitions(y,task,seed)
                a,b=calibration_partition(y[validation],task,c,seed)
                cal,remaining=validation[a],validation[b]
                control,selection=control_selection_partition(y,remaining,task,c,seed)
                original=dict(train=train,calibration=cal,control=control,selection=selection,test=test)
                validate_roles(original,len(x))
                path=directory/'source_v2_splits.npz'
                np.savez(path,**original)
                old=dict(dataset=name,seed=seed,task=task,outputs=c,split_file=str(path),split_sha256=sha256(path))
            else: original=load_roles(old,len(x))
            roles=widen_roles(y[original['train']],original,c,task,seed)
            validate_roles(roles,len(x))
            for role in ('calibration','test'): np.testing.assert_array_equal(roles[role],original[role])
            if task=='multiclass' and set(map(int,y[roles['train']]))!=set(range(c)):
                unsupported.append(dict(dataset=name,seed=seed,status='missing_training_classes',
                    classes_present=np.unique(y[roles['train']]).tolist(),expected=c))
                continue
            path=directory/'splits.npz'
            np.savez(path,**roles)
            labels=(np.eye(c)[y[roles['train']].astype(int)] if task=='multiclass' else y[roles['train']])
            case=dict(old,split_file=str(path),split_sha256=sha256(path),
                source_split_file=old['split_file'],source_split_sha256=old['split_sha256'],
                protocol_version=3,absolute_budget_bytes=131072 if c<=32 else 1048576,
                sizes={k:len(v) for k,v in roles.items()},existing_case_preserved=bool(
                    any(p['dataset']==name and p['seed']==seed for p in source['cases'])),
                train_meta=dict(rows=len(roles['train']),features=x.shape[1],outputs=c,
                    support=labels.sum(axis=0).astype(int).tolist(),label_density=float(labels.mean()),
                    feature_zero_fraction=float(np.mean(x[roles['train']]==0))),test_labels_used=False)
            cases.append(case)
            emit('absolute_expanded_roles',dataset=name,seed=seed,sha256=case['split_sha256'],sizes=case['sizes'])
    manifest=dict(version=3,datasets=datasets,cases=cases,unsupported_cases=unsupported,
        source_manifest=config['expanded_manifest'],source_sha256=config['expanded_manifest_sha256'],
        planned_cases=sum(len(s) for s in config['absolute_seeds'].values()),test_labels_used=False,
        qualification='fixed public subsets, reused objects; new seeds are not independent new datasets')
    write_report(output/'absolute_roles_manifest.json',manifest)
    report.update(status='complete',cases=len(cases),unsupported_cases=unsupported,new_fits=0,new_trees=0,
        test_evaluated=False,manifest_sha256=sha256(output/'absolute_roles_manifest.json'))
    emit('absolute_expanded_roles_complete',**{k:report[k] for k in ('cases','unsupported_cases','manifest_sha256')})


def absolute_history_training(config,report,output):
    import gc
    from threadpoolctl import threadpool_limits
    from experiments.cumulative_budget_study import train_cumulative_path,choose_absolute_models
    from experiments.inference_gpu_guard import SharedGpuObserver
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_absolute_history_core.py',
                    'tests/test_absolute_history_shell.py'],cwd=ROOT,check=True,timeout=600)
    rp=Path(config['cumulative_roles_manifest'])
    pp=Path(config['cumulative_development_manifest'])
    if sha256(rp)!=config['cumulative_roles_manifest_sha256'] or sha256(pp)!=config['cumulative_development_manifest_sha256']:
        raise RuntimeError('frozen V3 pilot input changed')
    roles_manifest=json.loads(rp.read_text(encoding='utf-8'))
    previous=json.loads(pp.read_text(encoding='utf-8'))
    records,ledger=[],dict(trees=0)
    with threadpool_limits(limits=1),SharedGpuObserver() as observer:
        for case in roles_manifest['cases']:
            old=next(r for r in previous['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed'])
            if old['split_sha256']!=case['split_sha256']:
                raise RuntimeError('pilot controls used different roles')
            for row in [old['full'],*old['decisions']]:
                if row.get('artifact') and sha256(row['artifact'])!=row['sha256']:
                    raise RuntimeError('frozen control model changed')
            x,y=load_expanded(case['dataset'],roles_manifest)
            roles=load_roles(case,len(x))
            development={k:(x[roles[k]].copy(),y[roles[k]].copy()) for k in ('train','calibration','control')}
            selection=(x[roles['selection']].copy(),y[roles['selection']].copy())
            directory=output/f'{case["dataset"]}_{case["seed"]}'
            directory.mkdir()
            trajectories,decisions=[],[]
            for method in config['cumulative_methods']:
                snapshots,trajectory=train_cumulative_path(method,case,development,config,directory/method,ledger)
                trajectories.append(trajectory)
                decisions.append(choose_absolute_models(method,[(f'block{b}',m) for b,m in snapshots],
                    development['calibration'],selection,old['reference_validation'],case['absolute_budget_bytes'],
                    config,directory))
            record=dict(dataset=case['dataset'],seed=case['seed'],task=case['task'],outputs=case['outputs'],
                split_file=case['split_file'],split_sha256=case['split_sha256'],protocol_version=3,
                budget_bytes=case['absolute_budget_bytes'],reference_validation=old['reference_validation'],
                full=old['full'],trajectories=trajectories,full_fits=[],decisions=[*old['decisions'],*decisions],
                reused_parent_tree_fits=sum(t['trees_built'] for t in old['trajectories'])+
                    sum(f['trees'] for f in old['full_fits']),test_evaluated=False)
            records.append(record)
            write_report(directory/'case.json',record)
            write_report(output/'cumulative_development_manifest.json',dict(records=records,tree_fits=ledger['trees'],
                source_roles_manifest=str(rp),source_roles_sha256=sha256(rp),source_controls_manifest=str(pp),
                source_controls_sha256=sha256(pp),test_evaluated=False))
            emit('absolute_history_case_sealed',dataset=case['dataset'],seed=case['seed'],
                 trees_built=ledger['trees'],decisions=[dict(method=d['method'],state=d['state']) for d in decisions],
                 test_evaluated=False)
            del development,selection,x,y
            gc.collect()
    report.update(status='development_complete',records=records,tree_fits=ledger['trees'],
        test_evaluated=False,gpu_observations=observer.samples,
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('absolute_history_development_complete',cases=len(records),trees=ledger['trees'],
         development_manifest_sha256=report['development_manifest_sha256'])
