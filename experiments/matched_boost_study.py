"""Strong native controls on exactly the external suite's objects and roles."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles,fit_dense
from experiments.shared_leaf_basis import LeafEnsemble,reference_class_model
from experiments.closed_budget_study import empty_model
from experiments.parallel_leaf_core import prefix
from experiments.history_basis_study import raw
from experiments.uniform_threshold_core import threshold_variants,choice_key,ThresholdModel


def reuse_fit(case,controls,depth):
    old=next((r for r in controls['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed']),None)
    if old is None: return None
    if old['split_sha256']!=case['split_sha256']: raise RuntimeError('native control reused different roles')
    fit=next(f for f in old['full_fits'] if f['depth']==depth)
    if sha256(fit['artifact'])!=fit['sha256']: raise RuntimeError('native reused dense artifact changed')
    return fit


def matched_boost_training(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_matched_boost_study.py',
        'tests/test_leaf_budget_runtime.py','tests/test_continuation_contract.py'],cwd=ROOT,check=True,timeout=300)
    rp=Path(config['matched_manifest'])
    cp=Path(config['matched_controls_manifest'])
    if sha256(rp)!=config['matched_manifest_sha256'] or sha256(cp)!=config['matched_controls_manifest_sha256']:
        raise RuntimeError('matched native dependencies changed')
    roles_manifest=json.loads(rp.read_text(encoding='utf-8'))
    controls=json.loads(cp.read_text(encoding='utf-8'))
    records,new_trees,reused_trees=[],0,0
    with threadpool_limits(limits=1):
        for case_id in config['matched_cases']:
            case=next(c for c in roles_manifest['cases'] if f'{c["dataset"]}_{c["seed"]}'==case_id)
            x,y=load_expanded(case['dataset'],roles_manifest)
            roles=load_roles(case,len(x))
            fitting_y=np.zeros_like(y)
            for key in ('train','calibration'): fitting_y[roles[key]]=y[roles[key]]
            for family in config['matched_methods']:
                directory=output/case_id/family
                directory.mkdir(parents=True)
                rows,artifacts,raw_files,fits=[],{}, {},[]
                for depth in (3,4,5):
                    old=reuse_fit(case,controls,depth) if family=='full' else None
                    if old is None:
                        if new_trees+200>config['matched_tree_cap']:
                            raise RuntimeError('matched tree cap exceeded before fit')
                        dense,fit=fit_dense(case,depth,config,x,fitting_y,roles,trees=200,method=family)
                        new_trees+=200
                        path=directory/f'dense-depth{depth}.npz'
                        dense.save(path)
                        fit.update(artifact=str(path),sha256=sha256(path),reused=False)
                    else:
                        dense=LeafEnsemble.load(old['artifact'])
                        fit=dict(old,reused=True)
                        reused_trees+=fit['trees']
                    fits.append(fit)
                    base=empty_model(reference_class_model(dense))
                    if len(base.base.bias)!=case['outputs']: raise RuntimeError('native output class domain mismatch')
                    for trees in (40,80,120,160,200):
                        identity=f'd{depth}-t{trees}'
                        model=prefix(base,trees)
                        a,b=raw(model,x[roles['calibration']]),raw(model,x[roles['selection']])
                        variants=threshold_variants(a,y[roles['calibration']],b,y[roles['selection']],case['task'],
                            config['decision_grid'],config['decision_shrinkage'])
                        rows.extend(dict(candidate_id=identity,parameters=model.parameter_count,payload_bytes=model.payload_bytes,**v)
                                    for v in variants)
                        path=directory/(identity+'.npz')
                        model.save(path)
                        rawpath=directory/(identity+'-development.npz')
                        np.savez_compressed(rawpath,calibration=a,selection=b)
                        artifacts[identity]=dict(artifact=str(path),sha256=sha256(path),depth=depth,rounds=trees,
                            coo_bytes=model.payload_bytes,csr_bytes=model.prepare_inference().resident_bytes,
                            native_file_bytes=path.stat().st_size,trees=model.base.n_trees,leaves=len(model.base.coefficients),
                            leaf_parameters=model.base.coefficients.size,parameters=model.parameter_count)
                        raw_files[identity]=dict(artifact=str(rawpath),sha256=sha256(rawpath))
                chosen=min(rows,key=choice_key)
                selected=ThresholdModel.load(artifacts[chosen['candidate_id']]['artifact'])
                selected=replace(selected,decision_shifts=np.array(chosen['decision_shifts'],np.float32))
                p=directory/'selected.npz'
                selected.save(p)
                result=dict(case_id=case_id,case=case,dataset=roles_manifest['datasets'][case['dataset']],
                    family=family,chosen=chosen,candidates=rows,artifacts=artifacts,development_predictions=raw_files,
                    selected_artifact=str(p),selected_sha256=sha256(p),fits=fits,test_evaluated=False)
                records.append(result)
                write_report(directory/'record.json',result)
                write_report(output/'development_manifest.json',dict(records=records,new_tree_fits=new_trees,
                    reused_tree_fits=reused_trees,test_evaluated=False,source_roles_manifest=str(rp),source_roles_sha256=sha256(rp),
                    source_controls_manifest=str(cp),source_controls_sha256=sha256(cp)))
                emit('matched_boost_family_sealed',case_id=case_id,family=family,new_trees=new_trees,
                     validation=chosen['validation'],test_evaluated=False)
    report.update(status='development_complete',new_tree_fits=new_trees,reused_tree_fits=reused_trees,
        cases=len(config['matched_cases']),records=len(records),test_evaluated=False,
        development_manifest_sha256=sha256(output/'development_manifest.json'))
    emit('matched_boost_development_complete',**{k:report[k] for k in
        ('new_tree_fits','reused_tree_fits','cases','records','test_evaluated','development_manifest_sha256')})
