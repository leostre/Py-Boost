"""Frozen M/r/s, targeted versus random support; whole-cohort seal before test."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.shared_leaf_basis import LeafEnsemble,functional_basis,compress,quality_guard
from experiments.cumulative_budget_core import absolute_allocations,serving_cost
from experiments.closed_budget_core import Topology
from experiments.sparse_leaf_study import base_at_rank
from experiments.sparse_leaf_core import (contrast,leaf_design,training_weights,coordinate_scores,
                                         score_order,make_sparse_model)
from experiments.uniform_threshold_core import ThresholdModel,threshold_variants,choice_key
from experiments.output_calibration_core import calibrated_quality
from experiments.history_basis_study import raw
from experiments.residual_address_core import random_support


def residual_address_study(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_residual_address_core.py'],cwd=ROOT,check=True,timeout=300)
    locks={}
    def locked(path,expected):
        if sha256(path)!=expected: raise RuntimeError('frozen residual input changed')
        locks[str(path)]=expected
        return Path(path)
    rp=locked(config['address_roles_manifest'],config['address_roles_manifest_sha256'])
    cp=locked(config['address_controls_manifest'],config['address_controls_manifest_sha256'])
    roles_manifest=json.loads(rp.read_text(encoding='utf-8'))
    controls=json.loads(cp.read_text(encoding='utf-8'))
    chosen=[]
    with threadpool_limits(limits=1):
        case_ids=([f'{c["dataset"]}_{c["seed"]}' for c in roles_manifest['cases']]
                  if config.get('address_all_cases') is True else config['address_cases'])
        if len(set(case_ids))!=len(case_ids): raise RuntimeError('duplicate address case')
        for case_id in case_ids:
            case=next(c for c in roles_manifest['cases'] if f'{c["dataset"]}_{c["seed"]}'==case_id)
            old=next(c for c in controls['records'] if c['dataset']==case['dataset'] and c['seed']==case['seed'])
            if case['split_sha256']!=old['split_sha256']: raise RuntimeError('different control roles')
            entry=roles_manifest['datasets'][case['dataset']]
            locked(entry['path'],entry['sha256'])
            locked(case['split_file'],case['split_sha256'])
            fit=next(f for f in old['full_fits'] if f['depth']==5)
            dense=LeafEnsemble.load(locked(fit['artifact'],fit['sha256']))
            locked(old['full']['artifact'],old['full']['sha256'])
            x,y=load_expanded(case['dataset'],roles_manifest)
            roles=load_roles(case,len(x))
            directory=output/case_id
            directory.mkdir()
            m,c=dense.coefficients.shape
            topology=Topology(dense.n_trees,len(dense.nodes),m)
            design=leaf_design(dense.leaf_indices(x[roles['train']]),m)
            contribution=design@dense.coefficients
            mean,basis,_=functional_basis(contribution,dense.task)
            allocations=absolute_allocations(topology,c,basis.shape[1],case['absolute_budget_bytes'])
            if not allocations:
                chosen.append(dict(case_id=case_id,state='no_rank_fits',models=[]))
                continue
            rank=min(a.rank for a in allocations)
            cap=max(a.nnz for a in allocations if a.rank==rank)
            full=compress(dense,mean,basis)
            base=base_at_rank(dense,mean,basis,rank,full)
            frequency=np.asarray(design.mean(axis=0)).ravel()
            residual=(contrast(dense.coefficients,dense.task)-contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
            teacher=contrast(contribution+dense.bias,dense.task)
            error=teacher-contrast((design@base.coefficients)@base.basis.T+base.bias,dense.task)
            weights=training_weights(y[roles['train']],teacher,dense.task,'balanced_mse',config['margin_scale'])
            targeted=score_order(coordinate_scores(design,residual,error,'balanced_mse',weights),residual)
            count=min(cap,len(targeted))
            orders=[('rank_only',targeted[:0]),('targeted',targeted[:count])]
            orders.extend((f'random_{s}',random_support(residual,count,s)) for s in config['address_random_seeds'])
            models=[]
            for family,order in orders:
                leaf=make_sparse_model(base,residual,order,len(order),frequency)
                model=ThresholdModel(leaf.base,leaf.rows,leaf.columns,leaf.values,np.zeros(c,np.float32))
                variants=threshold_variants(raw(model,x[roles['calibration']]),y[roles['calibration']],
                    raw(model,x[roles['selection']]),y[roles['selection']],dense.task,
                    config['decision_grid'],config['decision_shrinkage'])
                rows=[dict(candidate_id=family,parameters=model.parameter_count,payload_bytes=model.payload_bytes,**v) for v in variants]
                eligible=[r for r in rows if quality_guard(r['validation'],old['reference_validation'])]
                choice=min(eligible or rows,key=choice_key)
                model=replace(model,decision_shifts=np.array(choice['decision_shifts'],np.float32))
                cost=serving_cost(topology,c,rank,len(model.values))
                if (model.payload_bytes!=cost.coo_bytes or model.prepare_inference().resident_bytes!=cost.csr_bytes
                        or cost.maximum_bytes>case['absolute_budget_bytes'] or len(model.values)!=len(order)):
                    raise RuntimeError('matched residual cost contract failed')
                path=directory/(family+'.npz')
                model.save(path)
                models.append(dict(family=family,artifact=str(path),sha256=sha256(path),rank=rank,nnz=len(order),
                    leaves=m,trees=dense.n_trees,coo_bytes=cost.coo_bytes,csr_bytes=cost.csr_bytes,
                    selection_pass=bool(eligible),candidate=choice,candidates=rows))
            chosen.append(dict(case_id=case_id,case=case,full=old['full'],models=models,state='prepared'))
            emit('address_development_sealed',case_id=case_id,rank=rank,nnz=count,test_evaluated=False)
        seal=output/'selection_seal.json'
        write_report(seal,dict(records=chosen,input_locks=locks,new_trees=0,test_evaluated=False))
        seal_hash=sha256(seal)
        results=[]
        for record in chosen:
            if record['state']!='prepared': continue
            case=record['case']
            x,y=load_expanded(case['dataset'],roles_manifest)
            roles=load_roles(case,len(x))
            full=ThresholdModel.load(record['full']['artifact'])
            qref=calibrated_quality(y[roles['test']],raw(full,x[roles['test']]),case['task'],full.decision_shifts)
            for item in record['models']:
                model=ThresholdModel.load(locked(item['artifact'],item['sha256']))
                q=calibrated_quality(y[roles['test']],raw(model,x[roles['test']]),case['task'],model.decision_shifts)
                results.append(dict(case_id=record['case_id'],**item,quality=q,reference=qref,
                    test_pass=quality_guard(q,qref),joint_success=bool(item['selection_pass'] and quality_guard(q,qref))))
        if sha256(seal)!=seal_hash or any(sha256(p)!=h for p,h in locks.items()):
            raise RuntimeError('residual seal changed during evaluation')
    report.update(status='complete',results=results,development=chosen,new_trees=0,new_fits=0,
        selection_seal_sha256=seal_hash,test_is_reused_exploratory=True)
    emit('address_complete',cases=len(chosen),evaluations=len(results),new_trees=0)
