"""Six new width trajectories, frozen ordinary controls, sealed budget choices."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess

import numpy as np

from budget_quality_worker import digest,locked,write,event,native_load
from abl_quality_core import (CASES,MODES,representation_plans,fit_representation_family,
    prepare_rank,build_representation,fingerprint,array_ledger,design_raw)
from abl_quality_worker import (pin,before_test_data,source_logits,source_descriptor,
    evaluate_candidate,save_logits,check_native_environment,memory_guard)
from abl_quality_evaluation import evaluate_study
from abl_stability_core import NumericalTolerance,select_stable
from calibration_pilot_core import CalibrationPlan,Fold,Scheme
from experiments.sparse_leaf_core import leaf_design


def single_crossfit(case):
    old=locked(case['plans_path'],case['plans_sha256'])
    index=next(i for i,p in enumerate(old['plans']) if p['scheme']=='crossfit' and p['repeat']==0)
    value=old['plans'][index]
    plan=CalibrationPlan(Scheme.CROSSFIT,0,1.,tuple(Fold(*(tuple(f[k]) for k in
        ('temperature','shifts','validation'))) for f in value['folds']),
        tuple(value['refit_temperature']),tuple(value['refit_shifts']))
    return index,plan,dict(old,plans=[value])


def development_case(cfg,old,output,locks,fit_ledger):
    case_id=old['case_id'];record=old['record'];data=before_test_data(record,locks)
    old_index,plan,plan_payload=single_crossfit(old)
    directory=output/'development'/case_id;directory.mkdir(parents=True,exist_ok=True)
    plan_path=directory/'plans.json';write(plan_path,plan_payload)
    metas=old['metas'];rows=[dict(r,plan=0) for r in old['rows'] if r['plan']==old_index]
    for meta in metas.values():
        pin(meta['source']['artifact'],meta['source']['sha256'],locks)
        pin(meta['development_cache']['path'],meta['development_cache']['sha256'],locks)
        meta['minimum_leaf']=10
    new_items=[];native_exe=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    worker=Path(__file__).resolve().parent/'abl_width_worker.py'
    for depth in (3,5):
        if fit_ledger['fits']>=6 or fit_ledger['rounds']+800>4800:
            raise RuntimeError('six-trajectory width cap exceeded before fitting')
        work=output/'width_training'/case_id/f'd{depth}';work.mkdir(parents=True,exist_ok=True)
        request=dict(record={k:record[k] for k in ('case','dataset')},depth=depth,minimum_leaf=50,rounds=800,
            training_config={k:cfg[k] for k in ('lr','lambda_l2','ntrees','own_pool_limit_bytes')})
        path=work/'request.json';write(path,request)
        # Reserve the complete attempt before fitting. An interrupted/failed
        # trajectory must not disappear from the authorized training counter.
        fit_ledger['fits']+=1;fit_ledger['rounds']+=800
        write(output/'training_ledger.json',fit_ledger)
        subprocess.run([native_exe,str(worker),str(path),str(work/'models')],check=True,timeout=7200)
        trajectory=json.loads((work/'models/trajectory.json').read_text())
        if trajectory['status']!='complete' or trajectory['fit']['actual_minimum_leaf']!=50:
            raise RuntimeError('native width setting was not applied')
        fit_ledger['records'].append(trajectory['fit']);new_items.extend(trajectory['models'])
        write(output/'training_ledger.json',fit_ledger)
        event('abl_width_fit_checkpoint',case=case_id,depth=depth,model_fits=fit_ledger['fits'],rounds=fit_ledger['rounds'])
    if len(new_items)!=12:raise RuntimeError('new width prefix grid incomplete')
    for item in new_items:
        uid='pyboost/'+item['identity'];z=source_logits(item,locks)
        model=native_load(item['artifact']);ledger=array_ledger(model)
        if ledger!=item['ledger']:raise RuntimeError('new native/external CPU array accounting mismatch')
        np.testing.assert_allclose(model.prepare_inference().raw(data['dev_x'][:17]),z[:17],atol=2e-4,rtol=2e-5)
        meta=dict(uid=uid,family='pyboost',identity=item['identity'],source=source_descriptor(item),
            rank=None,nnz=0,cost_bytes=ledger['cost_bytes'],ledger=ledger,minimum_leaf=50,
            development_cache=save_logits(directory/(item['identity']+'-logits.npz'),z))
        metas[uid]=meta;rows.extend(evaluate_candidate(uid,z,data['dev_y'],[plan],cfg))
        del model
    requested_count=0;physical=0
    for item in (m for m in new_items if m['rounds'] in (160,200,400,800)):
        dense=native_load(item['artifact']);m=len(dense.base.coefficients)
        train_design=leaf_design(dense.base.leaf_indices(data['train_x']),m)
        dev_design=leaf_design(dense.base.leaf_indices(data['dev_x']),m)
        state=fit_representation_family(dense,train_design,data['train_y'])
        plans=representation_plans(len(dense.base.bias),m,len(data['train_y']),dense.parameter_count)
        seen={}
        for rank in sorted({p.rank for p in plans}):
            pure,residual,order=prepare_rank(dense,train_design,state,rank)
            for representation in (p for p in plans if p.rank==rank):
                requested_count+=1;model=build_representation(pure,residual,order,state,representation)
                key=fingerprint(model)
                if key in seen:
                    metas[seen[key]]['requested_plans'].append(asdict(representation));continue
                identity=f'{item["identity"]}-r{rank}-p{int(round(representation.residual_fraction*100)):02d}'
                uid='compact/'+identity;seen[key]=uid
                path=output/'models'/case_id/(identity+'.npz');path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
                z=design_raw(model,dev_design);ledger=array_ledger(model)
                metas[uid]=dict(uid=uid,family='compact',identity=identity,parent_uid='pyboost/'+item['identity'],
                    parent_source=source_descriptor(item),source=dict(artifact=str(path),sha256=digest(path)),
                    rank=rank,nnz=len(model.values),cost_bytes=ledger['cost_bytes'],ledger=ledger,
                    parent_parameters=dense.parameter_count,requested_plans=[asdict(representation)],
                    array_fingerprint=key,minimum_leaf=50,
                    development_cache=save_logits(directory/(identity+'-logits.npz'),z))
                scores=evaluate_candidate(uid,z,data['dev_y'],[plan],cfg);rows.extend(scores)
                write(directory/(identity+'.json'),dict(meta=metas[uid],rows=scores))
                physical+=1;memory_guard()
                event('abl_width_representation_saved',case=case_id,uid=uid,cost_bytes=ledger['cost_bytes'],
                    requested_new=requested_count,physical_new=physical,test_accessed=False)
            del pure,residual,order
        del dense,train_design,dev_design,state;gc.collect()
    if requested_count!=160:raise RuntimeError('per-case fixed width compression grid changed')
    tolerance=NumericalTolerance(1e-6,1e-12);choices=[];external=[]
    for mode in MODES:
        for objective in ('loss','f1_macro'):
            for group in ('native','ab','abl','deployed'):
                for ratio in cfg['budget_ratios']:
                    budget=int(np.floor(old['reference_b0_bytes']*ratio))+4
                    chosen=select_stable(rows,metas,group,mode,objective,budget,tolerance)
                    choices.append(dict(plan=0,scheme='crossfit',repeat=0,mode=mode,objective=objective,
                        group=group,budget_ratio=ratio,budget_bytes=budget,chosen=chosen))
            for family in ('catboost','xgboost','tabpfn8'):
                previous=next(c for c in old['external_choices'] if c['plan']==old_index and
                    c['mode']==mode and c['objective']==objective and c['group']==family)
                external.append(dict(previous,plan=0))
    case=dict(case_id=case_id,record=record,reference_b0_bytes=old['reference_b0_bytes'],
        common_overhead_budget_bytes=4,numerical_tolerance=asdict(tolerance),
        plans_path=str(plan_path),plans_sha256=digest(plan_path),metas=metas,rows=rows,
        budget_choices=choices,external_choices=external,requested_representations=requested_count,
        physical_representations=physical,reused_q1_compact_models=160,test_accessed=False,
        calibration_protocol='one crossfit repeat0; same frozen roles, no internal-repeat uncertainty claim')
    path=directory/'case.json';write(path,case)
    return dict(case_id=case_id,path=str(path),sha256=digest(path),candidates=len(metas),
        requested_representations=requested_count,physical_representations=physical)


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'));output=args.output
    native_environment=check_native_environment(cfg)
    diagnostic=locked(cfg['abl_diagnostic_completed']['path'],cfg['abl_diagnostic_completed']['sha256'])
    if diagnostic['status']!='complete' or diagnostic['serving_audit']['status']!='passed':
        raise RuntimeError('successful numerical diagnostic and serving audit required')
    source=cfg['abl_study_completed'];old_report=locked(source['path'],source['sha256'])
    old_root=Path(source['path']).parent;old_seal=locked(old_root/'pretest_choices.json',old_report['pretest_seal_sha256'])
    locks=dict(old_seal['input_locks']);cases=[];ledger=dict(fits=0,rounds=0,records=[])
    for path,sha in locks.items():
        if digest(path)!=sha:raise RuntimeError('original source changed before width stage')
    for case_id in CASES:
        entry=next(c for c in old_seal['cases'] if c['case_id']==case_id)
        case=locked(entry['path'],entry['sha256']);pin(entry['path'],entry['sha256'],locks)
        cases.append(development_case(cfg,case,output,locks,ledger))
    if ledger['fits']!=6 or ledger['rounds']!=4800:raise RuntimeError('width training accounting mismatch')
    for path,sha in locks.items():
        if digest(path)!=sha:raise RuntimeError('width sources changed before seal')
    seal=output/'pretest_choices.json'
    write(seal,dict(status='abl_choices_sealed_before_test',cases=cases,input_locks=locks,
        test_used_for_selection=False,new_model_fits=6,new_boosting_rounds=4800))
    event('abl_width_whole_grid_sealed',cases=3,new_model_fits=6,new_boosting_rounds=4800,sha256=digest(seal))
    result=evaluate_study(cfg,output,seal)
    result.update(status='complete',cases=cases,native_environment=native_environment,source=source,
        new_model_fits=6,new_boosting_rounds=4800,training_ledger=ledger,
        new_requested_representations=sum(c['requested_representations'] for c in cases),
        reused_q1_compact_models=480,pretest_seal_sha256=digest(seal),test_used_for_selection=False,
        protocol='min_leaf10/50 and depth3/5, numeric1e-6/1e-12, model-array cap plus fixed4byte T, crossfit repeat0')
    for path,sha in locks.items():
        if digest(path)!=sha:raise RuntimeError('width source changed after evaluation')
    write(output/'report.json',result)
    event('abl_width_study_complete',path=str(output/'report.json'),sha256=digest(output/'report.json'),
        new_model_fits=6,new_boosting_rounds=4800)


if __name__=='__main__':main()
