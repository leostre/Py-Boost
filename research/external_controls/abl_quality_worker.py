"""H-ABL-Q0/Q1: frozen-model compression, sealed selection, CPU evaluation.

This worker has no tree-training or weight-download entry point. Source features
and targets share one NPZ: only train/dev target slices leave the pre-seal loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from budget_quality_worker import digest,write,locked,event,native_load
from budget_quality_core import quality,paired_bootstrap
from calibration_pilot_core import (Scheme,make_plans,evaluate_plan,fit_controls,
    choose_policy,control_for,POLICIES,POLICY_BY_NAME)
from abl_quality_core import (CASES,RANK_FRACTIONS,RESIDUAL_FRACTIONS,MODES,GROUPS,
    representation_plans,source_models,fit_representation_family,prepare_rank,
    build_representation,fingerprint,array_ledger,design_raw,choose_budget,fitted_control)
from experiments.sparse_leaf_core import leaf_design,contrast
from experiments.shared_leaf_basis import quality_guard

ROLES=('calibration','control','selection')


def memory_guard():
    """Observe owned CPU work; do not mistake process RSS for model storage."""
    if sys.platform=='win32':return None
    import resource
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    if peak>32*1024**3:raise RuntimeError('H-ABL worker exceeded its 32 GiB CPU working-memory limit')
    return peak


def check_contract(cfg):
    block=cfg['abl_quality']
    fixed=dict(cases=list(CASES),depths=[3,5],source_rounds=[160,200,400,800],
        control_rounds=[40,80,160,200,400,800],rank_fractions=list(RANK_FRACTIONS),
        residual_fractions=list(RESIDUAL_FRACTIONS),selector='balanced_mse',
        repeats=10,schemes=['pooled','crossfit'],fraction=1.,folds=3,
        modes=list(MODES),tree_training_allowed=False,gpu_execution=False,
        maximum_compression_sources=24,maximum_representations=480,
        bootstrap_repeats=500,bootstrap_repeat_indices=[0],
        bootstrap_modes=['raw','temperature_shifts'])
    for key,expected in fixed.items():
        if block.get(key)!=expected:raise RuntimeError('outside approved H-ABL-Q1 contract: '+key)
    if cfg['budget_ratios']!=[.25,.5,.75,1,1.5,2,4]:raise RuntimeError('budget grid changed')
    return block


def check_native_environment(cfg):
    old=locked(cfg['frozen_hasboost_environment'],cfg['frozen_hasboost_sha256'])
    script=("import importlib.metadata as m,json,platform,subprocess,sys; "
        "print(json.dumps(dict(python=platform.python_version(),executable=sys.executable,"
        "system=platform.system(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),"
        "packages=sorted((d.metadata['Name'].lower(),d.version) for d in m.distributions()))))")
    actual=json.loads(subprocess.check_output([old['executable'],'-c',script],text=True,timeout=120))
    if actual!=old:raise RuntimeError('actual native package/Python/GPU contract changed')
    return dict(sha256=cfg['frozen_hasboost_sha256'],packages=len(actual['packages']),actual_runtime_checked=True)


def pin(path,expected,locks):
    path=str(path)
    if path not in locks:
        if digest(path)!=expected:raise RuntimeError('immutable source changed: '+path)
        locks[path]=expected


def source_input(cfg,locks):
    item=cfg['calibration_pilot']['source_development']
    pin(item['path'],item['sha256'],locks)
    return json.loads(Path(item['path']).read_text(encoding='utf-8'))


def before_test_data(record,locks):
    case,entry=record['case'],record['dataset']
    pin(case['split_file'],case['split_sha256'],locks)
    pin(entry['path'],entry['sha256'],locks)
    with np.load(case['split_file'],allow_pickle=False) as f:
        ids={r:f[r].copy() for r in ('train',*ROLES,'test')}
    all_ids=np.concatenate(list(ids.values()))
    if len(np.unique(all_ids))!=len(all_ids) or not np.array_equal(np.sort(all_ids),np.arange(len(all_ids))):
        raise RuntimeError('roles do not form an exact disjoint cohort')
    development=np.concatenate([ids[r] for r in ROLES])
    with np.load(entry['path'],allow_pickle=False) as f:
        xtrain=f['features'][ids['train']].astype(np.float32)
        xdev=f['features'][development].astype(np.float32)
        ytrain=f['target'][ids['train']].astype(np.int64)
        ydev=f['target'][development].astype(np.int64)
    if not np.array_equal(np.unique(ytrain),np.arange(case['outputs'])):
        raise RuntimeError('training output domain incomplete')
    return dict(train_x=xtrain,train_y=ytrain,dev_x=xdev,dev_y=ydev,ids=ids,
        dev_ids=development,sizes=[len(ids[r]) for r in ROLES])


def source_logits(item,locks):
    pin(item['artifact'],item['sha256'],locks)
    pin(item['development_file'],item['development_sha256'],locks)
    with np.load(item['development_file'],allow_pickle=False) as f:
        return np.concatenate([f[r] for r in ROLES]).astype(np.float64)


def source_descriptor(item):
    return {k:item[k] for k in ('identity','artifact','sha256','development_file',
        'development_sha256','depth','rounds','origin')}


def preflight(cfg,output):
    block=check_contract(cfg);started=perf_counter();locks={}
    native_environment=check_native_environment(cfg)
    original=source_input(cfg,locks);checks=[];control_ledgers={};sources=[]
    for case_id in CASES:
        records=[r for r in original['records'] if r['case_id']==case_id]
        native,models=source_models(records,case_id,'pyboost')
        data=before_test_data(native,locks)
        for family in ('pyboost','catboost','xgboost','tabpfn8'):
            record,items=source_models(records,case_id,family)
            if record['case']['split_sha256']!=native['case']['split_sha256'] or record['dataset']['sha256']!=native['dataset']['sha256']:
                raise RuntimeError('unequal data or roles between external families')
            for item in items:
                z=source_logits(item,locks)
                if z.shape!=(len(data['dev_y']),native['case']['outputs']) or not np.isfinite(z).all():
                    raise RuntimeError('source logits domain/shape mismatch')
        compressed=source_models(records,case_id,'pyboost',True)[1]
        compressed_ids={m['identity'] for m in compressed}
        for item in models:
            model=native_load(item['artifact']);prepared=model.prepare_inference()
            ledger=array_ledger(model,prepared)
            if ledger['prepared_bytes']!=item['ledger']['deployed_array_bytes']:
                raise RuntimeError('historical array ledger mismatch')
            uid=case_id+'/pyboost/'+item['identity'];control_ledgers[uid]=ledger
            # Full development replay, not only the displayed selected test point.
            expected=source_logits(item,locks)
            replay=prepared.raw(data['dev_x'])
            np.testing.assert_allclose(replay,expected,atol=2e-4,rtol=2e-5)
            check=dict(case=case_id,source=item['identity'],replay_max_error=float(np.max(np.abs(replay-expected))),
                full_rank_checked=False,cost_bytes=ledger['cost_bytes'])
            if item['identity'] in compressed_ids:
                train_design=leaf_design(model.base.leaf_indices(data['train_x']),len(model.base.coefficients))
                fitted=fit_representation_family(model,train_design,data['train_y'])
                rank=min(native['case']['outputs']-1,len(model.base.coefficients),len(data['train_y']))
                full,_,_=prepare_rank(model,train_design,fitted,rank)
                dev_design=leaf_design(model.base.leaf_indices(data['dev_x']),len(model.base.coefficients))
                raw=design_raw(full,dev_design)
                np.testing.assert_allclose(contrast(raw,'multiclass'),contrast(expected,'multiclass'),atol=2e-4,rtol=2e-5)
                from budget_quality_core import probabilities
                np.testing.assert_allclose(probabilities(raw),probabilities(expected),atol=2e-5,rtol=2e-5)
                target=output/'full_rank_checks'/case_id/(item['identity']+'.npz')
                target.parent.mkdir(parents=True,exist_ok=True);full.save(target)
                restored=native_load(target)
                if fingerprint(restored)!=fingerprint(full):raise RuntimeError('full rank round trip mismatch')
                check.update(full_rank_checked=True,rank=rank,
                    full_rank_probability_error=float(np.max(np.abs(probabilities(raw)-probabilities(expected)))))
                sources.append(dict(case=case_id,identity=item['identity'],sha256=item['sha256']))
                del train_design,dev_design,fitted,full,restored,raw
            checks.append(check)
            event('abl_preflight_source_checked',**check,new_tree_fits=0,test_targets_used=False)
            del model,prepared,replay;gc.collect()
    if len(sources)!=24:raise RuntimeError('must validate exactly 24 unique new sources')
    # Binary data remain unchanged negative controls; no artificial rank saving.
    anchors=[]
    for case_id in ('wdbc_4104','banknote_4204'):
        r,items=source_models(original['records'],case_id,'pyboost')
        data=before_test_data(r,locks)
        item=next(m for m in items if m['identity']=='d5-t200')
        z=source_logits(item,locks)
        if z.shape[1]!=2:raise RuntimeError('binary canonical output domain changed')
        anchors.append(dict(case=case_id,classes=2,maximum_contrast_rank=1,
            development_rows=len(data['dev_y']),full_rank_grid_not_expanded=True))
    for p,s in locks.items():
        if digest(p)!=s:raise RuntimeError('source changed during preflight')
    result=dict(status='passed',configuration=block,source_manifest=cfg['calibration_pilot']['source_development'],
        native_environment=native_environment,checks=checks,control_ledgers=control_ledgers,
        compression_sources=sources,binary_anchors=anchors,input_locks=locks,
        new_tree_fits=0,new_boosting_rounds=0,gpu_execution=False,test_targets_used=False,
        seconds=perf_counter()-started,peak_cpu_rss_bytes=memory_guard())
    write(output/'report.json',result)
    event('abl_preflight_complete',status='passed',sources=len(sources),dense_replays=len(checks),
        input_files=len(locks),report_path=str(output/'report.json'),report_sha256=digest(output/'report.json'),
        seconds=result['seconds'],new_tree_fits=0,gpu_execution=False)


def plans_for(data,cfg):
    return [p for repeat in range(10) for p in make_plans(data['dev_y'],data['sizes'],repeat,(1.,),
        cfg['calibration_pilot']['seed']) if p.scheme in (Scheme.POOLED,Scheme.CROSSFIT)]


def evaluate_candidate(uid,logits,y,plans,cfg):
    rows=[];refits={}
    for index,plan in enumerate(plans):
        scores,diagnostics=evaluate_plan(logits,y,plan,cfg['decision_grid'])
        refit_key=(plan.refit_temperature,plan.refit_shifts)
        if refit_key not in refits:
            refits[refit_key]=fit_controls(logits,y,*refit_key,cfg['decision_grid'])
        t,raw_shift,temp_shift=refits[refit_key]
        rows.append(dict(uid=uid,plan=index,scheme=plan.scheme.value,repeat=plan.repeat,
            validation=scores,fold_temperatures=[d['temperature'] for d in diagnostics],
            refit=dict(temperature=t,raw_shifts=raw_shift.tolist(),temperature_shifts=temp_shift.tolist())))
    return rows


def save_logits(path,logits):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,raw=np.asarray(logits,np.float64))
    return dict(path=str(path),sha256=digest(path))


def case_development(cfg,records,case_id,prechecked,output,locks):
    started=perf_counter();native,controls=source_models(records,case_id,'pyboost')
    data=before_test_data(native,locks);plans=plans_for(data,cfg)
    directory=output/'development'/case_id;directory.mkdir(parents=True,exist_ok=True)
    plan_path=directory/'plans.json'
    write(plan_path,dict(case=case_id,plans=[dict(asdict(p),scheme=p.scheme.value) for p in plans],
        train_ids=data['ids']['train'].tolist(),dev_ids=data['dev_ids'].tolist(),
        test_ids=data['ids']['test'].tolist(),target=data['dev_y'].tolist(),sizes=data['sizes']))
    b0=next(m['ledger']['deployed_array_bytes'] for m in native['models'] if m['identity']=='archival-d5-t200')
    metas={};rows=[]
    for family in ('pyboost','catboost','xgboost','tabpfn8'):
        record,items=source_models(records,case_id,family)
        for item in items:
            uid=family+'/'+item['identity'];z=source_logits(item,locks)
            ledger=(prechecked['control_ledgers'][case_id+'/'+uid] if family=='pyboost' else None)
            metas[uid]=dict(uid=uid,family=family,identity=item['identity'],source=source_descriptor(item),
                rank=None,nnz=0,cost_bytes=ledger['cost_bytes'] if ledger else None,
                ledger=ledger,development_cache=save_logits(directory/(uid.replace('/','-')+'-logits.npz'),z))
            score_rows=evaluate_candidate(uid,z,data['dev_y'],plans,cfg)
            rows.extend(score_rows)
            write(directory/(uid.replace('/','-')+'.json'),dict(meta=metas[uid],rows=score_rows))
            event('abl_control_saved',case=case_id,uid=uid,scenarios=len(score_rows),new_tree_fits=0,test_accessed=False)
    physical=0;requests=0
    for item in source_models(records,case_id,'pyboost',True)[1]:
        dense=native_load(item['artifact'])
        train_design=leaf_design(dense.base.leaf_indices(data['train_x']),len(dense.base.coefficients))
        dev_design=leaf_design(dense.base.leaf_indices(data['dev_x']),len(dense.base.coefficients))
        fitted=fit_representation_family(dense,train_design,data['train_y'])
        requested=representation_plans(len(dense.base.bias),len(dense.base.coefficients),len(data['train_y']),dense.parameter_count)
        seen={}
        for rank in sorted({p.rank for p in requested}):
            pure,residual,order=prepare_rank(dense,train_design,fitted,rank)
            for plan in (p for p in requested if p.rank==rank):
                requests+=1;model=build_representation(pure,residual,order,fitted,plan)
                key=fingerprint(model)
                if key in seen:
                    metas[seen[key]]['requested_plans'].append(asdict(plan))
                    continue
                identity=f'{item["identity"]}-r{rank}-p{int(round(plan.residual_fraction*100)):02d}'
                uid='compact/'+identity;seen[key]=uid
                path=output/'models'/case_id/(identity+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                model.save(path)
                z=design_raw(model,dev_design);ledger=array_ledger(model)
                meta=dict(uid=uid,family='compact',identity=identity,parent_uid='pyboost/'+item['identity'],
                    parent_source=source_descriptor(item),source=dict(artifact=str(path),sha256=digest(path)),
                    rank=rank,nnz=len(model.values),cost_bytes=ledger['cost_bytes'],ledger=ledger,
                    parent_parameters=dense.parameter_count,requested_plans=[asdict(plan)],array_fingerprint=key,
                    development_cache=save_logits(directory/(identity+'-logits.npz'),z))
                metas[uid]=meta
                evaluated=evaluate_candidate(uid,z,data['dev_y'],plans,cfg);rows.extend(evaluated)
                write(directory/(identity+'.json'),dict(meta=meta,rows=evaluated))
                physical+=1
                memory_guard()
                event('abl_representation_saved',case=case_id,parent=item['identity'],rank=rank,
                    residual_cap=plan.residual_fraction,nnz=len(model.values),cost_bytes=ledger['cost_bytes'],
                    unique_case_representations=physical,requested_case_representations=requests,
                    seconds=perf_counter()-started,new_tree_fits=0,test_accessed=False)
            del pure,residual,order
        del dense,train_design,dev_design,fitted;gc.collect()
    if requests>160 or physical>160:raise RuntimeError('approved per-case representation limit exceeded')
    choices=[];external=[]
    for index,plan in enumerate(plans):
        pool=[r for r in rows if r['plan']==index]
        for mode in MODES:
            for objective in ('loss','f1_macro'):
                for group in GROUPS:
                    for ratio in cfg['budget_ratios']:
                        budget=int(np.floor(b0*ratio))
                        chosen=choose_budget(pool,metas,group,mode,objective,budget)
                        choices.append(dict(plan=index,scheme=plan.scheme.value,repeat=plan.repeat,
                            mode=mode,objective=objective,group=group,budget_ratio=ratio,budget_bytes=budget,chosen=chosen))
                for family in ('catboost','xgboost','tabpfn8'):
                    chosen=choose_budget(pool,metas,family,mode,objective)
                    external.append(dict(plan=index,scheme=plan.scheme.value,repeat=plan.repeat,
                        mode=mode,objective=objective,group=family,chosen=chosen))
    case=dict(case_id=case_id,record=native,reference_b0_bytes=b0,plans_path=str(plan_path),
        plans_sha256=digest(plan_path),metas=metas,rows=rows,budget_choices=choices,external_choices=external,
        physical_representations=physical,requested_representations=requests,test_accessed=False)
    path=directory/'case.json';write(path,case)
    event('abl_case_development_sealed',case=case_id,candidates=len(metas),unique_representations=physical,
        choices=len(choices),seconds=perf_counter()-started,test_accessed=False)
    return dict(case_id=case_id,path=str(path),sha256=digest(path),
        candidates=len(metas),physical_representations=physical,requested_representations=requests)


def require_seal(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('status')!='abl_choices_sealed_before_test':raise RuntimeError('complete development seal required before test')
    for item in value['cases']:
        if digest(item['path'])!=item['sha256']:raise RuntimeError('case selection changed after seal')
    return value


def cpu_external_model(meta):
    if meta['family']=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(meta['source']['artifact'])
    if meta['family']=='xgboost':
        import xgboost as xgb
        model=xgb.Booster(model_file=meta['source']['artifact'])
        model.set_param(dict(device='cpu',nthread=2));return model
    raise ValueError('no new TabPFN context inference is permitted')


def external_raw(model,family,x,c):
    if family=='catboost':z=np.asarray(model.predict(x,prediction_type='RawFormulaVal',thread_count=2))
    else:
        import xgboost as xgb
        z=np.asarray(model.predict(xgb.DMatrix(x),output_margin=True))
    if z.ndim==1:z=np.column_stack((np.zeros(len(z)),z))
    if z.shape!=(len(x),c):raise RuntimeError('external class order/shape changed')
    return z.astype(np.float64)


def case_test_predictions(cfg,development,seal_path,output):
    require_seal(seal_path)
    record=development['record'];case,entry=record['case'],record['dataset']
    with np.load(case['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(entry['path'],allow_pickle=False) as f:
        x=f['features'][ids].astype(np.float32);target=f['target'][ids].astype(np.int64)
    directory=output/'test_predictions'/development['case_id'];directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'targets.npz',target=target,indices=ids)
    sources={};result={};audit=[];designs={}
    oldroot=Path(cfg['budget_quality_source']['report']).parent
    for uid,meta in development['metas'].items():
        source=meta['source']
        if digest(source['artifact'])!=source['sha256']:raise RuntimeError('source changed before test')
        family=meta['family'];origin=None
        if family=='compact':
            parent=meta['parent_uid']
            # At most eight membership matrices per task; release when done.
            if parent not in designs:
                model=native_load(source['artifact'])
                designs[parent]=leaf_design(model.base.leaf_indices(x),len(model.base.coefficients))
            else:model=native_load(source['artifact'])
            raw=design_raw(model,designs[parent]);origin='sealed_compact_model_cpu'
            del model
        else:
            if family not in sources:
                sources[family]=json.loads((oldroot/'test'/development['case_id']/family/'test.json').read_text())['predictions']
            cache=sources[family].get(meta['identity'])
            if cache:
                if digest(cache['path'])!=cache['sha256']:raise RuntimeError('old test cache changed')
                with np.load(cache['path'],allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['indices'],ids);np.testing.assert_array_equal(f['target'],target)
                    raw=f['raw'].astype(np.float64)
                origin='unchanged_saved_test_cache'
            elif family=='pyboost':
                model=native_load(source['artifact']).prepare_inference();raw=model.raw(x)
                origin='saved_full_model_cpu';del model
            elif family in ('catboost','xgboost'):
                # Device placement only changes the in-memory predictor, never
                # serialized trees. Replay all saved dev predictions first.
                model=cpu_external_model(meta)
                locks={};data=before_test_data(record,locks)
                replay=external_raw(model,family,data['dev_x'],case['outputs'])
                with np.load(meta['development_cache']['path']) as f:expected=f['raw']
                np.testing.assert_allclose(replay,expected,atol=2e-4,rtol=2e-5)
                raw=external_raw(model,family,x,case['outputs']);origin='saved_external_model_cpu_replayed'
                del model,data
            else:raise RuntimeError('fixed TabPFN test prediction missing; do not reload weights')
        cache=save_logits(directory/(uid.replace('/','-')+'.npz'),raw)
        result[uid]=cache;audit.append(dict(uid=uid,origin=origin,**cache))
    write(directory/'manifest.json',dict(case=development['case_id'],predictions=result,
        target_file=str(directory/'targets.npz'),target_sha256=digest(directory/'targets.npz'),
        pretest_seal_sha256=digest(seal_path),new_tree_fits=0))
    return target,result,audit


def study(cfg,output):
    check_contract(cfg);started=perf_counter();locks={}
    prechecked=locked(cfg['abl_preflight_completed']['path'],cfg['abl_preflight_completed']['sha256'])
    if prechecked['status']!='passed':raise RuntimeError('successful preflight required')
    for p,s in prechecked['input_locks'].items():pin(p,s,locks)
    original=source_input(cfg,locks)
    cases=[]
    for case_id in CASES:
        records=[r for r in original['records'] if r['case_id']==case_id]
        cases.append(case_development(cfg,records,case_id,prechecked,output,locks))
    for p,s in locks.items():
        if digest(p)!=s:raise RuntimeError('input changed before whole-study seal')
    if sum(c['requested_representations'] for c in cases)>480:raise RuntimeError('approved grid exceeded')
    seal=output/'pretest_choices.json'
    write(seal,dict(status='abl_choices_sealed_before_test',cases=cases,input_locks=locks,
        config_sha256=digest(output.parent.parent/'external_config.json'),
        new_tree_fits=0,gpu_execution=False,test_used_for_selection=False))
    event('abl_all_choices_sealed',cases=len(cases),sha256=digest(seal),test_accessed=False)
    # Import evaluation only after the sealed boundary. It cannot choose models.
    from abl_quality_evaluation import evaluate_study
    result=evaluate_study(cfg,output,seal)
    result.update(status='complete',config=cfg['abl_quality'],cases=cases,
        pretest_seal_sha256=digest(seal),new_tree_fits=0,new_boosting_rounds=0,gpu_execution=False,
        test_used_for_selection=False,seconds=perf_counter()-started,peak_cpu_rss_bytes=memory_guard())
    for p,s in locks.items():
        if digest(p)!=s:raise RuntimeError('immutable input changed during study')
    write(output/'report.json',result)
    event('abl_study_complete',report_path=str(output/'report.json'),report_sha256=digest(output/'report.json'),
        unique_representations=sum(c['physical_representations'] for c in cases),
        requested_representations=sum(c['requested_representations'] for c in cases),
        seconds=result['seconds'],new_tree_fits=0,gpu_execution=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    event('abl_effective_configuration',phase=cfg['phase'],configuration=cfg['abl_quality'],
        new_tree_fits=0,gpu_execution=False)
    if cfg['phase']=='abl_quality_preflight':preflight(cfg,args.output)
    elif cfg['phase']=='abl_quality_study':study(cfg,args.output)
    else:raise ValueError('unrecognized H-ABL phase')


if __name__=='__main__':main()
