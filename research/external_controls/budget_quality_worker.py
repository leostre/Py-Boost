"""Sequential bounded workers for H-QB1; interpreters and data are frozen."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parent
# Prefer the original snapshot when restored; otherwise use the publication source.
native_snapshot = ROOT/'native_source'
native_root = native_snapshot if (native_snapshot/'experiments').is_dir() else ROOT.parents[1]
sys.path.insert(0,str(native_root))
from budget_quality_core import variants, quality, select, paired_bootstrap


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.json')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def event(name,**kwargs): print(json.dumps(dict(event=name,**kwargs)),flush=True)


def locked(path,sha):
    if digest(path)!=sha: raise RuntimeError('immutable input changed: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source(cfg,name):
    r=cfg['bank_sources'][name]
    return locked(r['path'],r['sha256'])


def load_case(cfg,case_id,include_test=False):
    manifest=locked(cfg['matched_manifest'],cfg['matched_manifest_sha256'])
    case=next(c for c in manifest['cases'] if f'{c["dataset"]}_{c["seed"]}'==case_id)
    entry=manifest['datasets'][case['dataset']]
    if digest(entry['path'])!=entry['sha256'] or digest(case['split_file'])!=case['split_sha256']:
        raise RuntimeError('frozen data/roles changed')
    with np.load(case['split_file'],allow_pickle=False) as f:
        roles={r:f[r].copy() for r in ('train','calibration','control','selection','test')}
    with np.load(entry['path'],allow_pickle=False) as f:
        x=f['features'].astype(np.float32);y=f['target'].copy()
    indices=np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(indices),np.arange(len(y))):
        raise RuntimeError('roles are not a disjoint partition')
    if not np.array_equal(np.unique(y[roles['train']]),np.arange(case['outputs'])):
        raise RuntimeError('training class domain incomplete')
    names=('train','calibration','control','selection','test') if include_test else ('train','calibration','control','selection')
    data={r:(x[roles[r]],y[roles[r]]) for r in names}
    return case,entry,roles,data


def old_record(cfg,case_id,family):
    records=source(cfg,'native' if family=='pyboost' else 'external')['records']
    name='full' if family=='pyboost' else ('tabpfn_auto' if family=='tabpfn8' else family)
    return next(r for r in records if r['case_id']==case_id and r['family']==name)


def native_load(path):
    from experiments.uniform_threshold_core import ThresholdModel
    return ThresholdModel.load(path)


def native_raw(model,x):
    from experiments.history_basis_study import raw
    return raw(model,x)


def score(model,family,x,c):
    if family=='pyboost': return native_raw(model,x)
    if family=='catboost': z=np.asarray(model.predict(x,prediction_type='RawFormulaVal',thread_count=2))
    elif family=='xgboost':
        import xgboost as xgb
        z=np.asarray(model.predict(xgb.DMatrix(x),output_margin=True))
    elif family=='tabpfn8':
        p=np.asarray(model.predict_proba(x));z=np.log(np.clip(p,1e-12,1.))
        if not np.array_equal(model.classes_,np.arange(c)): raise RuntimeError('TabPFN class order changed')
    else: raise ValueError(family)
    if z.ndim==1 or (z.ndim==2 and z.shape[1]==1): z=np.column_stack((np.zeros(len(z)),z.reshape(-1)))
    if z.shape!=(len(x),c) or not np.isfinite(z).all(): raise RuntimeError('invalid canonical raw predictions')
    return z


def restore(item,family,cfg):
    if digest(item['artifact'])!=item['sha256']: raise RuntimeError('model changed before reload')
    if family=='pyboost': return native_load(item['artifact'])
    if family=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(item['artifact'])
    if family=='xgboost':
        import xgboost as xgb
        m=xgb.Booster(model_file=item['artifact']);m.set_param(dict(device='cuda',nthread=2));return m
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    return TabPFNClassifier.load_from_fit_state(item['artifact'],device='cuda')


def guard(cfg):
    s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    values=[int(v) for v in s.split()]
    if len(values)!=1 or values[0]*1024**2<cfg['minimum_free_bytes']:
        raise RuntimeError('insufficient shared A100 free memory; foreign jobs remain untouched')
    return values[0]*1024**2


def counts(model,family,rounds,c):
    if family=='pyboost':
        b=model.base
        return dict(rounds=rounds,trees=b.n_trees,leaves=len(b.coefficients),internal_nodes=len(b.nodes),
            leaf_parameters=b.coefficients.size,parameters=model.parameter_count,
            deployed_array_bytes=model.prepare_inference().resident_bytes,
            memory_kind='exact full deployed arrays; CSR+topology+bias+shift; float32 temperature extra')
    if family=='catboost':
        return dict(rounds=rounds,trees=int(model.tree_count_),leaves=int(model.get_tree_leaf_counts().sum()),
                    leaf_parameters=len(model.get_leaf_values()),memory_kind='native serialized bytes only; NOT resident memory')
    if family=='xgboost':
        from external_quality_core import tree_counts
        out=tree_counts([json.loads(t) for t in model.get_dump(dump_format='json')])
        if out['trees']!=rounds*(c if c>2 else 1): raise RuntimeError('scalar XGBoost tree count mismatch')
        return dict(out,rounds=rounds,memory_kind='native serialized bytes only; NOT resident memory')
    return dict(rounds=0,trees=0,leaves=0,leaf_parameters=0,memory_kind='shared TabPFN weights plus task context; not per-task allocation')


def save_model(model,family,path):
    if family=='pyboost': model.save(path)
    else: model.save_model(str(path))


def model_record(cfg,case_id,case,data,family,model,path,identity,depth,rounds,origin,fit,output_directory=None):
    from experiments.output_calibration_core import fit_decision_shifts
    directory=path.parent if output_directory is None else output_directory
    directory.mkdir(parents=True,exist_ok=True)
    if origin=='new': save_model(model,family,path)
    logits={r:score(model,family,data[r][0],case['outputs']) for r in ('calibration','control','selection')}
    npz=directory/(identity+'-development.npz')
    np.savez_compressed(npz,**logits)
    ledger=counts(model,family,rounds,case['outputs'])
    ledger['native_file_bytes']=path.stat().st_size
    # PyBoost includes existing C shifts already, plus one float32 T.
    # External native files need C float32 shifts plus T added separately.
    basebytes=(ledger['deployed_array_bytes']+4 if family=='pyboost'
               else path.stat().st_size+4*case['outputs']+4)
    rows=variants(logits,{r:data[r][1] for r in logits},fit_decision_shifts,
                  cfg['decision_grid'],cfg['decision_shrinkage'])
    for i,row in enumerate(rows):
        row.update(uid=f'{case_id}/{family}/{identity}/{i}',case_id=case_id,family=family,
            model_id=identity,cost_bytes=basebytes,origin=origin,depth=depth,rounds=rounds)
    item=dict(artifact=str(path),sha256=digest(path),identity=identity,depth=depth,rounds=rounds,
        origin=origin,case_id=case_id,family=family,ledger=ledger,fit=fit,
        development_file=str(npz),development_sha256=digest(npz),rows=rows)
    write(directory/(identity+'-record.json'),item)
    event('budget_snapshot_saved',case=case_id,family=family,identity=identity,origin=origin,
          rounds=rounds,cost_bytes=basebytes,raw_selection_loss=rows[0]['selection']['loss'],test_evaluated=False)
    return item


def training_worker(cfg,case_id,family,directory):
    directory.mkdir(parents=True,exist_ok=True)
    case,entry,roles,data=load_case(cfg,case_id)
    guard(cfg);records=[]
    old=old_record(cfg,case_id,family)
    if family=='tabpfn8':
        # Reuse the saved fitted context and eight unchanged estimators; no refit.
        item=next(iter(old['artifacts'].values()))
        model=restore(item,family,cfg)
        records.append(model_record(cfg,case_id,case,data,family,model,Path(item['artifact']),
                                    'frozen8',0,0,'frozen_reference',dict(new_fits=0),directory))
        del model
    else:
        for depth in cfg['depths']:
            guard(cfg);started=perf_counter()
            if family=='pyboost':
                from experiments.leaf_budget_runtime import fit_dense
                from experiments.shared_leaf_basis import reference_class_model
                from experiments.closed_budget_study import empty_model
                x=np.zeros((sum(len(v) for v in roles.values()),data['train'][0].shape[1]),np.float32)
                y=np.zeros(len(x),np.int32)
                for r in data: x[roles[r]]=data[r][0];y[roles[r]]=data[r][1]
                dense,fit=fit_dense(case,depth,cfg,x,y,roles,trees=cfg['rounds'])
                full=empty_model(reference_class_model(dense))
            elif family=='catboost':
                from catboost import CatBoostClassifier
                full=CatBoostClassifier(iterations=cfg['rounds'],depth=depth,learning_rate=.1,l2_leaf_reg=1,
                    loss_function='Logloss' if case['outputs']==2 else 'MultiClass',task_type='GPU',devices='0',
                    border_count=128,bootstrap_type='No',random_strength=0,boosting_type='Plain',
                    random_seed=case['seed'],thread_count=2,gpu_ram_part=.25,verbose=False,allow_writing_files=False)
                full.fit(*data['train']);fit={}
            elif family=='xgboost':
                import xgboost as xgb
                params=dict(max_depth=depth,eta=.1,reg_lambda=1,max_bin=128,subsample=1.,colsample_bytree=1.,
                    objective='binary:logistic' if case['outputs']==2 else 'multi:softprob',device='cuda',
                    tree_method='hist',nthread=2,seed=case['seed'],verbosity=1)
                if case['outputs']>2: params['num_class']=case['outputs']
                full=xgb.train(params,xgb.DMatrix(data['train'][0],label=data['train'][1]),num_boost_round=cfg['rounds'])
                fit={}
            else: raise ValueError(family)
            fit.update(seconds=perf_counter()-started,rounds=cfg['rounds'],depth=depth,
                       interpreter=sys.executable,new_fits=1)
            write(directory/f'd{depth}-fit.json',fit)
            event('budget_trajectory_trained',case=case_id,family=family,**fit)
            for rounds in cfg['prefixes']:
                if family=='pyboost':
                    from experiments.parallel_leaf_core import prefix
                    model=prefix(full,rounds)
                elif family=='catboost': model=full.copy();model.shrink(rounds)
                else: model=full[:rounds]
                identity=f'd{depth}-t{rounds}'
                suffix={'pyboost':'.npz','catboost':'.cbm','xgboost':'.ubj'}[family]
                records.append(model_record(cfg,case_id,case,data,family,model,directory/(identity+suffix),
                                            identity,depth,rounds,'new',fit))
                del model
            del full;gc.collect()
        # Depth4 stays an explicitly labelled historical control; old d5t200 fixes B0.
        for identity,item in old['artifacts'].items():
            if item['depth']!=4 and identity!='d5-t200': continue
            model=restore(item,family,cfg)
            records.append(model_record(cfg,case_id,case,data,family,model,Path(item['artifact']),
                'archival-'+identity,item['depth'],item['rounds'],'archival_control',dict(new_fits=0),directory))
            del model
    write(directory/'record.json',dict(case_id=case_id,case=case,dataset=entry,family=family,models=records,
                                      test_evaluated=False))


def seal_choices(cfg,records,output):
    choices=[]
    for case_id in cfg['quality_cases']:
        case_records=[r for r in records if r['case_id']==case_id]
        native=next(r for r in case_records if r['family']=='pyboost')
        oldbase=next(m for m in native['models'] if m['identity']=='archival-d5-t200')
        b0=oldbase['ledger']['deployed_array_bytes']
        for record in case_records:
            family=record['family']
            for policy in ('raw','temperature','raw_shifts','temperature_shifts'):
                rows=[r for m in record['models'] if m['origin']!='archival_control'
                      for r in m['rows'] if r['policy']==policy]
                for objective in ('loss','f1_macro'):
                    if family=='pyboost': budgets=[(v,int(np.floor(v*b0))) for v in cfg['budget_ratios']]
                    elif family=='tabpfn8': budgets=[('reference',None)]
                    else: budgets=[(f'rounds{t}',None) for t in cfg['prefixes']]
                    for label,budget in budgets:
                        pool=rows if family in ('pyboost','tabpfn8') else [r for r in rows if r['rounds']==int(label[6:])]
                        chosen=select(pool,objective,budget)
                        choices.append(dict(case_id=case_id,family=family,policy=policy,objective=objective,
                            budget_label=label,budget_bytes=budget,reference_b0_bytes=b0,chosen=chosen))
            if family!='tabpfn8':
                for m in record['models']:
                    if m['identity']=='archival-d5-t200':
                        for r in m['rows']:
                            if r['shift_fraction']==0 and r['policy'] in ('raw','temperature'):
                                choices.append(dict(case_id=case_id,family=family,policy=r['policy'],
                                    objective='historical200',budget_label='archival',chosen=r,reference_b0_bytes=b0))
    locks={cfg['matched_manifest']:cfg['matched_manifest_sha256']}
    for record in records:
        locks[record['case']['split_file']]=record['case']['split_sha256']
        locks[record['dataset']['path']]=record['dataset']['sha256']
        for model in record['models']:
            locks[model['artifact']]=model['sha256']
            locks[model['development_file']]=model['development_sha256']
    for path,sha in locks.items():
        if digest(path)!=sha: raise RuntimeError('input changed before pretest seal')
    result=dict(status='choices_sealed_before_test',choices=choices,records=records,input_locks=locks,
        test_evaluated=False,quality_gate_5percent=False,selection_roles='T:calibration,shifts:control,ranking:selection')
    write(output/'pretest_choices.json',result)
    event('all_budget_choices_sealed',choices=len(choices),sha256=digest(output/'pretest_choices.json'),test_evaluated=False)


def test_worker(cfg,record,output,seal,audit_only=False):
    evidence=locked(seal['path'],seal['sha256'])
    if evidence['status']!='choices_sealed_before_test': raise RuntimeError('pretest seal missing')
    models={m['identity']:m for m in record['models']}
    chosen=[x['chosen'] for x in evidence['choices'] if x['case_id']==record['case_id'] and
            x['family']==record['family'] and x['chosen'] is not None]
    selected=sorted({r['model_id'] for r in chosen})
    predictions={};audit=[]
    if audit_only:
        case,_,roles,data=load_case(cfg,record['case_id'])
        # Whole-cohort replay is completed by the parent before any test worker.
        for identity in selected:
            item=models[identity];model=restore(item,record['family'],cfg)
            raw=score(model,record['family'],data['selection'][0],case['outputs'])
            if digest(item['development_file'])!=item['development_sha256']: raise RuntimeError('development logits changed')
            with np.load(item['development_file']) as f: expected=f['selection'].copy()
            np.testing.assert_allclose(raw,expected,atol=2e-4,rtol=2e-5)
            audit.append(dict(identity=identity,max_replay_error=float(np.max(np.abs(raw-expected)))))
            del model;gc.collect()
        write(output/'pretest_replay.json',dict(audit=audit,test_evaluated=False,pretest_seal_sha256=seal['sha256']))
        return
    all_audit=Path(seal['path']).parent/'whole_cohort_audit.json'
    global_evidence=json.loads(all_audit.read_text())
    if global_evidence['pretest_seal_sha256']!=seal['sha256'] or global_evidence['status']!='passed':
        raise RuntimeError('whole cohort audit missing before test')
    case,_,roles,data=load_case(cfg,record['case_id'],include_test=True)
    for identity in selected:
        guard(cfg);item=models[identity];model=restore(item,record['family'],cfg)
        z=score(model,record['family'],data['test'][0],case['outputs'])
        path=output/(identity+'-test.npz')
        np.savez_compressed(path,raw=z,target=data['test'][1],indices=roles['test'])
        predictions[identity]=dict(path=str(path),sha256=digest(path))
        del model;gc.collect()
        event('budget_test_saved',case=record['case_id'],family=record['family'],identity=identity)
    write(output/'test.json',dict(predictions=predictions,pretest_seal_sha256=seal['sha256'],new_fits=0))


def summarize(cfg,records,output):
    evidence=json.loads((output/'pretest_choices.json').read_text(encoding='utf-8'))
    cache={};rows=[];interval_cache={}
    def prediction(case,family,identity):
        key=(case,family,identity)
        if key not in cache:
            manifest=json.loads((output/'test'/case/family/'test.json').read_text())
            item=manifest['predictions'][identity]
            if digest(item['path'])!=item['sha256']: raise RuntimeError('saved test changed')
            with np.load(item['path']) as f: cache[key]={k:f[k].copy() for k in f.files}
        return cache[key]
    for choice in evidence['choices']:
        if choice['chosen'] is None:
            rows.append(dict(choice,status='no_feasible_candidate'));continue
        r=choice['chosen'];case=choice['case_id'];family=choice['family']
        pred=prediction(case,family,r['model_id']);y=pred['target']
        q=quality(y,pred['raw'],r['temperature'],r['decision_shifts'])
        ref_policy='temperature' if r['policy'].startswith('temperature') else 'raw'
        reference=next(c['chosen'] for c in evidence['choices'] if c['case_id']==case and
            c['family']=='tabpfn8' and c['policy']==ref_policy and c['objective']=='loss')
        ref=prediction(case,'tabpfn8',reference['model_id'])
        if not np.array_equal(pred['indices'],ref['indices']) or not np.array_equal(y,ref['target']):
            raise RuntimeError('test comparisons are not row-paired')
        key=(r['uid'],reference['uid'])
        if key not in interval_cache:
            interval_cache[key]=paired_bootstrap(y,(pred['raw'],r['temperature'],r['decision_shifts']),
                (ref['raw'],reference['temperature'],reference['decision_shifts']),cfg['bootstrap_repeats'])
        rows.append(dict(choice,status='evaluated',test=q,comparison=interval_cache[key],
            reference_policy=ref_policy,reference_test=quality(y,ref['raw'],reference['temperature'],reference['decision_shifts'])))
    write(output/'report.json',dict(status='complete',results=rows,
        pretest_seal_sha256=digest(output/'pretest_choices.json'),
        scientific_trajectories=30,new_boosting_rounds=24000,nn_weight_updates=0,
        datasets=cfg['quality_cases'],new_depths=cfg['depths'],prefixes=cfg['prefixes'],
        test_is_reused_exploratory=True,test_used_for_selection=False,
        scope='stage1 only; no AB+L compression, capacity, latency or new datasets',
        reference='frozen TabPFN8; raw-to-raw and calibrated-to-calibrated',
        native_storage_not_resident_memory=True,quality_gate_5percent=False))
    event('budget_quality_stage1_complete',rows=len(rows),new_trajectories=30,new_boosting_rounds=24000,
          report_sha256=digest(output/'report.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--case');parser.add_argument('--family');parser.add_argument('--test-record',type=Path)
    parser.add_argument('--seal',type=Path);parser.add_argument('--seal-sha');parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.test_record:
        test_worker(cfg,json.loads(args.test_record.read_text()),args.output,dict(path=args.seal,sha256=args.seal_sha),args.audit_only);return
    if args.case:
        if args.family=='pyboost':
            from experiments.a100_runner import validate_environment
            validate_environment(dict(environment_lock=cfg['frozen_hasboost_environment']))
        training_worker(cfg,args.case,args.family,args.output);return
    subprocess.run([sys.executable,'-m','pytest','-q','-o','pythonpath=.',
                    'tests/test_budget_quality_core.py'],check=True,timeout=300)
    if (cfg['rounds'],cfg['depths'],cfg['prefixes'])!=(800,[3,5],[40,80,160,200,400,800]):
        raise RuntimeError('approved trajectory configuration changed')
    if len(cfg['quality_cases'])!=5 or len(set(cfg['quality_cases']))!=5:
        raise RuntimeError('approved five-task scientific cap changed')
    records=[]
    for case in cfg['quality_cases']:
        for family in ('pyboost','catboost','xgboost','tabpfn8'):
            directory=args.output/'development'/case/family
            exe='/root/hasboost_compactness_20260928/venv/bin/python' if family=='pyboost' else sys.executable
            subprocess.run([exe,__file__,str(args.config),str(directory),'--case',case,'--family',family],
                           check=True,timeout=cfg['worker_timeout_seconds'])
            records.append(json.loads((directory/'record.json').read_text()))
            write(args.output/'development_manifest.json',dict(records=records,test_evaluated=False))
    seal_choices(cfg,records,args.output)
    seal=args.output/'pretest_choices.json';sha=digest(seal)
    for mode in ('audit','test'):
        for record in records:
            family=record['family'];directory=args.output/'test'/record['case_id']/family
            path=args.output/'development'/record['case_id']/family/'record.json'
            exe='/root/hasboost_compactness_20260928/venv/bin/python' if family=='pyboost' else sys.executable
            command=[exe,__file__,str(args.config),str(directory),'--test-record',str(path),
                     '--seal',str(seal),'--seal-sha',sha]
            if mode=='audit': command.append('--audit-only')
            subprocess.run(command,check=True,timeout=cfg['worker_timeout_seconds'])
        if mode=='audit':
            for path,value in json.loads(seal.read_text())['input_locks'].items():
                if digest(path)!=value: raise RuntimeError('locked input changed during cohort audit')
            write(args.output/'whole_cohort_audit.json',dict(status='passed',pretest_seal_sha256=sha,
                cases=len(records),test_evaluated=False))
    summarize(cfg,records,args.output)


if __name__=='__main__': main()
