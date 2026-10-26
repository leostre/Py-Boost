"""Separate-process matched development; no test predictions or metrics."""
import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import numpy as np
from external_quality_core import canonical_raw,calibrated_rows,choose,tree_counts
from matched_data import digest
from sharing_worker import rss
from storage_graph import inventory


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def load_inputs(cfg,case_id):
    path=Path(cfg['matched_manifest'])
    if digest(path)!=cfg['matched_manifest_sha256']: raise RuntimeError('matched data seal changed')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    case=next(c for c in manifest['cases'] if f'{c["dataset"]}_{c["seed"]}'==case_id)
    entry=manifest['datasets'][case['dataset']]
    if digest(entry['path'])!=entry['sha256'] or digest(case['split_file'])!=case['split_sha256']:
        raise RuntimeError('matched data or role bytes changed')
    with np.load(entry['path'],allow_pickle=False) as f:
        x,y=f['features'],f['target']
        with np.load(case['split_file'],allow_pickle=False) as p:
            indices={k:p[k].copy() for k in ('train','calibration','control','selection','test')}
        joined=np.concatenate(list(indices.values()))
        if not np.array_equal(np.sort(joined),np.arange(len(x))): raise RuntimeError('roles overlap or incomplete')
        # Nothing outside these three roles is exposed to the model worker.
        data={k:(np.ascontiguousarray(x[indices[k]],dtype=np.float32),y[indices[k]].copy())
              for k in ('train','calibration','selection')}
    c=case['outputs']
    if not np.array_equal(np.unique(data['train'][1]),np.arange(c)):
        raise RuntimeError('training class domain incomplete')
    return case,entry,data


def gpu_guard(cfg):
    out=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=20).stdout.strip().splitlines()
    if len(out)!=1: raise RuntimeError('expected one approved GPU')
    free=int(out[0])*1024**2
    if free<cfg['minimum_free_bytes']: raise RuntimeError('insufficient free shared A100 memory')
    return free


def score_native(model,family,x,c):
    if family=='catboost':
        return canonical_raw(model.predict(x,prediction_type='RawFormulaVal',thread_count=2),model.classes_,c)
    import xgboost as xgb
    return canonical_raw(model.predict(xgb.DMatrix(x),output_margin=True),np.arange(c),c)


def native_family(cfg,case,data,family,directory):
    train,y=data['train']
    c=case['outputs']
    rows,artifacts,raw_files,fits=[],{}, {},[]
    for depth in cfg['depths']:
        available=gpu_guard(cfg)
        started=perf_counter()
        if family=='catboost':
            from catboost import CatBoostClassifier
            model=CatBoostClassifier(iterations=cfg['rounds'],depth=depth,learning_rate=.1,l2_leaf_reg=1,
                loss_function='Logloss' if c==2 else 'MultiClass',task_type='GPU',devices='0',
                border_count=128,bootstrap_type='No',random_strength=0,boosting_type='Plain',
                random_seed=case['seed'],thread_count=2,gpu_ram_part=.25,verbose=False,allow_writing_files=False)
            model.fit(train,y)
            full=model
        else:
            import xgboost as xgb
            params=dict(max_depth=depth,eta=.1,reg_lambda=1,max_bin=128,subsample=1.,colsample_bytree=1.,
                        objective='binary:logistic' if c==2 else 'multi:softprob',device='cuda',
                        tree_method='hist',nthread=2,seed=case['seed'],verbosity=1)
            if c!=2: params['num_class']=c
            full=xgb.train(params,xgb.DMatrix(train,label=y),num_boost_round=cfg['rounds'])
        elapsed=perf_counter()-started
        fits.append(dict(depth=depth,rounds=cfg['rounds'],seconds=elapsed,
            expected_scalar_trees=cfg['rounds']*(c if family=='xgboost' and c>2 else 1),
            free_gpu_bytes_before=available,own_native_gpu_peak_bytes=None,
            gpu_peak_limitation='native allocators not instrumented; not inferred from whole-device free memory'))
        for rounds in cfg['prefixes']:
            identity=f'd{depth}-t{rounds}'
            if family=='catboost':
                small=full.copy()
                small.shrink(rounds)
                path=directory/(identity+'.cbm')
                small.save_model(str(path))
                counts=dict(trees=int(small.tree_count_),leaves=int(small.get_tree_leaf_counts().sum()),
                    leaf_parameters=int(len(small.get_leaf_values())),
                    structure='symmetric splits per depth; not generic binary-node storage')
            else:
                small=full[:rounds]
                path=directory/(identity+'.ubj')
                small.save_model(path)
                counts=tree_counts([json.loads(s) for s in small.get_dump(dump_format='json')])
                expected=rounds*(c if c>2 else 1)
                if counts['trees']!=expected: raise RuntimeError('XGBoost round/scalar-tree accounting mismatch')
            start=perf_counter()
            a=score_native(small,family,data['calibration'][0],c)
            b=score_native(small,family,data['selection'][0],c)
            rawpath=directory/(identity+'-development.npz')
            np.savez_compressed(rawpath,calibration=a,selection=b)
            cost=dict(payload_bytes=path.stat().st_size+4*c,parameters=counts['leaf_parameters']+c)
            rows.extend(calibrated_rows(a,data['calibration'][1],b,data['selection'][1],'multiclass',identity,
                cost,cfg['decision_grid'],cfg['decision_shrinkage']))
            artifacts[identity]=dict(artifact=str(path),sha256=digest(path),native_bytes=path.stat().st_size,
                                     counts=counts,rounds=rounds,depth=depth)
            raw_files[identity]=dict(artifact=str(rawpath),sha256=digest(rawpath),prediction_seconds=perf_counter()-start)
            del small,a,b
        del full
        if family=='catboost': del model
        gc.collect()
    chosen=choose(rows)
    return dict(family=family,chosen=chosen,candidates=rows,artifacts=artifacts,
                development_predictions=raw_files,fits=fits,nn_weight_updates=0)


def tfm_family(cfg,case,data,family,directory):
    import torch
    from tabpfn import TabPFNClassifier
    available=gpu_guard(cfg)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    weights=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']/cfg['weights_filename']
    if digest(weights)!=cfg['weights_sha256']: raise RuntimeError('licensed weight SHA changed')
    started=perf_counter()
    model=TabPFNClassifier(model_path=str(weights),n_estimators='auto' if family=='tabpfn_auto' else 1,
        device='cuda',inference_precision='auto',fit_mode='fit_preprocessors',memory_saving_mode='auto',
        enable_torch_compile=False,random_state=case['seed'],n_preprocessing_jobs=2,show_progress_bar=False)
    model.fit(*data['train'])
    torch.cuda.synchronize()
    fit_seconds=perf_counter()-started
    if model.inference_config_.MAX_NUMBER_OF_CLASSES<case['outputs']:
        raise RuntimeError('checkpoint class capability unexpectedly reduced')
    start=perf_counter()
    a=canonical_raw(model.predict_proba(data['calibration'][0]),model.classes_,case['outputs'],probabilities=True)
    b=canonical_raw(model.predict_proba(data['selection'][0]),model.classes_,case['outputs'],probabilities=True)
    torch.cuda.synchronize()
    predict_seconds=perf_counter()-start
    path=directory/'context.tabpfn_fit'
    model.save_fit_state(path)
    rawpath=directory/'development.npz'
    np.savez_compressed(rawpath,calibration=a,selection=b)
    identity='official-default' if family=='tabpfn_auto' else 'one-estimator'
    cost=dict(payload_bytes=cfg['weights_bytes']+path.stat().st_size+4*case['outputs'],parameters=218987657+case['outputs'])
    rows=calibrated_rows(a,data['calibration'][1],b,data['selection'][1],'multiclass',identity,
                        cost,cfg['decision_grid'],cfg['decision_shrinkage'])
    return dict(family=family,chosen=choose(rows),candidates=rows,
        artifacts={identity:dict(artifact=str(path),sha256=digest(path),context_file_bytes=path.stat().st_size,
                    shared_checkpoint_bytes=cfg['weights_bytes'],checkpoint_sha256=cfg['weights_sha256'])},
        development_predictions={identity:dict(artifact=str(rawpath),sha256=digest(rawpath),prediction_seconds=predict_seconds)},
        fits=[dict(seconds=fit_seconds,context_fit_calls=1,nn_weight_updates=0,
            actual_n_estimators=int(model.n_estimators_),class_limit=int(model.inference_config_.MAX_NUMBER_OF_CLASSES),
            precision=str(model.inference_precision),resolved_dtype=str(getattr(model,'forced_inference_dtype_',None)),
            free_gpu_bytes_before=available,reachable_state=inventory(model),
            cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())],
        nn_weight_updates=0,context_not_free=True)


def child(cfg,case_id,family,directory):
    directory.mkdir(parents=True,exist_ok=True)
    case,entry,data=load_inputs(cfg,case_id)
    started=perf_counter()
    result=(native_family if family in ('catboost','xgboost') else tfm_family)(cfg,case,data,family,directory)
    result.update(case_id=case_id,case=case,dataset=entry,status='development_complete',test_evaluated=False,
        elapsed_seconds=perf_counter()-started,rss_bytes=rss(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(directory/'record.json',result)
    print(json.dumps(dict(event='external_development_case_sealed',case_id=case_id,family=family,
        validation=result['chosen']['validation'],elapsed_seconds=result['elapsed_seconds'],test_evaluated=False)),flush=True)


def smoke(cfg,directory):
    """Technical APIs on synthetic objects, priced separately from scientific fits."""
    rng=np.random.default_rng(20261021)
    records=[]
    for c in (2,3):
        x=rng.normal(size=(96,6)).astype(np.float32)
        y=np.arange(96,dtype=np.int32)%c
        data=dict(train=(x[:64],y[:64]),calibration=(x[64:80],y[64:80]),selection=(x[80:],y[80:]))
        case=dict(outputs=c,seed=910+c)
        for family in cfg['quality_families']:
            out=directory/f'c{c}'/family
            out.mkdir(parents=True)
            local=dict(cfg,depths=[3],rounds=4,prefixes=[2,4])
            record=(native_family if family in ('catboost','xgboost') else tfm_family)(local,case,data,family,out)
            item=record['artifacts'][record['chosen']['candidate_id']]
            cached=record['development_predictions'][record['chosen']['candidate_id']]
            with np.load(cached['artifact']) as f: expected=f['selection'].copy()
            if family=='catboost':
                from catboost import CatBoostClassifier
                model=CatBoostClassifier().load_model(item['artifact'])
                actual=score_native(model,family,data['selection'][0],c)
            elif family=='xgboost':
                import xgboost as xgb
                model=xgb.Booster(model_file=item['artifact'])
                actual=score_native(model,family,data['selection'][0],c)
            else:
                from tabpfn import TabPFNClassifier
                model=TabPFNClassifier.load_from_fit_state(item['artifact'],device='cuda')
                actual=canonical_raw(model.predict_proba(data['selection'][0]),model.classes_,c,probabilities=True)
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
            del model
            gc.collect()
            records.append(dict(family=family,outputs=c,reload_max_abs_error=float(np.max(np.abs(actual-expected))),
                                technical_fits=len(record['fits'])))
    write(directory/'smoke.json',dict(status='verified',records=records,scientific_fits=0,nn_weight_updates=0))
    print(json.dumps(dict(event='external_synthetic_gate_passed',workers=len(records),scientific_fits=0)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--case')
    parser.add_argument('--family')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        smoke(cfg,args.output)
        return
    if args.case:
        child(cfg,args.case,args.family,args.output)
        return
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_external_quality_core.py','tests/test_matched_data.py'],
                   check=True,timeout=300)
    subprocess.run([sys.executable,__file__,str(args.config),str(args.output/'synthetic'),'--smoke'],check=True,timeout=1200)
    records=[]
    for case in cfg['quality_cases']:
        for family in cfg['quality_families']:
            directory=args.output/case/family
            subprocess.run([sys.executable,__file__,str(args.config),str(directory),'--case',case,'--family',family],
                           check=True,timeout=cfg['worker_timeout_seconds'])
            records.append(json.loads((directory/'record.json').read_text(encoding='utf-8')))
            write(args.output/'progress.json',dict(records=records,test_evaluated=False))
    write(args.output/'development_manifest.json',dict(status='development_complete',records=records,
        matched_manifest=cfg['matched_manifest'],matched_manifest_sha256=cfg['matched_manifest_sha256'],
        external_environment_sha256=cfg['external_environment_sha256'],nn_weight_updates=0,test_evaluated=False))
    print(json.dumps(dict(event='external_development_complete',records=len(records),
        development_manifest_sha256=digest(args.output/'development_manifest.json'),test_evaluated=False)),flush=True)


if __name__=='__main__': main()
