"""Matched full-D calibration and cached TabPFN8 bank, frozen environments."""
from pathlib import Path
from time import perf_counter,perf_counter_ns
import gc,json,sys,os,resource
from dataclasses import asdict
import numpy as np
from scipy.special import softmax
from budget_quality_worker import locked,digest,write,event
from unified_external_core import data_before_test,evaluate,quality_row
from unified_frontier_core import make_folds
from external_quality_core import canonical_raw
from quality_worker import gpu_guard
from tfm_bank_worker import shared_spec,gpu_processes
from tfm_serving_core import request_plan,gpu_owner
from storage_graph import inventory,numerical_spans,union_bytes
from unified_evaluation_audit import audit_evaluations
from unified_gpu_ownership import torch_owner


def load_external(family,path):
    if family=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(path)
    import xgboost as xgb
    result=xgb.Booster(model_file=path);result.set_param({'nthread':2,'device':'cpu'});return result


def score_native(model,family,features,outputs):
    if family=='catboost':
        return canonical_raw(model.predict(features,prediction_type='RawFormulaVal',thread_count=2),model.classes_,outputs)
    import xgboost as xgb
    return canonical_raw(model.predict(xgb.DMatrix(features,nthread=2),output_margin=True),np.arange(outputs),outputs)


def summary(values):
    return dict(zip(('p50_ms','p95_ms','p99_ms'),map(float,np.percentile(values,[50,95,99]))))


def test_features(record):
    # No target/test labels exposed; predictions are sealed for a later evaluator.
    with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(record['dataset']['path'],allow_pickle=False) as f:return f['features'][ids].copy()


def run(cfg,out):
    started=perf_counter();spec=cfg['unified_external'];out.mkdir(parents=True,exist_ok=True)
    if spec.get('u4'):
        u4=locked(spec['u4']['path'],spec['u4']['sha256']);assert u4['status']=='unified_serving_complete'
    source_ref=spec.get('source_manifest',dict(path=cfg['quality_development_manifest'],sha256=cfg['quality_development_sha256']))
    source=locked(source_ref['path'],source_ref['sha256'])
    selected={};curves=[];inputs={};records={};tests={};test_predictions=[]
    # Strong ordinary controls keep every old depth/width, with the same shared-D scheme.
    for family in ('catboost','xgboost'):
        for cid in spec['cases']:
            record=next(r for r in source['records'] if r['case_id']==cid and r['family']==family)
            records[cid]=record;data=data_before_test(record);inputs[cid]=data['dev_x'];tests[cid]=test_features(record)
            folds=make_folds(data['dev_y'],'multiclass',record['case']['seed']+20261028);models=[]
            for uid,ref in record['artifacts'].items():
                assert digest(ref['artifact'])==ref['sha256'];model=load_external(family,ref['artifact'])
                raw=score_native(model,family,data['dev_x'],record['case']['outputs']);directory=out/cid/family/uid;directory.mkdir(parents=True,exist_ok=True)
                ev=evaluate(raw,data['dev_y'],'multiclass',folds,cfg,directory)
                raw_path=directory/'raw.npz';np.savez_compressed(raw_path,raw=raw)
                item=dict(uid=uid,case=cid,family=family,source=ref,cost_bytes=ref['native_bytes']+4+4*record['case']['outputs'],evaluations=ev,
                    raw=dict(path=str(raw_path),sha256=digest(raw_path)),audit=audit_evaluations(raw,data['dev_y'],'multiclass',ev),
                    cost_scope='native serialized artifact plus T and shifts; not process RSS or live C++ buffers')
                models.append(item);del model;gc.collect()
            chosen=min(models,key=lambda m:(quality_row(m['evaluations'],'multiclass')['quality']['loss'],m['cost_bytes'],m['uid']))
            selected[cid,family]=chosen;curves.extend(models);write(out/'ordinary_curves.json',curves)
            event('unified_external_curve',case=cid,family=family,models=len(models),selected=chosen['uid'])
    before_gpu=gpu_processes();available=gpu_guard(cfg)
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    weights=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']/cfg['weights_filename']
    assert digest(weights)==cfg['weights_sha256'];model_spec=shared_spec(weights,torch)
    tfms={};controls={};fits=[];probes={};tfm_points=[];prefix_memory=[]
    for cid in spec['cases']:
        record=records[cid];data=data_before_test(record);directory=out/cid/'tabpfn_auto';directory.mkdir(parents=True,exist_ok=True)
        t=perf_counter();model=TabPFNClassifier(model_path=model_spec,n_estimators='auto',device='cuda',inference_precision='auto',
            fit_mode='fit_with_cache',memory_saving_mode='auto',enable_torch_compile=False,random_state=record['case']['seed'],
            n_preprocessing_jobs=2,show_progress_bar=False)
        model.fit(data['train_x'],data['train_y']);torch.cuda.synchronize();assert model.n_estimators_==8
        fits.append(dict(case=cid,seconds=perf_counter()-t,estimators=8,weight_updates=0,train_rows=len(data['train_y'])))
        raw=canonical_raw(model.predict_proba(data['dev_x']),model.classes_,record['case']['outputs'],probabilities=True)
        folds=make_folds(data['dev_y'],'multiclass',record['case']['seed']+20261028)
        ev=evaluate(raw,data['dev_y'],'multiclass',folds,cfg,directory);control=quality_row(ev,'multiclass')['refit_control']
        raw_path=directory/'raw.npz';np.savez_compressed(raw_path,raw=raw)
        tfms[cid]=model;controls[cid]=control;probes[cid]=model.predict_proba(inputs[cid][:17])
        point=dict(case=cid,family='tabpfn_auto',uid='official-default',evaluations=ev,raw=dict(path=str(raw_path),sha256=digest(raw_path)),audit=audit_evaluations(raw,data['dev_y'],'multiclass',ev))
        tfm_points.append(point);write(out/'tfm_progress.json',dict(fits=fits,points=tfm_points))
        gc.collect();torch.cuda.empty_cache()
        prefix_memory.append(dict(tasks=len(tfms),last_task=cid,storage=inventory(list(tfms.values())),
            torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved()))
        write(out/'resident_prefix_memory.json',prefix_memory)
        event('unified_tfm_context_ready',**fits[-1])
    assert len({id(m.models_[0]) for m in tfms.values()})==1
    shared=union_bytes([s for m in tfms.values() for s in numerical_spans(m.models_)])
    assert shared==union_bytes(numerical_spans(next(iter(tfms.values())).models_))
    isolation=[]
    for cid,model in tfms.items():
        error=float(np.max(np.abs(model.predict_proba(inputs[cid][:17])-probes[cid])))
        assert error<=2e-5;isolation.append(dict(case=cid,max_abs=error))
    namespaces={os.getpid()}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):namespaces.update(map(int,line.split()[1:]))
    initialization_peaks=dict(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    owner=torch_owner(gpu_processes,namespaces)
    gc.collect();torch.cuda.empty_cache()
    storage=inventory(list(tfms.values()));storage.update(shared_weight_numerical_bytes=shared,
        torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
        driver_bytes=gpu_processes().get(owner['pid']) if owner['verified'] else None,gpu_ownership=owner)
    # Addresses describe simultaneously live, reachable numerical buffers only.
    # They permit subset-union accounting later; no pointer is dereferenced.
    context_spans={cid:[asdict(s) for s in numerical_spans(model)] for cid,model in tfms.items()}
    seal=dict(cases=spec['cases'],ordinary_selected=list(selected.values()),tfm_points=tfm_points,controls=controls,
        fits=fits,isolation=isolation,storage=storage,context_spans=context_spans,initialization_peaks=initialization_peaks,test_evaluated=False,quality_scope='adaptive shared-D crossfit')
    write(out/'pretest_seal.json',seal)
    # Frozen choices now produce raw test predictions, without reading test labels.
    for cid in spec['cases']:
        model=tfms[cid];raw=canonical_raw(model.predict_proba(tests[cid]),model.classes_,records[cid]['case']['outputs'],probabilities=True)
        file=out/cid/'tabpfn_auto'/'sealed_test_raw.npz';np.savez_compressed(file,raw=raw)
        test_predictions.append(dict(case=cid,family='tabpfn_auto',path=str(file),sha256=digest(file),record=records[cid],control=controls[cid]))
        write(out/'sealed_test_predictions.json',test_predictions)
    timings=[]
    # Cached mode first; each completed request is persisted immediately.
    for family in ('tabpfn_auto','catboost','xgboost'):
        models=tfms if family=='tabpfn_auto' else {cid:load_external(family,selected[cid,family]['source']['artifact']) for cid in spec['cases']}
        for batch in spec['batches']:
            for traffic in ('uniform','skew'):
                rows=[];schedule=request_plan(len(spec['cases']),spec['requests'],traffic)
                # Warm each resident task on this same batch size before timing.
                for cid in spec['cases']:
                    x=inputs[cid][np.arange(batch)%len(inputs[cid])]
                    if family=='tabpfn_auto':models[cid].predict_proba(x);torch.cuda.synchronize()
                    else:score_native(models[cid],family,x,records[cid]['case']['outputs'])
                if family=='tabpfn_auto':
                    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                begin=perf_counter()
                for request,task in enumerate(schedule):
                    cid=spec['cases'][task];x=inputs[cid][np.arange(batch)%len(inputs[cid])];start=perf_counter_ns()
                    if family=='tabpfn_auto':
                        raw=canonical_raw(models[cid].predict_proba(x),models[cid].classes_,records[cid]['case']['outputs'],probabilities=True)
                        torch.cuda.synchronize();control=controls[cid]
                    else:
                        raw=score_native(models[cid],family,x,records[cid]['case']['outputs']);control=quality_row(selected[cid,family]['evaluations'],'multiclass')['refit_control']
                    z=raw/control['temperature'];probability=np.clip(softmax(z,axis=1),1e-7,1-1e-7);labels=(z-control['shifts']).argmax(1)
                    ms=(perf_counter_ns()-start)/1e6;rows.append(dict(request=request,case=cid,ms=ms))
                    write(out/'latest_requests.json',dict(family=family,batch=batch,traffic=traffic,requests=rows))
                seconds=perf_counter()-begin
                timing=dict(family=family,batch=batch,traffic=traffic,concurrency=1,requests=rows,**summary([r['ms'] for r in rows]),
                    wall_seconds_including_checkpoint_io=seconds,samples_per_second=len(rows)*batch/seconds,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    hardware='A100 cached TabPFN8' if family=='tabpfn_auto' else 'CPU frozen native library threads2')
                timings.append(timing);write(out/'timings.json',timings);event('unified_external_serving_timing',**{k:v for k,v in timing.items() if k!='requests'})
        if family!='tabpfn_auto':
            for cid in spec['cases']:
                raw=score_native(models[cid],family,tests[cid],records[cid]['case']['outputs']);file=out/cid/family/'sealed_test_raw.npz';np.savez_compressed(file,raw=raw)
                test_predictions.append(dict(case=cid,family=family,path=str(file),sha256=digest(file),record=records[cid],
                    control=quality_row(selected[cid,family]['evaluations'],'multiclass')['refit_control']))
            write(out/'sealed_test_predictions.json',test_predictions);del models;gc.collect()
    result=dict(status='unified_external_serving_complete',timings=timings,storage=storage,context_spans=context_spans,initialization_peaks=initialization_peaks,fits=fits,curves=curves,tfm_points=tfm_points,prefix_memory=prefix_memory,
        test_predictions=test_predictions,seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),
        elapsed_seconds=perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        test_evaluated=False,nn_weight_updates=0,new_boosting_trees=0,limitations=[
            '32 requests do not estimate production tails reliably','CPU controls versus GPU TFM explicitly labelled',
            'serialized ordinary model bytes are not interchangeable with live TFM process memory',
            'one shared weight storage; each task context cache and preprocessing retained','old previously viewed test; later evaluation is exploratory'])
    write(out/'report.json',result);event('unified_external_serving_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
