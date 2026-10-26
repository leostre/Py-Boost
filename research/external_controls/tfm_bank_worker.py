"""Five real contexts, one shared checkpoint, transfer-inclusive J1 serving."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
from time import perf_counter,perf_counter_ns
import numpy as np
from external_quality_core import canonical_raw
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import quality_guard
from experiments.shared_leaf_basis import probabilities
from experiments.model_bank_core import latency_summary
from quality_worker import load_inputs,gpu_guard
from matched_data import digest
from sharing_worker import rss
from storage_graph import inventory,numerical_spans,union_bytes
from tfm_serving_core import atomic_write as write,request_plan,worker_plan,gpu_owner,interrupted_result


def gpu_processes():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return {int(s.split(',')[0]):int(s.split(',')[1])*1024**2 for s in out.splitlines() if s.strip() and 'N/A' not in s}


class RssPeak:
    def __enter__(self):
        self.peak=rss()
        self.stop=threading.Event()
        def poll():
            while not self.stop.wait(.005): self.peak=max(self.peak,rss())
        self.thread=threading.Thread(target=poll,daemon=True)
        self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak=max(self.peak,rss())


def shared_spec(weights,torch):
    from tabpfn.base import ModelSpecs,initialize_tabpfn_model
    # Deliberately NO n_estimators_override=1: auto must retain checkpoint8.
    models,configs,_,inference=initialize_tabpfn_model(weights,'classifier',devices=[torch.device('cuda')])
    if len(models)!=1 or inference.N_ESTIMATORS!=8:
        raise RuntimeError('unreviewed shared checkpoint/default ensemble')
    return ModelSpecs(models[0],configs[0],inference)


def child(cfg,mode,family,output):
    before_gpu=gpu_processes()
    available=gpu_guard(cfg)
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    root=Path(cfg['environment_root'])
    weights=root/'weights'/cfg['weights_revision']/cfg['weights_filename']
    if digest(weights)!=cfg['weights_sha256']: raise RuntimeError('shared weights changed')
    dp=Path(cfg['quality_development_manifest'])
    if digest(dp)!=cfg['quality_development_sha256']: raise RuntimeError('TFM development changed')
    development=json.loads(dp.read_text(encoding='utf-8'))
    models,inputs,probes,expected_probes,fits,metrics=[],{},[],[],[],[]
    paired_before,validation_sets,source_qualities,decision_shifts,cases,entries=[],[],[],[],[],[]
    own_pid=None
    ownership=dict(pid=None,verified=False,method='before_actual_inference')
    pending=None
    checkpoint_metadata={}
    started=perf_counter()
    stage='initialize'
    def checkpoint(**extra):
        checkpoint_metadata.update(extra)
        value=dict(status='running',mode=mode,family=family,stage=stage,tasks=cfg['quality_cases'],
            fits=fits,metrics=metrics,pending_trace=pending,context_fit_calls=len(fits),
            gpu_host_pid=own_pid,gpu_ownership=ownership,gpu_ownership_verified=ownership['verified'],
            nn_weight_updates=0,new_boosting_trees=0,raw_request_timings_retained=True,
            elapsed_seconds=perf_counter()-started,**checkpoint_metadata)
        write(output,value)
    checkpoint()
    try:
        spec=shared_spec(weights,torch)
        initialize_seconds=perf_counter()-started
        for case_id in cfg['quality_cases']:
            stage='prepare_'+case_id
            case,entry,data=load_inputs(cfg,case_id)
            source=next(r for r in development['records'] if r['case_id']==case_id and r['family']==family)
            model=TabPFNClassifier(model_path=spec,n_estimators='auto' if family=='tabpfn_auto' else 1,
                device='cuda',inference_precision='auto',fit_mode=mode,memory_saving_mode='auto',
                enable_torch_compile=False,random_state=case['seed'],n_preprocessing_jobs=2,show_progress_bar=False)
            begin=perf_counter()
            model.fit(*data['train'])
            torch.cuda.synchronize()
            fits.append(dict(case_id=case_id,seconds=perf_counter()-begin,
                actual_n_estimators=int(model.n_estimators_),nn_weight_updates=0))
            if model.n_estimators_!=source['fits'][0]['actual_n_estimators']:
                raise RuntimeError('shared bank silently changed number of estimators')
            models.append(model)
            probes.append(data['selection'][0][:16].copy())
            paired_before.append(model.predict_proba(probes[-1]))
            torch.cuda.synchronize()
            if own_pid is None:
                namespace_pids={os.getpid()}
                for line in Path('/proc/self/status').read_text().splitlines():
                    if line.startswith('NSpid:'): namespace_pids.update(map(int,line.split()[1:]))
                ownership=gpu_owner(before_gpu,gpu_processes(),namespace_pids)
                own_pid=ownership['pid']
            validation_sets.append((data['selection'][0].copy(),data['selection'][1].copy()))
            source_qualities.append(source['chosen']['validation'])
            decision_shifts.append(np.array(source['chosen']['decision_shifts'],np.float32))
            cases.append(case)
            entries.append(entry)
            saved=source['development_predictions'][source['chosen']['candidate_id']]
            if digest(saved['artifact'])!=saved['sha256']: raise RuntimeError('probe predictions changed')
            with np.load(saved['artifact']) as f: expected_probes.append(probabilities(f['selection'][:16],'multiclass'))
            with np.load(entry['path']) as f:
                prefix=np.ascontiguousarray(f['features'][:1024],dtype=np.float32).copy()
            for batch in cfg['bank_batches']: inputs[case_id,batch]=prefix[np.arange(batch)%len(prefix)].copy()
            checkpoint()
            print(json.dumps(dict(event='tfm_context_ready',mode=mode,family=family,case_id=case_id,
                actual_n_estimators=fits[-1]['actual_n_estimators'],fit_seconds=fits[-1]['seconds'],
                gpu_ownership=ownership)),flush=True)
            del data,model
            gc.collect()
        if len({id(m.models_[0]) for m in models})!=1: raise RuntimeError('bank weights not shared by reference')
        spans=[s for m in models for s in numerical_spans(m.models_)]
        one=union_bytes(numerical_spans(models[0].models_))
        if union_bytes(spans)!=one: raise RuntimeError('bank checkpoint storage duplicated')
        errors,cross_batch_errors,validation_qualities,isolation_passes=[],[],[],[]
        stage='validate_isolation'
        for index,(m,x,expected,before) in enumerate(zip(models,probes,expected_probes,paired_before)):
            actual=m.predict_proba(x)
            isolation_passes.append(bool(np.allclose(actual,before,atol=2e-5,rtol=2e-4)))
            errors.append(float(np.max(np.abs(actual-before))))
            cross_batch_errors.append(float(np.max(np.abs(actual-expected))))
            vx,vy=validation_sets[index]
            full=m.predict_proba(vx)
            z=canonical_raw(full,m.classes_,cases[index]['outputs'],probabilities=True)
            q=calibrated_quality(vy,z,'multiclass',decision_shifts[index])
            validation_qualities.append(dict(case_id=cfg['quality_cases'][index],quality=q,
                reference=source_qualities[index],within_original_five_percent=quality_guard(q,source_qualities[index])))
        validation_path=output.with_name(output.stem+'-pretest.json')
        write(validation_path,dict(paired_same_batch_isolation_passes=isolation_passes,
            paired_same_batch_max_errors=errors,cross_batch_16_vs_full_reference_errors=cross_batch_errors,
            full_selection_quality=validation_qualities,test_evaluated=False,model_settings_unchanged=True))
        checkpoint(pretest_validation_sha256=digest(validation_path),validation_qualities=validation_qualities,
                   paired_probe_max_abs_error=max(errors))
        if not all(isolation_passes) or not all(q['within_original_five_percent'] for q in validation_qualities):
            write(output,dict(status='prediction_contract_failed',mode=mode,family=family,fits=fits,
                diagnostic_file=str(validation_path),diagnostic_sha256=digest(validation_path),
                timing_not_measured=True,nn_weight_updates=0,no_automatic_weaker_fallback=True))
            print(json.dumps(dict(event='tfm_bank_prediction_contract_failed',mode=mode,family=family,
                paired_isolation_passes=isolation_passes,full_quality_passes=[q['within_original_five_percent'] for q in validation_qualities])),flush=True)
            return
        del validation_sets,paired_before,expected_probes,probes,full,actual,before
        # Include the serving decision arrays; don't mistake model-only inventory
        # for the full process or omit caller-owned thresholds.
        state=inventory([models,decision_shifts])
        gc.collect()
        torch.cuda.empty_cache()
        resident_rss=rss()
        initialization_peaks=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        checkpoint(reachable_state=state,resident_rss_bytes=resident_rss,
                   initialization_seconds=initialize_seconds,shared_weight_bytes=one,
                   initialization_torch_peaks=initialization_peaks,
                   validation_qualities=validation_qualities,pretest_validation_sha256=digest(validation_path))
        for batch in cfg['bank_batches']:
            stage='serve_batch_'+str(batch)
            with ThreadPoolExecutor(max_workers=1) as executor:
                def request(index,submitted):
                    value=models[index].predict_proba(inputs[cfg['quality_cases'][index],batch])
                    torch.cuda.synchronize()
                    if not np.isfinite(value).all(): raise RuntimeError('nonfinite shared bank response')
                    return index,(perf_counter_ns()-submitted)/1e6
                for index in range(len(models)):
                    executor.submit(request,index,perf_counter_ns()).result()
                for traffic in ('uniform','skew'):
                    schedule=request_plan(len(models),cfg['bank_requests'],traffic)
                    samples=[]
                    pending=dict(batch=batch,traffic=traffic,planned_requests=len(schedule),request_samples_ms=[])
                    checkpoint()
                    with RssPeak() as peak:
                        start=perf_counter()
                        for index in schedule:
                            samples.append(executor.submit(request,index,perf_counter_ns()).result())
                            pending['request_samples_ms'].append(dict(task_index=samples[-1][0],elapsed_ms=samples[-1][1]))
                            # Outside request latency; wall throughput includes this I/O.
                            checkpoint()
                        seconds=perf_counter()-start
                    driver=gpu_processes().get(own_pid) if own_pid is not None else None
                    metric=dict(batch=batch,jobs=1,traffic=traffic,**latency_summary([t for _,t in samples]),
                        wall_seconds=seconds,wall_qps=len(samples)/seconds,objects_per_second=len(samples)*batch/seconds,
                        request_samples_ms=[dict(task_index=i,elapsed_ms=t) for i,t in samples],
                        per_task={case_id:latency_summary([t for i,t in samples if i==j]) for j,case_id in enumerate(cfg['quality_cases'])
                                  if any(i==j for i,_ in samples)},
                        rss_peak_bytes=peak.peak,gpu_process_bytes=driver,
                        gpu_ownership=ownership,wall_includes_checkpoint_io=True,
                        cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    metrics.append(metric)
                    pending=None
                    checkpoint(reachable_state=state,resident_rss_bytes=resident_rss,
                        validation_qualities=validation_qualities,pretest_validation_sha256=digest(validation_path))
                    print(json.dumps(dict(event='tfm_bank_measurement',mode=mode,family=family,batch=batch,traffic=traffic,
                        p95_ms=metric['p95_ms'],rss_peak_bytes=metric['rss_peak_bytes'],gpu_process_bytes=driver)),flush=True)
        quality=[]
        stage='frozen_test_after_traffic'
        for index,(model,case,entry) in enumerate(zip(models,cases,entries)):
            with np.load(case['split_file']) as f: indices=f['test'].copy()
            with np.load(entry['path']) as f:
                test_x,test_y=f['features'][indices].copy(),f['target'][indices].copy()
            z=canonical_raw(model.predict_proba(test_x),model.classes_,case['outputs'],probabilities=True)
            q=calibrated_quality(test_y,z,'multiclass',decision_shifts[index])
            quality.append(dict(case_id=cfg['quality_cases'][index],quality=q))
            checkpoint(quality=quality)
        result=dict(status='complete',mode=mode,family=family,tasks=cfg['quality_cases'],fits=fits,metrics=metrics,quality=quality,
            shared_weight_bytes=one,independent_weight_sum_bytes=sum(union_bytes(numerical_spans(m.models_)) for m in models),
            reachable_state=state,resident_rss_bytes=resident_rss,paired_probe_max_abs_error=max(errors),
            cross_batch_reference_max_abs_error=max(cross_batch_errors),validation_qualities=validation_qualities,
            pretest_validation_sha256=digest(validation_path),
            initialization_seconds=initialize_seconds,initialization_torch_peaks=initialization_peaks,
            gpu_host_pid=own_pid,gpu_ownership_verified=ownership['verified'],gpu_ownership=ownership,available_before_bytes=available,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            context_fit_calls=len(fits),nn_weight_updates=0,new_boosting_trees=0,
            test_labels_loaded=True,test_is_reused_exploratory=True,shared_a100=True,
            latency_mode='ready resident contexts, J1, host/device transfers included')
    except torch.OutOfMemoryError as error:
        # Predeclared feasibility outcome under the SAME quota; never reduce
        # estimators or drop contexts and present the retry as the original bank.
        result=dict(status='resource_infeasible',mode=mode,family=family,stage=stage,
            completed_contexts=len(models),fits=fits,metrics=metrics,context_fit_calls=len(fits),
            pending_trace=pending,gpu_ownership=ownership,
            nn_weight_updates=0,error_type=type(error).__name__,error=str(error),
            torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            quota_fraction=cfg['torch_memory_fraction'],no_automatic_weaker_fallback=True)
    except Exception as error:
        checkpoint(error_type=type(error).__name__,error=str(error))
        raise
    write(output,result)
    print(json.dumps(dict(event='tfm_bank_worker_complete',mode=mode,family=family,status=result['status'])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['fit_preprocessors','fit_with_cache'])
    parser.add_argument('--family',choices=['tabpfn_auto','tabpfn_1'])
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.mode:
        child(cfg,args.mode,args.family,args.output)
        return
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_tfm_bank_plan.py','tests/test_tfm_serving_core.py','tests/test_storage_graph.py'],check=True,timeout=300)
    write(args.output/'traffic_seal.json',dict(config=cfg,test_labels_read=False,nn_weight_updates=0))
    workers=[]
    plan=worker_plan()
    for mode,family in plan:
        path=args.output/(mode+'_'+family+'.json')
        print(json.dumps(dict(event='tfm_bank_worker_start',mode=mode,family=family,requests=cfg['bank_requests'])),flush=True)
        command=[sys.executable,__file__,str(args.config),str(path),'--mode',mode,'--family',family]
        # An isolated process group lets a timed-out mode release ONLY its own
        # descendants before the next mode; never touch foreign CUDA contexts.
        process=subprocess.Popen(command,start_new_session=True)
        try:
            code=process.wait(timeout=cfg['worker_timeout_seconds'])
            result=json.loads(path.read_text(encoding='utf-8')) if code==0 else interrupted_result(
                path,mode,family,'worker_failed',returncode=code)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=10)
            # Descendants may outlive their parent; this PGID was created above
            # exclusively for this worker, never for any foreign GPU job.
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            result=interrupted_result(path,mode,family,'timeout',timeout_seconds=cfg['worker_timeout_seconds'])
        write(path,result)
        workers.append(result)
        write(args.output/'report.json',dict(status='running' if len(workers)<len(plan) else 'finished',workers=workers,
            worker_plan=plan,raw_request_timings_retained=True,nn_weight_updates=0,new_boosting_trees=0))
        print(json.dumps(dict(event='tfm_mode_finished',mode=mode,family=family,status=result['status'],
            complete_traces=len(result.get('metrics',[])),partial_requests=len((result.get('pending_trace') or {}).get('request_samples_ms',[])))),flush=True)
    write(args.output/'report.json',dict(status='complete' if all(w['status']=='complete' for w in workers) else 'partial',workers=workers,
        nn_weight_updates=0,new_boosting_trees=0,shared_a100=True,raw_request_timings_retained=True,
        interpretation='five distinct tasks; context fits priced, no NN weight updates; shared strong default not5weightcopies',
        source_config_sha256=digest(args.config),requests_per_trace=cfg['bank_requests'],original_trace_requests=128,
        protocol='first32ofold128, cachedauto/cached1/preprocessorsauto/preprocessors1, same5contexts/default8/5percent',
        timing_limitation='32requests per trace, few samples in each task tail, sharedGPU, not SLA certification; wall includes durable checkpoint I/O'))
    print(json.dumps(dict(event='tfm_bank_complete',workers=len(workers),nn_weight_updates=0)),flush=True)


if __name__=='__main__': main()
