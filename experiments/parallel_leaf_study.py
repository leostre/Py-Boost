"""Bounded parallel-increment + common-subspace comparison on old splits."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,emit,read_dataset,sha256,write_report
from experiments.history_basis_study import load_parent,raw,seal_model,evaluate_case
from experiments.online_leaf_study import summarize
from experiments.parallel_leaf_core import prefix,average_increments,recompress
from experiments.parallel_leaf_worker import fit_increment
from experiments.uniform_threshold_core import ThresholdModel
from experiments.output_calibration_core import calibrated_quality

VARIANTS = ('serial','serial_half_step','parallel_dense','parallel_compact')


def worker_warmup():
    # Spawn, not fork: each process owns an isolated CUDA RNG and runtime.
    from experiments.continuation_leaf_boosting import ContinuationLeafBoosting
    import cupy as cp
    cp.cuda.runtime.free(0)


def execute_requests(requests,pool,minimum_parallel_free_mib=8192):
    from experiments.inference_gpu_guard import gpu_probe
    ready,reason,probe = gpu_probe('shared',minimum_parallel_free_mib)
    concurrent = len(requests) == 2 and ready
    started = time.perf_counter()
    if concurrent:
        futures = [pool.submit(fit_increment,r) for r in requests]
        records = [f.result(timeout=600) for f in futures]
    else:
        records = [pool.submit(fit_increment,r).result(timeout=600) for r in requests]
    overlap = max(0,min(r['fit_finished_ns'] for r in records)-max(r['fit_started_ns'] for r in records)) if len(records)==2 else 0
    return records,dict(execution='concurrent' if concurrent else 'sequential',
        fallback_reason=reason if len(requests)==2 and not ready else None,capacity_probe=probe,
        wall_seconds=time.perf_counter()-started,fit_overlap_seconds=overlap/1e9,
        sum_worker_fit_seconds=sum(r['seconds'] for r in records),
        sum_individual_pool_peaks=sum(r['training_pool_peak_bytes'] for r in records))


def train_variant(variant,initial,x,y,splits,config,directory,pool,seed):
    checkpoint = initial
    cycles,fit_jobs = [],0
    started = time.perf_counter()
    initial.save(directory/'checkpoint_start.npz')
    train,validation = splits['train'],splits['validation']
    rank = initial.base.basis.shape[1]
    for cycle in range(4):
        cycle_dir = directory/f'cycle{cycle}'
        cycle_dir.mkdir()
        initial_path = cycle_dir/'initial.npz'
        np.savez(initial_path,train=raw(checkpoint,x[train]).astype(np.float32),
                 validation=raw(checkpoint,x[validation]))
        parallel = variant.startswith('parallel')
        requests = [dict(data_path=str(directory.parent/'worker_data.npz'),initial_path=str(initial_path),
            config=config,task=initial.base.task,seed=seed+80000+100*cycle+j,trees=20 if parallel else 40,
            lr=.05 if variant=='serial_half_step' else .1,subsample=.8,colsample=.8,
            output=str(cycle_dir/f'increment{j}.npz'),minimum_free_mib=2048) for j in range(2 if parallel else 1)]
        records,execution = execute_requests(requests,pool)
        increments = [ThresholdModel.load(r['output']) for r in requests]
        before_checkpoint = checkpoint
        merged = average_increments(checkpoint,increments)
        expected_trees = 40+(cycle+1)*40
        assert merged.n_trees == expected_trees
        before = calibrated_quality(y[train],raw(merged,x[train]),initial.base.task,np.empty(0,np.float32))
        merge_started = time.perf_counter()
        if variant != 'parallel_dense':
            candidates = [average_increments(before_checkpoint,[a]) for a in increments]
            checkpoint,compression = recompress(merged,candidates,x[train],rank,config['online_residual_fraction'])
        else:
            checkpoint,compression = merged,None
        merge_seconds = time.perf_counter()-merge_started
        after = calibrated_quality(y[train],raw(checkpoint,x[train]),initial.base.task,np.empty(0,np.float32))
        path = cycle_dir/'checkpoint.npz'
        checkpoint.save(path)
        row = dict(cycle=cycle,built_trees=expected_trees,workers=records,execution=execution,
            compression=compression,merge_seconds=merge_seconds,training_before_merge=before,training_after_merge=after,
            parameters=checkpoint.parameter_count,payload_bytes=checkpoint.payload_bytes,
            checkpoint_sha256=sha256(path),initial_sha256=sha256(initial_path),
            initial_payload_bytes=initial_path.stat().st_size)
        cycles.append(row)
        fit_jobs += len(records)
        write_report(cycle_dir/'cycle.json',row)
        emit('parallel_cycle_complete',variant=variant,seed=seed,cycle=cycle,
            trees=expected_trees,execution=execution['execution'],fit_overlap=execution['fit_overlap_seconds'],
            payload_bytes=checkpoint.payload_bytes,training_loss_before=before['loss'],training_loss_after=after['loss'])
    record = dict(variant=variant,cycles=cycles,native_fit_jobs=fit_jobs,new_built_trees=160,
        prefix_trees=40,total_built_trees=200,seconds=time.perf_counter()-started,
        worker_fit_seconds=sum(c['execution']['sum_worker_fit_seconds'] for c in cycles),
        worker_wall_seconds=sum(c['execution']['wall_seconds'] for c in cycles),
        merge_seconds=sum(c['merge_seconds'] for c in cycles))
    write_report(directory/'fit.json',record)
    return checkpoint,record


def prepare_parallel_case(name,seed,config,output,dataset,locks,pool):
    x,y,source_indices,metadata = dataset
    parent,splits,controls,dense = load_parent(name,seed,config,locks)
    del dense
    np.testing.assert_array_equal(source_indices,splits['source_indices'])
    directory = output/f'{name}_{seed}'
    directory.mkdir()
    np.savez(directory/'splits.npz',**splits)
    np.savez(directory/'worker_data.npz',train_x=x[splits['train']],train_y=y[splits['train']],
             validation_x=x[splits['validation']],validation_y=y[splits['validation']])
    initial = prefix(controls['fixed_hybrid'],40)
    initial.save(directory/'common_prefix.npz')
    order = np.random.default_rng(seed+3197).permutation(VARIANTS).tolist()
    models,records = dict(controls),[]
    for variant in order:
        variant_dir = directory/variant
        variant_dir.mkdir()
        deployed,record = train_variant(variant,initial,x,y,splits,config,variant_dir,pool,seed)
        records.append(record)
        models[variant] = deployed
        emit('parallel_variant_complete',dataset=name,seed=seed,variant=variant,seconds=record['seconds'],
             native_fit_jobs=record['native_fit_jobs'],new_trees=record['new_built_trees'])
    # First-cycle branch0 exactly equals the first20 serial full-step trees:
    # same checkpoint, data, seed, LR and subsampling, before any synchronization.
    serial = ThresholdModel.load(directory/'serial/cycle0/increment0.npz')
    branch = ThresholdModel.load(directory/'parallel_compact/cycle0/increment0.npz')
    first20 = prefix(serial,20)
    np.testing.assert_array_equal(first20.base.nodes,branch.base.nodes)
    np.testing.assert_array_equal(first20.base.coefficients,branch.base.coefficients)
    choices,replay = [],dict(features=x[splits['validation']])
    for method,model in models.items():
        choices.append(seal_model(method,model,directory,output,x,y,splits['calibration'],splits['selection'],
                                  splits['validation'],config,replay))
    mapping = dict(online_full='online_full',fixed_hybrid='online_hybrid',fixed_more_rank='online_more_rank')
    for new,old in mapping.items():
        chosen = next(r for r in choices if r['method']==new)
        previous = next(r for r in parent['choices'] if r['method']==old)
        assert chosen['threshold_fraction'] == previous['threshold_fraction'] and chosen['validation'] == previous['validation']
    np.savez(directory/'replay.npz',**replay)
    result = dict(dataset=name,seed=seed,task=metadata['task'],data_sha256=metadata['sha256'],choices=choices,
        fits=records,fit_order=order,split_sha256=sha256(directory/'splits.npz'),
        common_prefix_sha256=sha256(directory/'common_prefix.npz'),first20_native_match=True,
        parent_report_sha256=config['online_report_hashes'][name],new_built_trees=640,native_fit_jobs=24)
    write_report(directory/'selection.json',result)
    subprocess.run([sys.executable,'-m','experiments.history_basis_verify',str(directory)],cwd=ROOT,check=True,timeout=180)
    return result


def parallel_leaf_study(config,report,output):
    from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver
    started = time.perf_counter()
    def expired(signum,frame):
        raise TimeoutError('parallel study walltime exhausted')
    previous = signal.signal(signal.SIGALRM,expired)
    signal.alarm(config['walltime_seconds'])
    ready,reason,probe = gpu_probe('shared',8192)
    if not ready:
        raise RuntimeError(reason)
    guard = SharedGpuObserver()
    cases,records,locks = [],[],{}
    report.update(status='running',scientific_new_trees=0,scientific_fit_jobs=0,test_selection=False)
    try:
        with guard,threadpool_limits(limits=1):
            subprocess.run([sys.executable,'-m','pytest','-q','tests/test_parallel_leaf_core.py',
                'tests/test_parallel_study.py'],cwd=ROOT,check=True,timeout=300)
            datasets = {n:read_dataset(n,config) for n in config['datasets']}
            emit('parallel_protocol_sealed',datasets=config['datasets'],variants=list(VARIANTS),new_trees=9600,
                 native_fit_jobs=360,test_evaluation='after_all_selections_sealed')
            with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=worker_warmup) as pool:
                for name in config['datasets']:
                    for seed in config['seeds_by_dataset'][name]:
                        case = prepare_parallel_case(name,seed,config,output,datasets[name],locks,pool)
                        cases.append(case)
                        report.update(cases=cases,scientific_new_trees=640*len(cases),scientific_fit_jobs=24*len(cases),
                                      elapsed_seconds=time.perf_counter()-started)
                        write_report(output/'progress.json',report)
            write_report(output/'selection_manifest.json',dict(cases=cases,file_locks=locks))
            sealed = sha256(output/'selection_manifest.json')
            emit('parallel_all_selections_sealed',sha256=sealed,cases=len(cases))
            for case in cases:
                x,y,_,_ = datasets[case['dataset']]
                records.extend(evaluate_case(case,output,x,y))
                emit('parallel_test_case_complete',dataset=case['dataset'],seed=case['seed'])
            assert sha256(output/'selection_manifest.json') == sealed
            for path,expected in locks.items():
                assert sha256(path) == expected,path
        summaries = summarize(records)
        report.update(status='done',cases=cases,records=records,summaries=summaries,file_locks=locks,
            selection_sha256=sealed,scientific_new_trees=640*len(cases),scientific_fit_jobs=24*len(cases),
            gpu_probe=probe,gpu_guard_samples=guard.samples,elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        emit('parallel_study_complete',cases=len(cases),new_trees=report['scientific_new_trees'],
             fit_jobs=report['scientific_fit_jobs'],summaries=summaries,elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',cases=cases,error=f'{type(error).__name__}: {error}',
            gpu_probe=probe,gpu_guard_samples=guard.samples)
        write_report(output/'report.json',report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
