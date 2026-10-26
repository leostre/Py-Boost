"""Preregistered common-trajectory and closed-loop H6 comparison."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,build_model,emit,read_dataset,sha256,write_report
from experiments.block_leaf_model import BlockLeafModel,load_prediction_model
from experiments.gradient_basis_core import GradientBasisState,gradient_signal,captured_energy
from experiments.online_leaf_study import raw as single_raw,summarize
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import probabilities,quality_guard
from experiments.uniform_threshold_core import ThresholdModel,threshold_variants

SOURCES = ('current','mean','history')


def raw(model,x):
    return model.raw(x) if isinstance(model,BlockLeafModel) else single_raw(model,x)


def task(model):
    return model.task if isinstance(model,BlockLeafModel) else model.base.task


def calibrate(model,x,y,calibration,selection,config):
    variants = threshold_variants(raw(model,x[calibration]),y[calibration],raw(model,x[selection]),
        y[selection],task(model),config['decision_grid'],config['decision_shrinkage'])
    choice = min(variants,key=lambda r:(-r['validation']['f1_macro'],r['validation']['loss'],r['threshold_fraction']))
    shifts = np.array(choice['decision_shifts'],np.float32)
    selected = model.with_shifts(shifts) if isinstance(model,BlockLeafModel) else replace(model,decision_shifts=shifts)
    return selected,choice,variants


def seal_model(method,model,directory,output,x,y,calibration,selection,validation,config,replay):
    selected,choice,variants = calibrate(model,x,y,calibration,selection,config)
    path = directory/f'{method}.npz'
    selected.save(path)
    loaded = load_prediction_model(path)
    actual = loaded.predict(x[validation])
    np.testing.assert_allclose(actual,selected.predict(x[validation]),rtol=0,atol=2e-14)
    replay[method],replay[method+'_labels'] = actual,loaded.predict_labels(x[validation])
    blocks = loaded.blocks if isinstance(loaded,BlockLeafModel) else (loaded,)
    return dict(method=method,artifact=path.relative_to(output).as_posix(),sha256=sha256(path),
        parameters=loaded.parameter_count,payload_bytes=loaded.payload_bytes,file_bytes=path.stat().st_size,
        resident_array_bytes=loaded.resident_array_bytes if isinstance(loaded,BlockLeafModel) else loaded.payload_bytes,
        rank=max(b.base.coefficients.shape[1] for b in blocks),blocks=len(blocks),
        reachable_leaves=sum(len(b.base.coefficients) for b in blocks),nnz=sum(len(b.values) for b in blocks),
        threshold_fraction=choice['threshold_fraction'],validation=choice['validation'],threshold_variants=variants)


def load_parent(name,seed,config,locks):
    root = Path(config['online_sources'][name])
    path = root/'report.json'
    if sha256(path) != config['online_report_hashes'][name]:
        raise RuntimeError('frozen OPT1 report changed')
    locks[str(path)] = sha256(path)
    report = json.loads(path.read_text())
    assert report['status'] == 'done' and report['scientific_fits'] == 20
    assert report['environment_sha256'] == config['expected_environment_sha256']
    for key in ('ntrees','lr','max_depth','lambda_l2','online_warmup','online_rank_fraction','online_residual_fraction',
                'decision_grid','decision_shrinkage'):
        assert report['config'][key] == config[key],key
    for file,expected in report['file_locks'].items():
        if sha256(file) != expected:
            raise RuntimeError('frozen OPT1 input changed: '+file)
        locks[file] = expected
    case = next(c for c in report['cases'] if c['seed'] == seed and c['dataset'] == name)
    directory = root/f'{name}_{seed}'
    assert sha256(directory/'splits.npz') == case['split_sha256']
    locks[str(directory/'splits.npz')] = case['split_sha256']
    with np.load(directory/'splits.npz',allow_pickle=False) as f:
        splits = {k:f[k].copy() for k in f.files}
    controls = {}
    for old,new in [('online_full','online_full'),('online_hybrid','fixed_hybrid'),('online_more_rank','fixed_more_rank')]:
        choice = next(c for c in case['choices'] if c['method'] == old)
        p = root/choice['artifact']
        if sha256(p) != choice['sha256']:
            raise RuntimeError('frozen OPT1 model changed')
        locks[str(p)] = choice['sha256']
        controls[new] = ThresholdModel.load(p)
    dense_path = directory/('dense_training_reference.npz' if case['task'] == 'multiclass' else 'online_full.npz')
    locks[str(dense_path)] = sha256(dense_path)
    dense = ThresholdModel.load(dense_path)
    assert dense.base.representation == 'dense'
    return case,splits,controls,dense


def verify_prefix(exported,fixed,count):
    first = exported.blocks[0]
    assert first.base.n_trees == count
    n,m = int(fixed.base.node_offsets[count]),int(fixed.base.leaf_offsets[count])
    np.testing.assert_array_equal(first.base.nodes,fixed.base.nodes[:n])
    np.testing.assert_array_equal(first.base.node_offsets,fixed.base.node_offsets[:count+1])
    np.testing.assert_array_equal(first.base.leaf_offsets,fixed.base.leaf_offsets[:count+1])
    np.testing.assert_array_equal(first.base.coefficients,fixed.base.coefficients[:m])
    np.testing.assert_array_equal(first.base.basis,fixed.base.basis)
    np.testing.assert_array_equal(exported.bias,fixed.base.bias)
    for key in ('rows','columns','values'):
        np.testing.assert_array_equal(getattr(first,key),getattr(fixed,key)[fixed.rows < m])


def common_trajectory(dense,fixed,x,y,config):
    """All estimators see identical G/H, never labels outside training."""
    import cupy as cp
    from py_boost.gpu.losses.losses import loss_alias
    started = time.perf_counter()
    loss = loss_alias['crossentropy' if task(dense) == 'multiclass' else 'logloss']
    target = cp.asarray(y[:,None] if y.ndim == 1 else y)
    target = loss.preprocess_input(target)
    scores = cp.asarray(np.broadcast_to(dense.base.bias,(len(x),len(dense.base.bias))).copy())
    leaves = dense.base.leaf_indices(x)
    estimators = {s:GradientBasisState(s,config['history_beta']) for s in SOURCES}
    current = {s:fixed.base.basis.copy() for s in SOURCES}
    events,rows,active = [],[],None
    period,obs = config['history_basis_period'],config['history_observation_period']
    for step in range(dense.base.n_trees+1):
        if step >= config['online_warmup'] and step%obs == 0:
            g,h = loss(target,scores)
            z = gradient_signal(g.get(),h.get(),config['lambda_l2'],task(dense))
            if active is not None and obs <= step-active <= period:
                for source in SOURCES:
                    rows.append(dict(source=source,origin_step=active,future_step=step,age=step-active,
                        captured_energy=captured_energy(z,current[source])))
            refresh = step >= 2*config['online_warmup'] and step < dense.base.n_trees and step%period == 0
            for source,state in estimators.items():
                if source != 'current' or refresh:
                    state.observe(z)
                if refresh:
                    current[source],diagnostic = state.estimate(task(dense),fixed.base.basis.shape[1],current[source])
                    events.append(dict(source=source,state_step=step,**diagnostic))
            if refresh:
                active = step
        if step < dense.base.n_trees:
            scores += cp.asarray(dense.base.coefficients[leaves[step]])
    replay_error = float(np.max(np.abs(probabilities(scores.get(),task(dense))-dense.predict(x))))
    if replay_error > 2e-5:
        raise RuntimeError('frozen dense trajectory replay differs')
    return dict(events=events,future_capture=rows,seconds=time.perf_counter()-started,
                native_float32_replay_probability_error=replay_error,scientific_fits=0)


def prepare_case(name,seed,config,output,dataset,locks):
    import cupy as cp
    from experiments.history_leaf_boosting import HistoryLeafBoosting
    x,y,source_indices,metadata = dataset
    parent,splits,controls,dense = load_parent(name,seed,config,locks)
    np.testing.assert_array_equal(splits['source_indices'],source_indices)
    train,validation,calibration,selection = [splits[k] for k in ('train','validation','calibration','selection')]
    directory = output/f'{name}_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    np.savez(directory/'splits.npz',**splits)
    common = common_trajectory(dense,controls['fixed_hybrid'],x[train],y[train],config)
    write_report(directory/'common_trajectory.json',common)
    emit('history_common_trajectory_complete',dataset=name,seed=seed,rows=len(common['future_capture']),
         seconds=common['seconds'],replay_error=common['native_float32_replay_probability_error'])
    del dense
    models,records = dict(controls),[]
    order = np.random.default_rng(seed+619).permutation(SOURCES).tolist()
    for source in order:
        native = build_model('full',metadata['task'],len(controls['online_full'].base.bias),config,
                             seed,config['max_depth'],config['lambda_l2'])
        model = HistoryLeafBoosting(**native.params,basis_source=source,task=metadata['task'],
            warmup=config['online_warmup'],rank_fraction=config['online_rank_fraction'],
            residual_fraction=config['online_residual_fraction'],beta=config['history_beta'],
            basis_period=config['history_basis_period'],observation_period=config['history_observation_period'])
        del native
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        model.fit(x[train],y[train],eval_sets=[dict(X=x[validation],y=y[validation])])
        cp.cuda.get_current_stream().synchronize()
        seconds = time.perf_counter()-started
        deployed = model.export()
        assert model.warm_fingerprint == parent['common_warm_fingerprint']
        verify_prefix(deployed,controls['fixed_hybrid'],2*config['online_warmup'])
        replay_error = float(np.max(np.abs(deployed.predict(x[validation])-
            probabilities(model.validation_final_raw[0],metadata['task']))))
        assert replay_error <= 2e-5 and deployed.n_trees == config['ntrees']
        record = dict(source=source,fit_seconds=seconds,observer_seconds=model.observer_seconds,
            training_pool_peak_bytes=model.training_pool_peak,stage_ms=model.stage_ms,trace=model.trace,
            observer_trace=model.observer_trace,basis_events=model.basis_events,
            warm_fingerprint=model.warm_fingerprint,prefix_verified=True,
            training_export_max_probability_error=replay_error,
            parameters=deployed.parameter_count,payload_bytes=deployed.payload_bytes,blocks=len(deployed.blocks))
        write_report(directory/f'fit_{source}.json',record)
        records.append(record)
        models[source] = deployed
        emit('history_fit_complete',dataset=name,seed=seed,source=source,seconds=seconds,
             observer_seconds=model.observer_seconds,blocks=record['blocks'],replay_error=replay_error)
        del model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    choices,replay = [],dict(features=x[validation])
    for method,model in models.items():
        choices.append(seal_model(method,model,directory,output,x,y,calibration,selection,validation,config,replay))
    mapping = dict(online_full='online_full',fixed_hybrid='online_hybrid',fixed_more_rank='online_more_rank')
    for new,old in mapping.items():
        chosen = next(r for r in choices if r['method'] == new)
        previous = next(r for r in parent['choices'] if r['method'] == old)
        assert chosen['threshold_fraction'] == previous['threshold_fraction']
        assert chosen['validation'] == previous['validation']
    np.savez(directory/'replay.npz',**replay)
    result = dict(dataset=name,seed=seed,task=metadata['task'],data_sha256=metadata['sha256'],
        common_warm_fingerprint=parent['common_warm_fingerprint'],choices=choices,fits=records,fit_order=order,
        split_sha256=sha256(directory/'splits.npz'),parent_report_sha256=config['online_report_hashes'][name])
    write_report(directory/'selection.json',result)
    subprocess.run([sys.executable,'-m','experiments.history_basis_verify',str(directory)],cwd=ROOT,check=True,timeout=180)
    return result


def evaluate_case(case,output,x,y):
    directory = output/f"{case['dataset']}_{case['seed']}"
    with np.load(directory/'splits.npz',allow_pickle=False) as f:
        test = f['test'].copy()
    full = next(r for r in case['choices'] if r['method'] == 'online_full')
    full_model = load_prediction_model(output/full['artifact'])
    reference = calibrated_quality(y[test],raw(full_model,x[test]),case['task'],full_model.decision_shifts)
    rows = []
    for choice in case['choices']:
        path = output/choice['artifact']
        if sha256(path) != choice['sha256']:
            raise RuntimeError('sealed model changed')
        model = load_prediction_model(path)
        scores = raw(model,x[test])
        q = calibrated_quality(y[test],scores,case['task'],model.decision_shifts)
        plain = calibrated_quality(y[test],scores,case['task'],np.empty(0,np.float32))
        rows.append(dict(dataset=case['dataset'],seed=case['seed'],test=q,test_uncalibrated=plain,
            test_reference=reference,quality_pass=quality_guard(q,reference),
            validation_quality_pass=quality_guard(choice['validation'],full['validation']),
            parameter_ratio=choice['parameters']/full['parameters'],payload_ratio=choice['payload_bytes']/full['payload_bytes'],
            loss_ratio=q['loss']/reference['loss'],macro_retained=q['f1_macro']/reference['f1_macro'],
            weighted_retained=q['f1_weighted']/reference['f1_weighted'],**choice))
    write_report(directory/'test.json',rows)
    return rows


def history_basis_study(config,report,output):
    import cupy as cp
    from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver
    started = time.perf_counter()
    def expired(signum,frame):
        raise TimeoutError('history study walltime exhausted')
    old_handler = signal.signal(signal.SIGALRM,expired)
    signal.alarm(config['walltime_seconds'])
    ready,reason,probe = gpu_probe('shared',config['inference_gpu_minimum_free_mib'])
    if not ready:
        raise RuntimeError(reason)
    cp.cuda.runtime.free(0)
    guard = SharedGpuObserver()
    cases,records,locks = [],[],{}
    report.update(status='running',scientific_fits=0,test_selection=False)
    try:
        with guard,threadpool_limits(limits=1):
            subprocess.run([sys.executable,'-m','pytest','-q','tests/test_gradient_basis_core.py',
                'tests/test_block_leaf_model.py','tests/test_history_study.py'],cwd=ROOT,check=True,timeout=300)
            datasets = {n:read_dataset(n,config) for n in config['datasets']}
            emit('history_protocol_sealed',datasets=config['datasets'],sources=list(SOURCES),scientific_fits=45,
                 test_evaluation='after_all_selections_sealed')
            for name in config['datasets']:
                for seed in config['seeds_by_dataset'][name]:
                    case = prepare_case(name,seed,config,output,datasets[name],locks)
                    cases.append(case)
                    report.update(cases=cases,scientific_fits=3*len(cases),elapsed_seconds=time.perf_counter()-started)
                    write_report(output/'progress.json',report)
                    emit('history_selection_case_complete',dataset=name,seed=seed,fits=3*len(cases))
            write_report(output/'selection_manifest.json',dict(cases=cases,file_locks=locks))
            sealed = sha256(output/'selection_manifest.json')
            emit('history_all_selections_sealed',sha256=sealed,cases=len(cases))
            for case in cases:
                x,y,_,_ = datasets[case['dataset']]
                records.extend(evaluate_case(case,output,x,y))
                emit('history_test_case_complete',dataset=case['dataset'],seed=case['seed'])
            assert sha256(output/'selection_manifest.json') == sealed
            for path,expected in locks.items():
                assert sha256(path) == expected,path
        summaries = summarize(records)
        report.update(status='done',scientific_fits=3*len(cases),cases=cases,records=records,summaries=summaries,
            file_locks=locks,selection_sha256=sealed,gpu_probe=probe,gpu_guard_samples=guard.samples,
            elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        emit('history_study_complete',scientific_fits=report['scientific_fits'],cases=len(cases),
             summaries=summaries,elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',cases=cases,
            scientific_fits=3*len(cases),gpu_probe=probe,gpu_guard_samples=guard.samples)
        write_report(output/'report.json',report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_handler)
