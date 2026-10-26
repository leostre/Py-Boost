"""Fixed-basis training feedback vs same-basis posthoc on sealed old splits."""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT,build_model,emit,read_dataset,sha256,write_report
from experiments.online_leaf_core import project_part,stitch,residual_rank_allowance
from experiments.shared_leaf_basis import probabilities,quality_guard,reference_class_model
from experiments.sparse_leaf_core import leaf_design
from experiments.uniform_threshold_core import ThresholdModel,threshold_variants
from experiments.output_calibration_core import calibrated_quality

MODES = ('full','rank','hybrid','more_rank')


def raw(model,x):
    return model.raw_from_design(leaf_design(model.base.leaf_indices(x),len(model.base.coefficients)))


def posthoc(full,features,mode,config):
    basis = full['basis']
    rank = max(1,int(np.floor(basis.shape[1]*config['online_rank_fraction'])))
    if mode == 'more_rank':
        rank = min(basis.shape[1],rank+residual_rank_allowance(config['online_residual_fraction'],basis.shape[0]))
    b = basis[:,:rank].copy()
    fraction = config['online_residual_fraction'] if mode == 'hybrid' else 0.
    parts = [project_part(p,w,b,fraction) for p,w in zip(full['parts'],full['weights'])]
    prefix = parts[:config['online_warmup']]
    prefix_model = stitch(prefix,b,full['bias'],features.shape[1],full['task'])
    leaves = prefix_model.base.leaf_indices(features)
    new = np.zeros((len(features),len(full['bias'])),float)
    offset = 0
    for p,indices in zip(prefix,leaves):
        new += p.decoded(b)[indices-offset]
        offset += len(p.coefficients)
    bias = (full['bias']+full['warm_mean']-new.mean(axis=0)).astype(np.float32)
    return stitch(parts,b,bias,features.shape[1],full['task'])


def choose_threshold(model,x,y,calibration,selection,config):
    variants = threshold_variants(raw(model,x[calibration]),y[calibration],
        raw(model,x[selection]),y[selection],model.base.task,
        config['decision_grid'],config['decision_shrinkage'])
    selected = min(variants,key=lambda r:(-r['validation']['f1_macro'],r['validation']['loss'],r['threshold_fraction']))
    calibrated = replace(model,decision_shifts=np.array(selected['decision_shifts'],np.float32))
    return calibrated,selected,variants


def prepare_case(name,seed,config,output,x,y,source_indices,metadata):
    from experiments.online_leaf_boosting import CompactLeafBoosting
    import cupy as cp
    source = Path(config['source_dir'])/f'{name}_{seed}'/'splits.npz'
    development = Path(config['inference_parent_dir'])/f'{name}_{seed}'/'development_splits.npz'
    directory = output/f'{name}_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as f:
        indices = {k:f[k].copy() for k in ('train','validation','test')}
        np.testing.assert_array_equal(source_indices,f['source_indices'])
    with np.load(development,allow_pickle=False) as f:
        train,calibration,selection = [f[k].copy() for k in ('train','calibration','selection')]
    np.testing.assert_array_equal(train,indices['train'])
    np.testing.assert_array_equal(np.sort(np.r_[calibration,selection]),np.sort(indices['validation']))
    assert len(np.unique(np.r_[train,calibration,selection,indices['test']])) == len(x)
    np.savez(directory/'splits.npz',**indices,calibration=calibration,selection=selection,source_indices=source_indices)
    c = int(y.max()+1) if metadata['task'] == 'multiclass' else y.shape[1]
    models,fit_records,full_snapshot,seen_warm = {},[],None,None
    order = np.random.default_rng(seed+9127).permutation(MODES).tolist()
    for mode in order:
        native = build_model('full',metadata['task'],c,config,seed,config['max_depth'],config['lambda_l2'])
        model = CompactLeafBoosting(**native.params,compact_mode=mode,task=metadata['task'],
            warmup=config['online_warmup'],rank_fraction=config['online_rank_fraction'],
            residual_fraction=config['online_residual_fraction'])
        del native
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        model.fit(x[train],y[train],eval_sets=[dict(X=x[indices['validation']],y=y[indices['validation']])])
        cp.cuda.get_current_stream().synchronize()
        fit_seconds = time.perf_counter()-started
        if seen_warm is None:
            seen_warm = model.warm_fingerprint
        if model.warm_fingerprint != seen_warm:
            raise RuntimeError('common full warmup differs across variants')
        deployed = model.export()
        replay = probabilities(raw(deployed,x[indices['validation']]),metadata['task'])
        stored_training = probabilities(model.validation_final_raw[0],metadata['task'])
        replay_error = float(np.max(np.abs(replay-stored_training)))
        np.testing.assert_allclose(replay,stored_training,atol=2e-5,rtol=0)
        if len(model.parts) != config['ntrees']:
            raise RuntimeError('tree count changed')
        if mode == 'full':
            full_snapshot = dict(parts=model.parts,weights=model.full_weights,bias=model.study_bias.copy(),
                basis=model.warm_basis_full.copy(),warm_mean=model.warm_train_mean.copy(),task=metadata['task'])
            if metadata['task'] == 'multiclass':
                deployed = replace(deployed,base=reference_class_model(deployed.base))
            np.savez(directory/'full_leaf_curvature.npz',diagonal=np.concatenate(model.full_weights),
                offsets=np.r_[0,np.cumsum([len(a) for a in model.full_weights])],
                warm_basis=model.warm_basis_full,warm_mean=model.warm_train_mean)
        else:
            assert model.full_weights == [] and model.warm_memberships == [] and model.warm_basis_full is None
        models['online_'+mode] = deployed
        record = dict(mode=mode,fit_seconds=fit_seconds,rank=c if model.basis is None else model.basis.shape[1],
            training_pool_peak_bytes=model.training_pool_peak,stage_ms=model.stage_ms,trace=model.trace,
            full_statistics_diagnostic_bytes=sum(w.nbytes for w in model.full_weights),
            warm_fingerprint=model.warm_fingerprint,training_export_max_probability_error=replay_error,
            export_parameter_count=deployed.parameter_count,export_payload_bytes=deployed.payload_bytes,
            history=model.history)
        fit_records.append(record)
        write_report(directory/f'fit_{mode}.json',record)
        emit('online_fit_complete',dataset=name,seed=seed,mode=mode,seconds=fit_seconds,
             rank=record['rank'],pool_peak_bytes=model.training_pool_peak,replay_error=replay_error,
             common_warm_fingerprint=seen_warm)
        del model,stored_training,replay
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    assert full_snapshot is not None
    derived = []
    for mode in MODES[1:]:
        started = time.perf_counter()
        models['posthoc_'+mode] = posthoc(full_snapshot,x[train],mode,config)
        online,offline = models['online_'+mode],models['posthoc_'+mode]
        np.testing.assert_array_equal(online.base.basis,offline.base.basis)
        np.testing.assert_allclose(online.base.bias,offline.base.bias,atol=2e-7,rtol=0)
        # Prefix target/support are identical; feedback can differ only later.
        m_online = int(online.base.leaf_offsets[config['online_warmup']])
        m_offline = int(offline.base.leaf_offsets[config['online_warmup']])
        assert m_online == m_offline
        np.testing.assert_array_equal(online.base.coefficients[:m_online],offline.base.coefficients[:m_offline])
        for left,right in ((online,offline),):
            a,b = left.rows < m_online,right.rows < m_offline
            for field in ('rows','columns','values'):
                np.testing.assert_array_equal(getattr(left,field)[a],getattr(right,field)[b])
        derived.append(dict(mode=mode,seconds=time.perf_counter()-started))
    choices,replay_arrays = [],{'features':x[indices['validation']]}
    for method,model in models.items():
        selected,choice,variants = choose_threshold(model,x,y,calibration,selection,config)
        path = directory/f'{method}.npz'
        selected.save(path)
        loaded = ThresholdModel.load(path)
        probability = loaded.predict(x[indices['validation']])
        replay_arrays[method] = probability
        decisions = loaded.predict_labels(x[indices['validation']])
        replay_arrays[method+'_labels'] = decisions
        np.testing.assert_allclose(probability,selected.predict(x[indices['validation']]),rtol=0,atol=2e-14)
        choices.append(dict(method=method,artifact=str(path.relative_to(output)).replace('\\','/'),
            sha256=sha256(path),parameters=loaded.parameter_count,payload_bytes=loaded.payload_bytes,
            file_bytes=path.stat().st_size,reachable_leaves=len(loaded.base.coefficients),
            rank=loaded.base.coefficients.shape[1],nnz=len(loaded.values),
            threshold_fraction=choice['threshold_fraction'],validation=choice['validation'],
            threshold_variants=variants))
    np.savez(directory/'replay.npz',**replay_arrays)
    result = dict(dataset=name,seed=seed,task=metadata['task'],data_sha256=metadata['sha256'],
        source_split_sha256=sha256(source),development_split_sha256=sha256(development),
        split_sha256=sha256(directory/'splits.npz'),common_warm_fingerprint=seen_warm,
        fits=fit_records,posthoc_cost=derived,choices=choices,fit_order=order)
    write_report(directory/'selection.json',result)
    subprocess.run([sys.executable,'-m','experiments.online_leaf_verify',str(directory)],cwd=ROOT,check=True,timeout=180)
    return result


def evaluate_case(case,output,x,y):
    directory = output/f"{case['dataset']}_{case['seed']}"
    with np.load(directory/'splits.npz',allow_pickle=False) as f:
        test = f['test'].copy()
    full = next(r for r in case['choices'] if r['method'] == 'online_full')
    full_model = ThresholdModel.load(output/full['artifact'])
    reference = calibrated_quality(y[test],raw(full_model,x[test]),case['task'],full_model.decision_shifts)
    rows = []
    for choice in case['choices']:
        path = output/choice['artifact']
        if sha256(path) != choice['sha256']:
            raise RuntimeError('sealed model changed')
        model = ThresholdModel.load(path)
        scores = raw(model,x[test])
        q = calibrated_quality(y[test],scores,case['task'],model.decision_shifts)
        plain = calibrated_quality(y[test],scores,case['task'],np.empty(0,np.float32))
        rows.append(dict(dataset=case['dataset'],seed=case['seed'],
            test=q,test_uncalibrated=plain,test_reference=reference,
            quality_pass=quality_guard(q,reference),
            validation_quality_pass=quality_guard(choice['validation'],full['validation']),
            parameter_ratio=choice['parameters']/full['parameters'],
            payload_ratio=choice['payload_bytes']/full['payload_bytes'],
            loss_ratio=q['loss']/reference['loss'],macro_retained=q['f1_macro']/reference['f1_macro'],
            weighted_retained=q['f1_weighted']/reference['f1_weighted'],**choice))
    write_report(directory/'test.json',rows)
    return rows


def summarize(records):
    summaries = []
    for dataset,method in sorted({(r['dataset'],r['method']) for r in records}):
        rows = [r for r in records if (r['dataset'],r['method']) == (dataset,method)]
        summaries.append(dict(dataset=dataset,method=method,splits=len(rows),
            quality_passes=sum(r['quality_pass'] for r in rows),
            validation_quality_passes=sum(r['validation_quality_pass'] for r in rows),
            joint_75_successes=sum(r['validation_quality_pass'] and r['quality_pass'] and r['parameter_ratio'] <= .75 and r['payload_ratio'] <= .75 for r in rows),
            median_parameter_ratio=float(np.median([r['parameter_ratio'] for r in rows])),
            median_payload_ratio=float(np.median([r['payload_ratio'] for r in rows])),
            median_loss_ratio=float(np.median([r['loss_ratio'] for r in rows])),
            median_macro_retained=float(np.median([r['macro_retained'] for r in rows])),
            median_weighted_retained=float(np.median([r['weighted_retained'] for r in rows]))))
    return summaries


def online_leaf_study(config,report,output):
    from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver
    from threadpoolctl import threadpool_limits
    import cupy as cp
    started = time.perf_counter()
    def expired(signum,frame):
        raise TimeoutError('online leaf study walltime exhausted')
    previous = signal.signal(signal.SIGALRM,expired)
    signal.alarm(config['walltime_seconds'])
    ready,reason,probe = gpu_probe('shared',config['inference_gpu_minimum_free_mib'])
    if not ready:
        raise RuntimeError(reason)
    cp.cuda.runtime.free(0)
    guard = SharedGpuObserver()
    cases,records = [],[]
    report.update(status='running',scientific_fits=0,teacher_test_selection=False,
        interpretation='Fixed-basis compact feedback prototype,not tensor history or parallel training')
    locks = {}
    try:
        with guard,threadpool_limits(limits=1):
            # PyBoost creates a stdout handler during import. In-process pytest
            # binds it to its temporary captured stream, closed after the test
            # session. Isolate the regression suite; keep training stdout live.
            subprocess.run([sys.executable,'-m','pytest','-q','tests/test_online_leaf_core.py'],
                           cwd=ROOT,check=True,timeout=300)
            datasets = {name:read_dataset(name,config) for name in config['datasets']}
            old = Path(config['prepared_source_manifest'])
            if sha256(old) != config['prepared_source_manifest_sha256']:
                raise RuntimeError('old cohort changed')
            old_manifest = json.loads(old.read_text())
            old_selection = Path(config['inference_parent_dir'])/'selection_manifest.json'
            if sha256(old_selection) != config['inference_parent_manifest_sha256']:
                raise RuntimeError('old calibration selection manifest changed')
            old_selections = json.loads(old_selection.read_text())['selections']
            locks[str(old)] = sha256(old)
            locks[str(old_selection)] = sha256(old_selection)
            for name in config['datasets']:
                locks[str(Path(config['data_dir'])/f'{name}.npz')] = datasets[name][3]['sha256']
                for seed in config['seeds_by_dataset'][name]:
                    old_case = next(c for c in old_manifest['cases'] if c['dataset'] == name and c['seed'] == seed)
                    source = str(Path(config['source_dir'])/f'{name}_{seed}'/'splits.npz')
                    expected = old_manifest['file_locks'][old_case['split_file']]
                    if sha256(source) != expected:
                        raise RuntimeError('old split changed')
                    locks[source] = expected
                    development = str(Path(config['inference_parent_dir'])/f'{name}_{seed}'/'development_splits.npz')
                    old_chosen = next(r for r in old_selections if r['dataset'] == name and r['seed'] == seed)
                    if sha256(development) != old_chosen['development_split_sha256']:
                        raise RuntimeError('old calibration/selection partition changed')
                    locks[development] = old_chosen['development_split_sha256']
            emit('online_protocol_sealed',datasets=config['datasets'],methods=list(MODES),
                posthoc=['rank','hybrid','more_rank'],scientific_fits=sum(len(config['seeds_by_dataset'][n]) for n in config['datasets'])*4,
                frozen_files=len(locks),test_evaluation='after_all_selections_sealed')
            for name in config['datasets']:
                x,y,source_indices,metadata = datasets[name]
                for seed in config['seeds_by_dataset'][name]:
                    case = prepare_case(name,seed,config,output,x,y,source_indices,metadata)
                    cases.append(case)
                    report.update(scientific_fits=len(cases)*4,cases=cases,
                        elapsed_seconds=time.perf_counter()-started)
                    write_report(output/'progress.json',report)
                    emit('online_selection_case_complete',dataset=name,seed=seed,fits=len(cases)*4,
                         elapsed_seconds=report['elapsed_seconds'])
            write_report(output/'selection_manifest.json',dict(cases=cases,file_locks=locks))
            sealed_sha = sha256(output/'selection_manifest.json')
            emit('online_all_selections_sealed',sha256=sealed_sha,cases=len(cases))
            for case in cases:
                x,y,_,_ = datasets[case['dataset']]
                records.extend(evaluate_case(case,output,x,y))
                emit('online_test_case_complete',dataset=case['dataset'],seed=case['seed'])
            if sha256(output/'selection_manifest.json') != sealed_sha:
                raise RuntimeError('sealed selection changed')
            for path,expected in locks.items():
                if sha256(path) != expected:
                    raise RuntimeError('frozen input changed during study')
        summaries = summarize(records)
        report.update(status='done',records=records,cases=cases,summaries=summaries,
            file_locks=locks,selection_sha256=sealed_sha,scientific_fits=len(cases)*4,
            gpu_probe=probe,gpu_guard_samples=guard.samples,elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        emit('online_leaf_complete',scientific_fits=report['scientific_fits'],cases=len(cases),
            frozen_files=len(locks),summaries=summaries,elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',cases=cases,
                      scientific_fits=len(cases)*4,gpu_probe=probe,gpu_guard_samples=guard.samples)
        write_report(output/'report.json',report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
