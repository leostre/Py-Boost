"""H-LR1 only: frozen topology, training-calibrated PCA, held-out rank."""
from __future__ import annotations

from dataclasses import replace
import gc
import signal
import subprocess
import sys
import time
import numpy as np

from experiments.a100_runner import (ROOT, build_model, emit, make_partitions,
    model_complexity, quality, read_dataset, sha256, write_report)
from experiments.shared_leaf_basis import (LeafEnsemble, compress, export_teacher,
    functional_basis, probabilities, quality_guard, reference_class_model, select_rank)
from experiments.shared_leaf_gpu import GpuLeafEnsemble


def assert_close(a, b, label, tolerance=2e-5):
    error = float(np.max(np.abs(a - b)))
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f'{label}: maximum probability difference {error} > {tolerance}')
    return error


def timed_gpu(model, features):
    import cupy as cp
    model.predict(features)
    times = []
    for _ in range(3):
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        model.predict(features)
        cp.cuda.get_current_stream().synchronize()
        times.append(1000 * (time.perf_counter() - started))
    return float(np.median(times))


def preflight(config):
    import cupy as cp
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_shared_leaf_basis.py',
                    'tests/test_compactness_core.py'], cwd=ROOT, check=True)
    rng = np.random.default_rng(772)
    x = rng.normal(size=(512, 6)).astype(np.float32)
    x[::17, 0] = np.nan
    y = ((np.nan_to_num(x[:, 0]) > 0).astype(int) + 2 * (x[:, 1] > 0)).astype(np.int32)
    fingerprints = []
    for _ in range(2):
        model = build_model('full', 'multiclass', 4, config, 991, 3, 1., trees=4)
        model.fit(x[:400], y[:400], eval_sets=[{'X': x[400:], 'y': y[400:]}])
        native = model.predict(x[400:])
        fingerprints.append(native.tobytes())
        dense = export_teacher(model, x.shape[1], 'multiclass')
        assert_close(dense.predict(x[400:]), native, 'toy CPU/native')
        assert_close(GpuLeafEnsemble(dense).predict(x[400:]), native, 'toy GPU/native')
        mean, basis, _ = functional_basis(dense.contributions(dense.leaf_indices(x[:400])), 'multiclass')
        compact = compress(dense, mean, basis)
        assert_close(GpuLeafEnsemble(compact).predict(x[400:]), native, 'toy full contrast rank')
        del model, dense, compact
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    if fingerprints[0] != fingerprints[1]:
        raise RuntimeError('toy teacher fit not deterministic')
    emit('hlr1_preflight', status='passed', deterministic_fits=2, routing_nan_check=True)


def one_split(name, seed, config, output, features, target, source_indices, metadata):
    import cupy as cp
    task = metadata['task']
    c = int(target.max() + 1) if task == 'multiclass' else target.shape[1]
    train, validation, test = make_partitions(target, task, seed)
    directory = output / f'{name}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    split_path = directory / 'splits.npz'
    np.savez(split_path, train=train, validation=validation, test=test, source_indices=source_indices)
    model = build_model('full', task, c, config, seed, config['max_depth'], config['lambda_l2'])
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    model.fit(features[train], target[train], eval_sets=[{'X': features[validation], 'y': target[validation]}])
    cp.cuda.get_current_stream().synchronize()
    train_seconds = time.perf_counter() - started
    if len(model.models) != config['ntrees']:
        raise RuntimeError('teacher tree count changed')
    dense = export_teacher(model, features.shape[1], task)
    native_cost = model_complexity(model, [config['ntrees']])[0]
    if dense.parameter_count != native_cost['parameters']:
        raise RuntimeError('reachable coefficient accounting differs from native tree accounting')
    reference = reference_class_model(dense) if task == 'multiclass' else dense
    dense.save(directory / 'dense_teacher.npz')
    reference.save(directory / 'reference.npz')
    train_leaves = dense.leaf_indices(features[train])
    valid_leaves = dense.leaf_indices(features[validation])
    native_validation = model.predict(features[validation])
    validation_reference = quality(target[validation], native_validation, task)
    native_parity = assert_close(dense.predict(features[validation]), native_validation, 'real CPU/native')
    reference_parity = assert_close(reference.predict(features[validation]), native_validation, 'reference gauge')
    compression_started = time.perf_counter()
    contributions = dense.contributions(train_leaves)
    mean, basis, singular = functional_basis(contributions, task)
    full = compress(dense, mean, basis)
    # Sum actual float32 leaf factors, not projected teacher test logits.
    validation_latent = full.contributions(valid_leaves)
    curve = []
    for rank in range(1, basis.shape[1] + 1):
        b = basis[:, :rank]
        bias = (dense.bias + mean - (mean @ b) @ b.T).astype(np.float32)
        raw = validation_latent[:, :rank] @ full.basis[:, :rank].T + bias
        parameters = len(dense.coefficients) * rank + c * rank + c
        curve.append({'rank': rank, 'parameters': parameters,
                      'quality': quality(target[validation], probabilities(raw, task), task),
                      'retained_centered_energy': float(np.sum(singular[:rank] ** 2) / np.sum(singular ** 2))
                      if np.any(singular) else 1.})
    rank, feasible = select_rank(curve, validation_reference, reference.parameter_count,
                                 config['parameter_fraction'])
    b = basis[:, :rank]
    selected = replace(full, coefficients=full.coefficients[:, :rank].copy(),
                       basis=full.basis[:, :rank].copy(),
                       bias=(dense.bias + mean - (mean @ b) @ b.T).astype(np.float32))
    compact_path = directory / 'compact.npz'
    selected.save(compact_path)
    loaded = LeafEnsemble.load(compact_path)
    selected_validation = quality(target[validation], loaded.predict(features[validation]), task)
    for key in selected_validation:
        if abs(selected_validation[key] - curve[rank - 1]['quality'][key]) > 1e-10:
            raise RuntimeError('rank selection prediction differs from standalone artifact')
    # Written before any use of test predictions or test labels.
    emit('hlr1_rank_selected', dataset=name, seed=seed, rank=rank, feasible=feasible,
         reference_parameters=reference.parameter_count, parameters=selected.parameter_count,
         validation=selected_validation)
    compression_seconds = time.perf_counter() - compression_started
    full_rank_parity = assert_close(probabilities(full.raw_from_leaves(valid_leaves), task),
                                   native_validation, 'full-rank functional model')
    cpu_prediction = loaded.predict(features[test])
    gpu_compact = GpuLeafEnsemble(loaded)
    compact_gpu_parity = assert_close(gpu_compact.predict(features[test]), cpu_prediction, 'compact GPU/CPU')
    native_test = model.predict(features[test])
    reference_test = quality(target[test], native_test, task)
    selected_test = quality(target[test], cpu_prediction, task)
    gpu_reference = GpuLeafEnsemble(reference)
    latency_reference = timed_gpu(gpu_reference, features[test])
    latency_compact = timed_gpu(gpu_compact, features[test])
    np.savez(directory / 'replay.npz', features=features[test], probabilities=cpu_prediction)
    record = {
        'dataset': name, 'seed': seed, 'task': task, 'outputs': c,
        'train_rows': len(train), 'validation_rows': len(validation), 'test_rows': len(test),
        'data_sha256': metadata['sha256'], 'split_sha256': sha256(split_path),
        'teacher_train_seconds': train_seconds, 'compression_and_selection_seconds': compression_seconds,
        'topology_unchanged': True, 'boosting_steps': dense.n_trees,
        'reachable_leaves': len(dense.coefficients), 'native_cost': native_cost,
        'reference_representation': reference.representation,
        'dense_parameters': dense.parameter_count, 'reference_parameters': reference.parameter_count,
        'parameters': selected.parameter_count, 'parameter_reduction': 1 - selected.parameter_count / reference.parameter_count,
        'reference_payload_bytes': reference.payload_bytes, 'compact_payload_bytes': loaded.payload_bytes,
        'reference_file_bytes': (directory / 'reference.npz').stat().st_size,
        'compact_file_bytes': compact_path.stat().st_size, 'compact_sha256': sha256(compact_path),
        'rank': rank, 'validation_budget_feasible': feasible,
        'validation_reference': validation_reference, 'validation_selected': selected_validation,
        'test_reference': reference_test, 'test_selected': selected_test,
        'test_quality_guard': quality_guard(selected_test, reference_test),
        'joint_success': feasible and quality_guard(selected_test, reference_test),
        'test_loss_ratio': selected_test['loss'] / reference_test['loss'],
        'reference_gpu_median_ms': latency_reference, 'compact_gpu_median_ms': latency_compact,
        'latency_policy': 'same standalone routing/gather implementation, warm 3 repeats, host input/output; exploratory',
        'parity_max_probability_error': {'native_cpu': native_parity, 'reference_class': reference_parity,
                                        'full_rank': full_rank_parity, 'compact_gpu_cpu': compact_gpu_parity},
        'validation_curve': curve,
    }
    np.savez(directory / 'calibration_diagnostics.npz', mean=mean, singular_values=singular,
             basis=basis, train_contribution_mean=contributions.mean(axis=0))
    write_report(directory / 'record.json', record)
    del model, dense, reference, selected, full, loaded, gpu_reference, gpu_compact
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return record


def shared_leaf_study(config, report, output):
    def expired(signum, frame):
        raise TimeoutError('approved H-LR1 server compute budget exhausted')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    started = time.perf_counter()
    records = []
    report.update(status='running', records=records,
                  interpretation='H-LR1 post-training compression only; no training-compute saving claim')
    try:
        preflight(config)
        for name in config['datasets']:
            features, target, source_indices, metadata = read_dataset(name, config)
            emit('data_verified', dataset=name, **metadata)
            for seed in config['seeds_by_dataset'][name]:
                record = one_split(name, seed, config, output, features, target, source_indices, metadata)
                records.append(record)
                report['completed_fits'] = len(records)
                write_report(output / 'report.json', report)
                emit('hlr1_split_finished', **{k: record[k] for k in ('dataset', 'seed', 'rank',
                     'parameter_reduction', 'test_loss_ratio', 'test_quality_guard', 'joint_success')},
                     completed=len(records), elapsed_seconds=time.perf_counter() - started)
        subprocess.run([sys.executable, '-m', 'experiments.shared_leaf_verify', str(output)],
                       cwd=ROOT, check=True)
        summaries = []
        for name in config['datasets']:
            rows = [r for r in records if r['dataset'] == name]
            summaries.append({'dataset': name, 'splits': len(rows),
                'validation_feasible': sum(r['validation_budget_feasible'] for r in rows),
                'joint_successes': sum(r['joint_success'] for r in rows),
                'ranks': [r['rank'] for r in rows],
                'median_parameter_reduction': float(np.median([r['parameter_reduction'] for r in rows])),
                'median_test_loss_ratio': float(np.median([r['test_loss_ratio'] for r in rows])),
                'median_payload_reduction': float(np.median([1-r['compact_payload_bytes']/r['reference_payload_bytes'] for r in rows]))})
        report.update(status='complete', summaries=summaries, standalone_replay='passed',
                      elapsed_seconds=time.perf_counter() - started)
        emit('hlr1_summary', summaries=summaries, elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.perf_counter() - started)
        write_report(output / 'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
