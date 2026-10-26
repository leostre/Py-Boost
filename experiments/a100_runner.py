"""Fixed entry point for A100 setup validation and H1 compactness study."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from experiments.a100_environment import environment_snapshot
from experiments.compactness_core import (
    ValidationCandidate, count_tree_payload, select_compact, validate_partition,
    normalize_multiclass_probabilities,
)


def emit(event, **data):
    print(json.dumps({'event': event, **data}, ensure_ascii=False, allow_nan=False), flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_environment(config):
    path = Path(config['environment_lock'])
    expected = json.loads(path.read_text())
    actual = json.loads(json.dumps(environment_snapshot()))
    if actual != expected:
        raise RuntimeError('Python/CUDA environment changed since registration')
    return sha256(path)


def read_dataset(name, config):
    directory = Path(config['data_dir'])
    entry = json.loads(Path(__file__).with_name('a100_data_manifest.json').read_text())[name]
    path = directory / f'{name}.npz'
    if sha256(path) != entry['sha256']:
        raise RuntimeError(f'dataset fingerprint mismatch: {name}')
    with np.load(path, allow_pickle=False) as data:
        features = np.ascontiguousarray(data['features'], dtype=np.float32)
        target = data['target'].copy()
        indices = data['source_indices'].copy()
    if list(features.shape) != entry['features_shape'] or list(target.shape) != entry['target_shape']:
        raise RuntimeError('dataset shape changed')
    return features, target, indices, entry


def make_partitions(target, task, seed):
    rows = np.arange(len(target))
    if task == 'multiclass':
        train_validation, test = train_test_split(rows, test_size=.2, random_state=seed,
                                                  stratify=target)
        train, validation = train_test_split(train_validation, test_size=.25,
                                             random_state=seed + 1,
                                             stratify=target[train_validation])
    else:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
        train_validation, test = next(splitter.split(rows[:, None], target))
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
        a, b = next(splitter.split(rows[train_validation, None], target[train_validation]))
        train, validation = train_validation[a], train_validation[b]
    validate_partition(train, validation, test, len(target))
    return train, validation, test


def quality(target, probabilities, task):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError('nonfinite predictions')
    if task == 'multiclass':
        probabilities = normalize_multiclass_probabilities(probabilities)
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        loss = float(-np.log(p[np.arange(len(target)), target.astype(int)]).mean())
        predicted = probabilities.argmax(axis=1)
    else:
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise RuntimeError('multilabel prediction outside [0, 1]')
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        loss = float(-(target * np.log(p) + (1 - target) * np.log1p(-p)).mean())
        predicted = probabilities >= .5
    return {'loss': loss,
            'f1_weighted': float(f1_score(target, predicted, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(target, predicted, average='macro', zero_division=0))}


def build_model(method, task, n_outputs, config, seed, depth, regularization, trees=None, callbacks=None):
    from py_boost import GradientBoosting, SketchBoost
    arguments = dict(loss='crossentropy' if task == 'multiclass' else 'logloss',
                     ntrees=config['ntrees'] if trees is None else trees,
                     lr=config['lr'], max_depth=depth, lambda_l2=regularization,
                     min_data_in_leaf=10, subsample=1., colsample=1.,
                     max_bin=128, es=0, seed=seed, verbose=200, callbacks=callbacks)
    if method == 'full':
        model = GradientBoosting(**arguments)
    elif method == 'sketchboost_proj_half':
        model = SketchBoost(**arguments, sketch_method='proj',
                            sketch_outputs=max(1,int(np.ceil(n_outputs*.5))),use_hess=False)
    elif method.startswith('sketchboost_topk_'):
        fractions = {'half': .5, 'quarter': .25, 'threequarters': .75}
        fraction = fractions[method.removeprefix('sketchboost_topk_')]
        model = SketchBoost(**arguments, sketch_method='topk',
                            sketch_outputs=max(1, int(np.ceil(n_outputs * fraction))), use_hess=True)
        if config.get('identity_full_sketch', False):
            from experiments.mechanism_sketch import IdentityAtFullTopK
            model.params['multioutput_sketch'] = IdentityAtFullTopK(n_outputs)
    else:
        raise ValueError(f'unknown method {method}')
    model.params.update(full_leaf_values=True, deterministic_leaf_values=True,
                        deterministic_histogram=True, deterministic_histogram_layout='hybrid')
    return model


def as_numpy(values):
    return values.get() if hasattr(values, 'get') else np.asarray(values)


def model_complexity(model, checkpoints):
    counts = [count_tree_payload(*(as_numpy(getattr(tree, name)) for name in
                                  ('test_format', 'test_format_offsets', 'values', 'group_index')))
              for tree in model.models]
    result = []
    for checkpoint in checkpoints:
        totals = {name: sum(getattr(c, name) for c in counts[:checkpoint])
                  for name in counts[0].__dataclass_fields__}
        totals['parameters'] = totals['leaf_coefficients'] + int(model.base_score.size)
        totals['inference_payload_bytes'] += int(model.base_score.nbytes)
        totals['boosting_steps'] = checkpoint
        result.append(totals)
    return result


def preflight(config, report):
    import cupy as cp
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    'tests/test_compactness_core.py'], cwd=ROOT, check=True)
    for name in config['datasets']:
        _, _, _, metadata = read_dataset(name, config)
        emit('data_verified', dataset=name, **metadata)
    rng = np.random.default_rng(123)
    features = rng.normal(size=(512, 8)).astype(np.float32)
    target = ((features[:, 0] > 0).astype(int) + 2 * (features[:, 1] > 0)).astype(np.int32)
    checks = []
    for method in config['methods']:
        fingerprints = []
        for _ in range(2):
            model = build_model(method, 'multiclass', 4, config, 991, 3, 1., trees=4)
            model.fit(features[:400], target[:400])
            prediction = model.predict(features[400:])
            counts = model_complexity(model, [4])[0]
            fingerprints.append(hashlib.sha256(prediction.tobytes()).hexdigest())
            checks.append({'method': method, **counts, **quality(target[400:], prediction, 'multiclass')})
            del model
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError(f'preflight not repeatable: {method}')
    report.update(status='passed', gpu_checks=checks,
                  scientific_claim='deployment verified; H1 not tested')
    emit('preflight_summary', **report)


def h1(config, report, output):
    import cupy as cp
    selections = []
    records = []
    expected_fits = (len(config['datasets']) * len(config['seeds']) * len(config['methods'])
                     * len(config['depths']) * len(config['lambda_l2']))
    completed = 0
    report.update(status='running', expected_fits=expected_fits, records=records, selections=selections)
    for name in config['datasets']:
        features, target, source_indices, metadata = read_dataset(name, config)
        task = metadata['task']
        n_outputs = int(target.max() + 1) if task == 'multiclass' else target.shape[1]
        for seed in config['seeds']:
            train, validation, test = make_partitions(target, task, seed)
            split_file = output / 'splits' / f'{name}_{seed}.npz'
            split_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(split_file, train=train, validation=validation, test=test,
                                source_indices=source_indices)
            current_records = []
            for method, depth, regularization in itertools.product(
                    config['methods'], config['depths'], config['lambda_l2']):
                model = build_model(method, task, n_outputs, config, seed, depth, regularization)
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                model.fit(features[train], target[train],
                          eval_sets=[{'X': features[validation], 'y': target[validation]}])
                cp.cuda.get_current_stream().synchronize()
                train_seconds = time.perf_counter() - started
                if len(model.models) != config['ntrees']:
                    raise RuntimeError('unexpected early stopping')
                complexity = model_complexity(model, config['checkpoints'])
                predictions = {}
                for split, indices in [('train', train), ('validation', validation), ('test', test)]:
                    predictions[split] = model.predict_staged(
                        features[indices], iterations=[t - 1 for t in config['checkpoints']])
                for index, checkpoint in enumerate(config['checkpoints']):
                    all_trees = model.models
                    inference_samples = []
                    try:
                        model.models = all_trees[:checkpoint]
                        model.predict(features[test])
                        for _ in range(3):
                            cp.cuda.get_current_stream().synchronize()
                            prediction_started = time.perf_counter()
                            model.predict(features[test])
                            cp.cuda.get_current_stream().synchronize()
                            inference_samples.append((time.perf_counter() - prediction_started) * 1000)
                    finally:
                        model.models = all_trees
                    candidate_id = f'{name}/{seed}/{method}/d{depth}/l2{regularization}/t{checkpoint}'
                    record = {
                        'candidate_id': candidate_id, 'dataset': name, 'seed': seed,
                        'method': method, 'max_depth': depth, 'lambda_l2': regularization,
                        'lr': config['lr'], **complexity[index],
                        'full_fit_train_seconds': train_seconds,
                        'inference_median_ms': float(np.median(inference_samples)),
                        'inference_repeats': len(inference_samples),
                        'split_sha256': sha256(split_file), 'data_sha256': metadata['sha256'],
                        'train': quality(target[train], predictions['train'][index], task),
                        'validation': quality(target[validation], predictions['validation'][index], task),
                        'test': quality(target[test], predictions['test'][index], task),
                    }
                    records.append(record)
                    current_records.append(record)
                completed += 1
                report['completed_fits'] = completed
                write_report(output / 'report.json', report)
                emit('fit_finished', completed=completed, expected=expected_fits,
                     dataset=name, seed=seed, method=method, depth=depth,
                     lambda_l2=regularization, seconds=train_seconds)
                del model, predictions
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
            # Test metrics are not passed to the selection function.
            for scope in [*config['methods'], 'pooled']:
                scope_records = [r for r in current_records if scope == 'pooled' or r['method'] == scope]
                choices = [ValidationCandidate(r['candidate_id'], r['parameters'],
                                               r['validation']['loss']) for r in scope_records]
                lookup = {r['candidate_id']: r for r in scope_records}
                for tolerance in config['tolerances']:
                    reference, compact = select_compact(choices, tolerance)
                    a, b = lookup[reference.candidate_id], lookup[compact.candidate_id]
                    reduction = 1 - compact.parameters / reference.parameters
                    test_loss_ratio = b['test']['loss'] / a['test']['loss']
                    selection = {
                        'dataset': name, 'seed': seed, 'scope': scope, 'tolerance': tolerance,
                        'reference': reference.candidate_id, 'compact': compact.candidate_id,
                        'reference_parameters': reference.parameters, 'compact_parameters': compact.parameters,
                        'parameter_reduction': reduction, 'test_loss_ratio': test_loss_ratio,
                        'test_quality_within_tolerance': test_loss_ratio <= 1 + tolerance,
                        'joint_success': reduction >= config['min_parameter_reduction']
                                        and test_loss_ratio <= 1 + tolerance,
                    }
                    selections.append(selection)
                    emit('selection', **selection)
            write_report(output / 'report.json', report)
    summaries = []
    for name, scope, tolerance in itertools.product(config['datasets'],
            [*config['methods'], 'pooled'], config['tolerances']):
        rows = [s for s in selections if s['dataset'] == name and s['scope'] == scope
                and s['tolerance'] == tolerance]
        summaries.append({'dataset': name, 'scope': scope, 'tolerance': tolerance,
                          'repeats': len(rows), 'successes': sum(s['joint_success'] for s in rows),
                          'median_parameter_reduction': float(np.median([s['parameter_reduction'] for s in rows])),
                          'median_test_loss_ratio': float(np.median([s['test_loss_ratio'] for s in rows]))})
    report.update(status='complete', summaries=summaries,
                  interpretation='Exploratory headroom within the fixed grid, not proof of adaptive control or RMT.')
    emit('h1_summary', completed_fits=completed, summaries=summaries)


def main():
    config_file = Path(__file__).with_name('a100_config.json')
    config = json.loads(config_file.read_text())
    environment_sha = validate_environment(config)
    output = ROOT / config['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    report = {'phase': config['phase'], 'config': config, 'config_sha256': sha256(config_file),
              'environment_sha256': environment_sha,
              'selection_policy': 'validation-only; test never controls training or candidate choice',
              'probability_policy': 'multiclass component-clipping drift <= d*1e-6 + 1e-6; normalize rows for loss',
              'model_cost': 'reachable leaf coefficients plus base score; thresholds and padding reported separately'}
    emit('configuration', **report)
    with threadpool_limits(limits=8):
        if config['phase'] == 'held_budget_event':
            from experiments.held_budget_event_study import held_budget_event_study
            held_budget_event_study(config, report, output)
        elif config['phase'] == 'reserve_validation_training':
            from experiments.absolute_reserve_validation import reserve_validation_training
            reserve_validation_training(config, report, output)
        elif config['phase'] == 'absolute_variant_training':
            from experiments.absolute_variant_study import absolute_variant_training
            absolute_variant_training(config, report, output)
        elif config['phase'] == 'absolute_history_training':
            from experiments.absolute_history_study import absolute_history_training
            absolute_history_training(config, report, output)
        elif config['phase'] == 'absolute_roles_preflight':
            from experiments.absolute_history_study import prepare_absolute_roles
            prepare_absolute_roles(config, report, output)
        elif config['phase'] == 'direct_leaf_solver':
            from experiments.direct_leaf_study import direct_leaf_study
            direct_leaf_study(config, report, output)
        elif config['phase'] == 'cumulative_final_test':
            from experiments.cumulative_budget_audit import cumulative_final_test
            cumulative_final_test(config, report, output)
        elif config['phase'] == 'cumulative_training':
            from experiments.cumulative_budget_study import cumulative_training
            cumulative_training(config, report, output)
        elif config['phase'] == 'cumulative_preflight':
            from experiments.cumulative_budget_study import cumulative_preflight
            cumulative_preflight(config, report, output)
        elif config['phase'] == 'bank_allocation':
            from experiments.bank_allocation_study import bank_allocation_study
            bank_allocation_study(config, report, output)
        elif config['phase'] == 'model_bank_capacity':
            from experiments.model_bank_capacity import model_bank_capacity
            model_bank_capacity(config, report, output)
        elif config['phase'] == 'model_bank_inventory':
            from experiments.model_bank_study import model_bank_inventory
            model_bank_inventory(config, report, output)
        elif config['phase'] == 'birds_selection_development':
            from experiments.birds_selection_study import birds_selection_development
            birds_selection_development(config, report, output)
        elif config['phase'] == 'birds_selection_test':
            from experiments.birds_selection_study import birds_selection_test
            birds_selection_test(config, report, output)
        elif config['phase'] == 'guard_evidence':
            from experiments.guard_evidence import guard_evidence
            guard_evidence(config, report, output)
        elif config['phase'] == 'guard_budget_audit':
            from experiments.guard_budget_audit import guard_budget_audit
            guard_budget_audit(config, report, output)
        elif config['phase'] == 'guard_budget_loop':
            from experiments.guard_budget_study import guard_budget_study
            guard_budget_study(config, report, output)
        elif config['phase'] == 'prefix_budget_control':
            from experiments.prefix_budget_study import prefix_budget_study
            prefix_budget_study(config, report, output)
        elif config['phase'] == 'closed_budget_loop':
            from experiments.closed_budget_study import closed_budget_study
            closed_budget_study(config, report, output)
        elif config['phase'] == 'leaf_geometry':
            from experiments.leaf_geometry_study import leaf_geometry_study
            leaf_geometry_study(config, report, output)
        elif config['phase'] == 'joint_budget_audit':
            from experiments.joint_budget_audit import joint_budget_audit
            joint_budget_audit(config, report, output)
        elif config['phase'] == 'joint_budget_map':
            from experiments.joint_budget_study import joint_budget_study
            joint_budget_study(config, report, output)
        elif config['phase'] == 'budget_preflight':
            from experiments.leaf_budget_runtime import budget_preflight
            budget_preflight(config, report, output)
        elif config['phase'] == 'expanded_budget_data':
            from experiments.expanded_budget_data import expanded_budget_data
            expanded_budget_data(config, report, output)
        elif config['phase'] == 'rank_residual_map':
            from experiments.rank_residual_map_study import rank_residual_map_study
            rank_residual_map_study(config, report, output)
        elif config['phase'] == 'traversal_preflight':
            from experiments.traversal_preflight import traversal_preflight
            traversal_preflight(config, report, output)
        elif config['phase'] == 'parallel_leaf_study':
            from experiments.parallel_leaf_study import parallel_leaf_study
            parallel_leaf_study(config, report, output)
        elif config['phase'] == 'parallel_leaf_preflight':
            from experiments.parallel_leaf_preflight import parallel_leaf_preflight
            parallel_leaf_preflight(config, report, output)
        elif config['phase'] == 'history_basis_study':
            from experiments.history_basis_study import history_basis_study
            history_basis_study(config, report, output)
        elif config['phase'] == 'history_basis_preflight':
            from experiments.history_basis_preflight import history_basis_preflight
            history_basis_preflight(config, report, output)
        elif config['phase'] == 'online_leaf_study':
            from experiments.online_leaf_study import online_leaf_study
            online_leaf_study(config, report, output)
        elif config['phase'] == 'online_leaf_preflight':
            from experiments.online_leaf_preflight import online_leaf_preflight
            online_leaf_preflight(config, report, output)
        elif config['phase'] == 'gpu_stage_profile':
            from experiments.gpu_stage_profile import gpu_stage_study
            gpu_stage_study(config, report, output)
        elif config['phase'] == 'prepared_inference':
            from experiments.prepared_inference_study import prepared_inference_study
            prepared_inference_study(config, report, output)
        elif config['phase'] == 'inference_benchmark':
            from experiments.inference_benchmark_study import inference_benchmark_study
            inference_benchmark_study(config, report, output)
        elif config['phase'] == 'preflight':
            preflight(config, report)
        elif config['phase'] in ('h1_headroom', 'h2_static'):
            h1(config, report, output)
        elif config['phase'] in ('h3_growth', 'h4_history', 'h5_transfer', 'mechanism_ablation'):
            from experiments.controlled_study import controlled_study
            controlled_study(config, report, output)
        elif config['phase'] == 'risk_budget':
            from experiments.risk_budget_study import risk_budget_study
            risk_budget_study(config, report, output)
        elif config['phase'] == 'shared_leaf_basis':
            from experiments.shared_leaf_study import shared_leaf_study
            shared_leaf_study(config, report, output)
        elif config['phase'] == 'sparse_leaf_comparison':
            from experiments.sparse_leaf_study import sparse_leaf_study
            sparse_leaf_study(config, report, output)
        elif config['phase'] == 'tiny_residual_calibration':
            from experiments.tiny_residual_study import tiny_residual_study
            tiny_residual_study(config, report, output)
        elif config['phase'] == 'uniform_threshold_comparison':
            from experiments.uniform_threshold_study import uniform_threshold_study
            uniform_threshold_study(config, report, output)
        else:
            raise ValueError(f'unknown phase {config["phase"]}')
    write_report(output / 'report.json', report)
    emit('final', status=report['status'], report=str(output / 'report.json'))


if __name__ == '__main__':
    main()
