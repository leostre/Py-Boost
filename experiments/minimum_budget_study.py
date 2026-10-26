"""One frozen-grid minimum-memory check, without fitting trees or thresholds."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, read_dataset, sha256, write_report
from experiments.minimum_budget_core import choose_minimum, QualifiedMinimum
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import LeafEnsemble, compress, quality_guard
from experiments.sparse_leaf_core import (Budget, Cost, contrast, coordinate_scores,
    leaf_design, score_order, training_weights)
from experiments.sparse_leaf_study import base_at_rank, verify_sources
from experiments.uniform_threshold_core import ThresholdModel, family_allows
from experiments.uniform_threshold_study import FAMILIES, materialize_uniform


def minimum_plan(parent):
    """Consume only sealed development metrics; no evaluation data or test API."""
    rows, reference = parent['validation_candidates'], parent['validation_reference']
    decisions = []
    for old in parent['selections']:
        result = choose_minimum(rows, reference, parent['reference_cost']['parameters'],
                                old['family'], Budget(**old['budget']))
        selected = isinstance(result, QualifiedMinimum)
        if selected != old['validation_quality_pass']:
            raise RuntimeError('frozen-grid quality feasibility changed')
        if selected and result.candidate['payload_bytes'] > old['candidate']['payload_bytes']:
            raise RuntimeError('minimum is more expensive than a qualified old choice')
        decisions.append({'policy': 'minimum_bytes', 'status': 'selected' if selected else 'no_qualified_candidate',
            'budget': old['budget'], 'family': old['family'],
            'residual_fraction_cap': old['residual_fraction_cap'],
            'validation_quality_pass': selected,
            'feasible_count': result.feasible_count,
            'qualified_count': result.qualified_count if selected else 0,
            'frontier_ids': [r['candidate_id'] for r in result.frontier] if selected else [],
            'candidate': result.candidate.copy() if selected else None})
        decisions.append({**old, 'policy': 'maximum_f1', 'status': 'selected'})
    return decisions


def copy_checked(source, destination, expected):
    if sha256(source) != expected:
        raise RuntimeError(f'source hash mismatch: {source}')
    shutil.copyfile(source, destination)
    if sha256(destination) != expected:
        raise RuntimeError('copy hash mismatch')


def prepare_minimum(name, seed, config, output, data, input_manifest):
    x, y, source_indices, metadata = data
    started = time.perf_counter()
    parent_dir = Path(config['minimum_parent_dir'])/f'{name}_{seed}'
    source = Path(config['source_dir'])/f'{name}_{seed}'
    directory = output/f'{name}_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    parent = json.loads((parent_dir/'selection.json').read_text())
    if (parent['dataset'], parent['seed'], parent['data_sha256']) != (name, seed, metadata['sha256']):
        raise RuntimeError('parent identity or dataset changed')
    decisions = minimum_plan(parent)
    copy_checked(parent_dir/'selection.json', directory/'source_selection.json',
                 input_manifest['files'][f'{name}_{seed}/selection.json'])
    copy_checked(parent_dir/'development_splits.npz', directory/'development_splits.npz',
                 parent['development_split_sha256'])
    with np.load(directory/'development_splits.npz', allow_pickle=False) as f:
        train, cal, select = (f[k].copy() for k in ('train', 'calibration', 'selection'))
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices, f['source_indices'])
        np.testing.assert_array_equal(train, f['train'])
        np.testing.assert_array_equal(np.sort(np.r_[cal, select]), np.sort(f['validation']))
        # Inspect only indices, not evaluation labels, during preparation.
        partitions = [train, cal, select, f['test']]
        if len(np.unique(np.concatenate(partitions))) != sum(map(len, partitions)):
            raise RuntimeError('partitions overlap')
    dense, reference = (LeafEnsemble.load(source/f'{n}.npz') for n in ('dense_teacher', 'reference'))
    with np.load(source/'calibration_diagnostics.npz', allow_pickle=False) as f:
        mean, basis = f['mean'].copy(), f['basis'].copy()
    design = leaf_design(dense.leaf_indices(x[select]), len(dense.coefficients))
    known = {s['candidate']['candidate_id']: s for s in [parent['full'], *parent['selections']]}
    full_choice = dict(parent['full'])
    all_choices = [full_choice, *[d for d in decisions if d['candidate'] is not None]]
    new_rows = {s['candidate']['candidate_id']: s['candidate'] for s in all_choices
                if s['candidate']['candidate_id'] not in known}
    plan = None
    if new_rows:
        train_design = leaf_design(dense.leaf_indices(x[train]), len(dense.coefficients))
        full = compress(dense, mean, basis)
        teacher = contrast(train_design@dense.coefficients+dense.bias, dense.task)
        latent = train_design@full.coefficients
        plan = {'full': full, 'bases': {}, 'orders': {},
                'frequency': np.asarray(train_design.mean(axis=0)).ravel()}
        weights = {}
        for chosen in new_rows.values():
            rank = chosen['rank']
            if rank not in plan['bases']:
                plan['bases'][rank] = base_at_rank(dense, mean, basis, rank, full)
            key = (rank, chosen['selector'])
            if not chosen['nnz'] or key in plan['orders']:
                continue
            selector = chosen['selector']
            if selector not in weights:
                weights[selector] = training_weights(y[train], teacher, dense.task, selector, config['margin_scale'])
            base = plan['bases'][rank]
            residual = (contrast(dense.coefficients, dense.task)
                - contrast(base.coefficients.astype(float)@base.basis.T, dense.task)).astype(np.float32)
            error = teacher-contrast(latent[:, :rank]@base.basis.T+base.bias, dense.task)
            plan['orders'][key] = score_order(
                coordinate_scores(train_design, residual, error, selector, weights[selector]), residual)
    saved = {}
    for selected in all_choices:
        chosen = selected['candidate']
        cid = chosen['candidate_id']
        if cid not in saved:
            path = directory/f'{cid}.npz'
            if cid in known:
                old = known[cid]
                copy_checked(Path(config['minimum_parent_dir'])/old['artifact'], path, old['sha256'])
            else:
                materialize_uniform(chosen, plan, dense, reference, mean, basis).save(path)
            loaded = ThresholdModel.load(path)
            if (loaded.parameter_count, loaded.payload_bytes) != (chosen['parameters'], chosen['payload_bytes']):
                raise RuntimeError('stored model cost differs from frozen curve')
            np.testing.assert_array_equal(loaded.decision_shifts, np.asarray(chosen['decision_shifts'], np.float32))
            q = calibrated_quality(y[select], loaded.raw_from_design(design), dense.task, loaded.decision_shifts)
            if any(abs(q[k]-chosen['validation'][k]) > 1e-7 for k in q):
                raise RuntimeError(f'candidate reconstruction changed validation metrics: {cid}, {q}, {chosen["validation"]}')
            saved[cid] = {'artifact': str(path.relative_to(output)).replace('\\', '/'),
                          'sha256': sha256(path), 'file_bytes': path.stat().st_size}
        selected.update(saved[cid])
    record = {'dataset': name, 'seed': seed, 'task': dense.task, 'data_sha256': metadata['sha256'],
              'development_split_sha256': sha256(directory/'development_splits.npz'),
              'source_selection_sha256': sha256(directory/'source_selection.json'),
              'reference_cost': parent['reference_cost'], 'full': full_choice,
              'validation_reference': parent['validation_reference'],
              'calibration_rows': len(cal), 'selection_rows': len(select), 'selections': decisions,
              'prepare_seconds': time.perf_counter()-started, 'new_artifacts': len(new_rows)}
    write_report(directory/'selection.json', record)
    emit('minimum_split_sealed', dataset=name, seed=seed, selected=len(saved),
         missing=sum(d['candidate'] is None for d in decisions), new_artifacts=len(new_rows))
    return record


def test_minimum(record, config, output, data):
    x, y, _, _ = data
    source = Path(config['source_dir'])/f'{record["dataset"]}_{record["seed"]}'
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        test = f['test'].copy()
    dense = LeafEnsemble.load(source/'dense_teacher.npz')
    design = leaf_design(dense.leaf_indices(x[test]), len(dense.coefficients))
    cache = {}
    def replay(selected):
        path = output/selected['artifact']
        if sha256(path) != selected['sha256']:
            raise RuntimeError('sealed model changed')
        if selected['artifact'] not in cache:
            model = ThresholdModel.load(path)
            cache[selected['artifact']] = calibrated_quality(y[test], model.raw_from_design(design),
                                                             dense.task, model.decision_shifts)
        return cache[selected['artifact']]
    full_quality = replay(record['full'])
    full_record = {'dataset': record['dataset'], 'seed': record['seed'], 'task': dense.task,
                   **record['full'], 'test': full_quality}
    records = []
    for selected in record['selections']:
        chosen = selected['candidate']
        metrics = replay(selected) if chosen else None
        passed = bool(metrics is not None and quality_guard(metrics, full_quality))
        if chosen:
            assert Budget(**selected['budget']).allows(Cost(chosen['parameters'], chosen['payload_bytes']))
            assert family_allows(chosen, selected['family'], record['reference_cost']['parameters'])
        records.append({'dataset': record['dataset'], 'seed': record['seed'], 'task': dense.task,
            **selected, 'test': metrics, 'test_reference': full_quality, 'test_quality_pass': passed,
            'joint_success': bool(selected['validation_quality_pass'] and passed),
            'reference_cost': record['reference_cost'],
            'macro_retained': metrics['f1_macro']/full_quality['f1_macro'] if metrics else None,
            'parameter_reduction': 1-chosen['parameters']/record['reference_cost']['parameters'] if chosen else None,
            'payload_reduction': 1-chosen['payload_bytes']/record['reference_cost']['payload_bytes'] if chosen else None})
    emit('minimum_test_finished', dataset=record['dataset'], seed=record['seed'], decisions=len(records))
    return full_record, records


def aggregate_minimum(records):
    summaries, pairs = [], []
    groups = sorted({(r['dataset'], r['budget']['kind'], r['budget']['fraction']) for r in records})
    for dataset, kind, fraction in groups:
        group = [r for r in records if (r['dataset'], r['budget']['kind'], r['budget']['fraction']) == (dataset, kind, fraction)]
        for policy in ('minimum_bytes', 'maximum_f1'):
            controls = {r['seed']: r for r in group if r['policy'] == policy and r['family'] == 'rank_only'}
            for family in FAMILIES:
                rows = [r for r in group if r['policy'] == policy and r['family'] == family]
                selected = [r for r in rows if r['candidate'] is not None]
                summaries.append({'dataset': dataset, 'kind': kind, 'fraction': fraction, 'family': family, 'policy': policy,
                    'splits': len(rows), 'selected_count': len(selected), 'missing_count': len(rows)-len(selected),
                    'validation_passes': sum(r['validation_quality_pass'] for r in rows),
                    'test_passes': sum(r['test_quality_pass'] for r in rows),
                    'joint_successes': sum(r['joint_success'] for r in rows),
                    'mean_macro_selected': float(np.mean([r['test']['f1_macro'] for r in selected])) if selected else None,
                    'median_parameter_reduction_selected': float(np.median([r['parameter_reduction'] for r in selected])) if selected else None,
                    'median_payload_reduction_selected': float(np.median([r['payload_reduction'] for r in selected])) if selected else None})
                for r in rows:
                    control = controls[r['seed']]
                    comparable = r['candidate'] is not None and control['candidate'] is not None
                    both_safe = r['joint_success'] and control['joint_success']
                    pairs.append({'dataset': dataset, 'seed': r['seed'], 'kind': kind, 'fraction': fraction,
                        'policy': policy, 'family': family, 'comparable': comparable, 'both_safe': both_safe,
                        'hybrid_success': r['joint_success'], 'rank_success': control['joint_success'],
                        'safe_byte_win': bool(both_safe and r['candidate']['payload_bytes'] < control['candidate']['payload_bytes']),
                        'rank_missing_hybrid_safe': bool(control['candidate'] is None and r['joint_success']),
                        'rank_safe_hybrid_not_safe': bool(control['joint_success'] and not r['joint_success']),
                        'byte_saving_vs_rank': 1-r['candidate']['payload_bytes']/control['candidate']['payload_bytes'] if comparable else None,
                        'macro_delta_vs_rank': r['test']['f1_macro']-control['test']['f1_macro'] if comparable else None})
    primary = []
    for dataset in sorted({r['dataset'] for r in records}):
        p = [r for r in pairs if r['dataset'] == dataset and r['kind'] == 'joint' and r['fraction'] == .75
             and r['policy'] == 'minimum_bytes' and r['family'] == 'sparse_05']
        safe = [r['byte_saving_vs_rank'] for r in p if r['both_safe']]
        primary.append({'dataset': dataset, 'pairs': len(p), 'both_safe': len(safe),
            'safe_byte_wins': sum(r['safe_byte_win'] for r in p),
            'hybrid_successes': sum(r['hybrid_success'] for r in p), 'rank_successes': sum(r['rank_success'] for r in p),
            'rank_missing_hybrid_safe': sum(r['rank_missing_hybrid_safe'] for r in p),
            'rank_safe_hybrid_not_safe': sum(r['rank_safe_hybrid_not_safe'] for r in p),
            'median_byte_saving_both_safe': float(np.median(safe)) if safe else None,
            'descriptive_support': sum(r['safe_byte_win'] for r in p) >= 3
                and sum(r['hybrid_success'] for r in p) >= sum(r['rank_success'] for r in p)})
    return summaries, pairs, primary


def minimum_budget_study(config, report, output):
    def expired(signum, frame):
        raise TimeoutError('approved one-hour CPU check exhausted')
    old_signal = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    started = time.perf_counter()
    report.update(status='running', teacher_fits=0, basis_fits=0, threshold_fits=0, gpu_runs=0,
        selection_policy='frozen validation grid; minimum bytes subject to all original full-quality constraints; no invalid fallback',
        model_cost='unchanged exact-reference denominators; C thresholds charged to every stored model; tree structures included',
        interpretation='exploratory reused tests; minimum only within old grid and caps; no training or latency saving claim')
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_minimum_budget_core.py',
                        'tests/test_uniform_threshold_core.py', 'tests/test_output_calibration_core.py',
                        'tests/test_sparse_leaf_core.py', 'tests/test_compactness_core.py'], cwd=ROOT, check=True)
        teacher_manifest = json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text())
        verify_sources(Path(config['source_dir']), teacher_manifest)
        input_manifest = json.loads((ROOT/'experiments/minimum_source_manifest.json').read_text())
        parent_dir = Path(config['minimum_parent_dir'])
        for relative, expected in input_manifest['files'].items():
            if sha256(parent_dir/relative) != expected:
                raise RuntimeError(f'parent input changed: {relative}')
        datasets = {name: read_dataset(name, config) for name in config['datasets']}
        selections = []
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                selections.append(prepare_minimum(name, seed, config, output, datasets[name], input_manifest))
                gc.collect()
        seal_path = output/'selection_manifest.json'
        write_report(seal_path, {'source_run': input_manifest['source_run'],
            'source_manifest_sha256': sha256(ROOT/'experiments/minimum_source_manifest.json'),
            'test_labels_used_in_selection': False, 'selections': selections})
        seal = sha256(seal_path)
        emit('minimum_all_selections_sealed', sha256=seal, decisions=sum(len(r['selections'])+1 for r in selections),
             test_scoring_started=False)
        full_records, records = [], []
        for selected in selections:
            if sha256(seal_path) != seal:
                raise RuntimeError('decision manifest changed after sealing')
            full_record, rows = test_minimum(selected, config, output, datasets[selected['dataset']])
            full_records.append(full_record)
            records.extend(rows)
        summaries, pairs, primary = aggregate_minimum(records)
        report.update(status='complete', full_records=full_records, records=records, summaries=summaries,
            pairs=pairs, primary=primary, selection_manifest_sha256=seal,
            input_manifest_sha256=sha256(ROOT/'experiments/minimum_source_manifest.json'),
            elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        emit('minimum_summary', primary=primary,
             policies={p: {'choices': sum(r['policy'] == p for r in records),
                           'missing': sum(r['policy'] == p and r['candidate'] is None for r in records),
                           'joint_successes': sum(r['policy'] == p and r['joint_success'] for r in records)}
                       for p in ('minimum_bytes', 'maximum_f1')})
        emit('minimum_complete', teacher_fits=0, threshold_fits=0, basis_fits=0, gpu_runs=0,
             elapsed_seconds=report['elapsed_seconds'], selection_manifest_sha256=seal)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_signal)
