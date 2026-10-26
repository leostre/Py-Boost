"""H-B1: fill missing byte-matched comparisons, never fit a tree or a basis."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, read_dataset, sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, compress, quality_guard
from experiments.sparse_leaf_core import (Cost, budget_for, contrast, coordinate_scores,
    leaf_design, score_order, training_weights, select_candidate)
from experiments.sparse_leaf_study import base_at_rank, verify_sources
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants
from experiments.uniform_threshold_study import materialize_uniform
from experiments.output_calibration_core import calibrated_quality
from experiments.rank_residual_map_core import representation_bytes, matched_allocations, choose_family


def check_locks(locks):
    for path, expected in locks.items():
        if sha256(path) != expected:
            raise RuntimeError('frozen input changed: ' + path)


def prepare_case(name, seed, config, output, data, old_record):
    x, target, source_indices, metadata = data
    case = f'{name}_{seed}'
    directory = output / case
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(config['source_dir']) / case
    old = Path(config['inference_parent_dir']) / case
    previous = json.loads((old / 'selection.json').read_text())
    if previous != old_record:
        # The manifest intentionally excludes the large development candidate grid.
        if {k: v for k, v in previous.items() if k not in ('validation_candidates', 'full_candidates')} != old_record:
            raise RuntimeError('old per-case selection differs from its sealed manifest')
    with np.load(source / 'splits.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices, f['source_indices'])
        old_train, old_validation = f['train'].copy(), f['validation'].copy()
    with np.load(old / 'development_splits.npz', allow_pickle=False) as f:
        parts = {k: f[k].copy() for k in ('train', 'calibration', 'selection')}
    if sha256(old / 'development_splits.npz') != previous['development_split_sha256']:
        raise RuntimeError('development split changed')
    np.testing.assert_array_equal(parts['train'], old_train)
    np.testing.assert_array_equal(np.sort(np.r_[parts['calibration'], parts['selection']]), np.sort(old_validation))
    if len(np.intersect1d(parts['calibration'], parts['selection'])):
        raise RuntimeError('calibration/selection overlap')
    np.savez(directory / 'development_splits.npz', **parts)
    dense, reference = (LeafEnsemble.load(source / (k + '.npz')) for k in ('dense_teacher', 'reference'))
    with np.load(source / 'calibration_diagnostics.npz', allow_pickle=False) as f:
        mean, basis = f['mean'].copy(), f['basis'].copy()
    m, c = dense.coefficients.shape
    fixed = dense.payload_bytes - 4 * m * c + 4 * c
    cost_reference = Cost(reference.parameter_count, reference.payload_bytes)
    if asdict(cost_reference) != previous['reference_cost']:
        raise RuntimeError('reference denominator changed')
    full = compress(dense, mean, basis)
    designs = {k: leaf_design(dense.leaf_indices(x[v]), m) for k, v in parts.items()}
    latents = {k: d @ full.coefficients for k, d in designs.items()}
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    teacher = contrast(designs['train'] @ dense.coefficients + dense.bias, dense.task)
    weights = {s: training_weights(target[parts['train']], teacher, dense.task, s, config['margin_scale'])
               for s in config['tiny_selectors']}
    rows_old = previous['validation_candidates']
    rank_rows = [r for r in rows_old if r['kind'] == 'rank_only']
    cached = {}
    for row in rows_old:
        cached.setdefault((row['rank'], row['selector'], row['nnz']), []).append(row)
    anchors = []
    for fraction in config['budget_fractions']:
        budget = budget_for(cost_reference, 'joint', fraction)
        chosen, passed = select_candidate(rank_rows, previous['validation_reference'], budget)
        anchors.append({'fraction': fraction, 'budget': asdict(budget), 'candidate': chosen,
                        'validation_quality_pass': passed})
    plans, requested = [], {}
    for anchor in anchors:
        q = anchor['candidate']['rank']
        assert representation_bytes(fixed, m, c, q, 0, 'coo') == anchor['candidate']['payload_bytes']
        for storage in config['map_storage_formats']:
            allocations = matched_allocations(fixed, m, c, q, 5 * cost_reference.parameters // 100, storage)
            plans.append({'fraction': anchor['fraction'], 'storage': storage,
                          'anchor': anchor, 'allocations': [asdict(a) for a in allocations]})
            for a in allocations:
                for selector in config['tiny_selectors']:
                    requested[(a.rank, selector, a.nnz)] = None
    bases, orders, new_rows, stats, unavailable = {}, {}, [], {'reused_variants': 0, 'new_variants': 0,
                                                            'threshold_fits': 0}, []
    for rank in sorted({k[0] for k in requested}):
        base = base_at_rank(dense, mean, basis, rank, full)
        bases[rank] = base
        residual = (contrast(dense.coefficients, dense.task)
                    - contrast(base.coefficients.astype(float) @ base.basis.T, dense.task)).astype(np.float32)
        error = teacher - contrast(latents['train'][:, :rank] @ base.basis.T + base.bias, dense.task)
        for selector in config['tiny_selectors']:
            order = score_order(coordinate_scores(designs['train'], residual, error, selector, weights[selector]), residual)
            orders[(rank, selector)] = order
            counts = sorted(k[2] for k in requested if k[:2] == (rank, selector))
            for count in counts:
                key = (rank, selector, count)
                if count > len(order):
                    unavailable.append({'rank': rank, 'selector': selector, 'nnz': count,
                                        'positive_score_coordinates': len(order)})
                    requested[key] = []
                    continue
                if key in cached:
                    requested[key] = cached[key]
                    stats['reused_variants'] += len(cached[key])
                    continue
                from experiments.sparse_leaf_core import make_sparse_model
                leaf = make_sparse_model(base, residual, order, count, frequency)
                if len(leaf.values) != count:
                    raise RuntimeError('zero residual broke exact matching')
                variants = threshold_variants(leaf.raw_from_design(designs['calibration']), target[parts['calibration']],
                    leaf.raw_from_design(designs['selection']), target[parts['selection']], dense.task,
                    config['decision_grid'], config['decision_shrinkage'])
                stats['threshold_fits'] += 1
                built = []
                for variant in variants:
                    suffix = str(variant['threshold_fraction']).replace('.', 'p')
                    row = {'candidate_id': f'sparse-r{rank}-{selector}-{count}-t{suffix}',
                           'kind': 'sparse', 'rank': rank, 'selector': selector, 'nnz': count,
                           'parameters': leaf.parameter_count + c, 'payload_bytes': leaf.payload_bytes + 4 * c,
                           **variant}
                    assert row['payload_bytes'] == representation_bytes(fixed, m, c, rank, count, 'coo')
                    built.append(row)
                requested[key] = built
                new_rows.extend(built)
                stats['new_variants'] += len(built)
        emit('map_rank_prepared', dataset=name, seed=seed, rank=rank, **stats)
    saved = {}
    materialize_plan = {'bases': bases, 'full': full, 'orders': orders, 'frequency': frequency}

    def save(chosen):
        cid = chosen['candidate_id']
        if cid in saved:
            return saved[cid]
        model = materialize_uniform(chosen, materialize_plan, dense, reference, mean, basis)
        path = directory / (cid + '.npz')
        model.save(path)
        loaded = ThresholdModel.load(path)
        q = calibrated_quality(target[parts['selection']], loaded.raw_from_design(designs['selection']), dense.task,
                               loaded.decision_shifts)
        if any(abs(q[k] - chosen['validation'][k]) > 1e-7 for k in q):
            raise RuntimeError('cached/new chosen model failed independent metric replay')
        if loaded.parameter_count != chosen['parameters'] or loaded.payload_bytes != chosen['payload_bytes']:
            raise RuntimeError('serialized coefficient/COO cost mismatch')
        prepared = loaded.prepare_inference()
        if loaded.base.representation == 'low_rank':
            expected = representation_bytes(fixed, m, c, chosen['rank'], chosen['nnz'], 'csr')
            if prepared.resident_bytes != expected:
                raise RuntimeError('prepared CSR array bytes mismatch')
        # Only development features enter this parity check.
        probes = x[parts['selection'][:min(17, len(parts['selection']))]]
        np.testing.assert_allclose(prepared.predict(probes), loaded.predict(probes), rtol=0, atol=2e-12)
        saved[cid] = {'artifact': str(path.relative_to(output)).replace('\\', '/'), 'sha256': sha256(path),
                      'file_bytes': path.stat().st_size, 'coo_bytes': loaded.payload_bytes,
                      'csr_bytes': prepared.resident_bytes}
        return saved[cid]

    full_choice = previous['full']['candidate']
    full_saved = {'candidate': full_choice, **save(full_choice)}
    decisions, map_rows = [], []
    for plan in plans:
        anchor = plan['anchor']['candidate']
        anchor_saved = save(anchor)
        candidates = []
        for allocation in plan['allocations']:
            for selector in config['tiny_selectors']:
                candidates.extend(requested[(allocation['rank'], selector, allocation['nnz'])])
        for row in candidates:
            matching = representation_bytes(fixed, m, c, row['rank'], row['nnz'], plan['storage'])
            map_rows.append({'fraction': plan['fraction'], 'storage': plan['storage'],
                             'anchor_id': anchor['candidate_id'], 'matching_bytes': matching,
                             'slack_bytes': anchor['payload_bytes'] - matching,
                             'candidate': row})
        for percent in (0, 1, 2, 5):
            chosen, passed = choose_family(candidates, anchor, previous['validation_reference'], percent,
                                            cost_reference.parameters)
            payload = save(chosen)
            matching = payload['coo_bytes'] if plan['storage'] == 'coo' else payload['csr_bytes']
            slack = anchor['payload_bytes'] - matching
            if not 0 <= slack < (12 if plan['storage'] == 'coo' else 8):
                raise RuntimeError('selected pair is not byte-matched')
            decisions.append({'fraction': plan['fraction'], 'storage': plan['storage'], 'cap_percent': percent,
                              'candidate': chosen, **payload, 'anchor_id': anchor['candidate_id'],
                              'matching_bytes': matching, 'anchor_bytes': anchor['payload_bytes'],
                              'slack_bytes': slack, 'validation_quality_pass': passed,
                              'nonzero_variants_available': sum(r['nnz'] * 100 <= percent * cost_reference.parameters
                                                                for r in candidates),
                              'original_joint_coo_pass': budget_for(cost_reference, 'joint', plan['fraction']).allows(
                                  Cost(chosen['parameters'], chosen['payload_bytes'])),
                              'anchor_artifact': anchor_saved['artifact']})
    record = {'dataset': name, 'seed': seed, 'task': dense.task, 'reference_cost': asdict(cost_reference),
              'leaves': m, 'outputs': c, 'fixed_bytes': fixed, 'data_sha256': metadata['sha256'],
              'old_selection_sha256': sha256(old / 'selection.json'),
              'development_split_sha256': sha256(directory / 'development_splits.npz'),
              'validation_reference': previous['validation_reference'], 'full': full_saved,
              'anchors': anchors, 'decisions': decisions, 'allocations': plans,
              'unavailable_allocations': unavailable, 'reuse': stats,
              'candidate_rows': map_rows, 'new_candidate_rows': new_rows}
    write_report(directory / 'selection.json', record)
    emit('map_case_sealed', dataset=name, seed=seed, decisions=len(decisions),
         unique_models=len(saved), unavailable=len(unavailable), **stats)
    return {k: v for k, v in record.items() if k not in ('candidate_rows', 'new_candidate_rows', 'allocations')}


def evaluate_case(record, config, output, data):
    x, y, _, _ = data
    case = f'{record["dataset"]}_{record["seed"]}'
    with np.load(Path(config['source_dir']) / case / 'splits.npz', allow_pickle=False) as f:
        test = f['test'].copy()
    full = ThresholdModel.load(output / record['full']['artifact'])
    design = leaf_design(full.base.leaf_indices(x[test]), len(full.base.coefficients))
    full_quality = calibrated_quality(y[test], full.raw_from_design(design), full.base.task, full.decision_shifts)
    cache, rows = {}, []
    for selection in record['decisions']:
        artifact = selection['artifact']
        if sha256(output / artifact) != selection['sha256']:
            raise RuntimeError('chosen model changed after sealing')
        if artifact not in cache:
            model = ThresholdModel.load(output / artifact)
            cache[artifact] = calibrated_quality(y[test], model.raw_from_design(design), model.base.task,
                                                model.decision_shifts)
        q = cache[artifact]
        rows.append({'dataset': record['dataset'], 'seed': record['seed'], **selection,
                     'test': q, 'test_reference': full_quality, 'test_quality_pass': quality_guard(q, full_quality),
                     'joint_success': selection['validation_quality_pass'] and quality_guard(q, full_quality),
                     'test_macro_retained': q['f1_macro'] / full_quality['f1_macro'],
                     'test_weighted_retained': q['f1_weighted'] / full_quality['f1_weighted'],
                     'test_loss_ratio': q['loss'] / full_quality['loss']})
    emit('map_test_case', dataset=record['dataset'], seed=record['seed'], decisions=len(rows))
    return rows


def summarize(records):
    summaries = []
    for name, fraction, storage in sorted({(r['dataset'], r['fraction'], r['storage']) for r in records}):
        block = [r for r in records if (r['dataset'], r['fraction'], r['storage']) == (name, fraction, storage)]
        baseline = {r['seed']: r for r in block if r['cap_percent'] == 0}
        for percent in (0, 1, 2, 5):
            group = [r for r in block if r['cap_percent'] == percent]
            deltas = [r['test']['f1_macro'] - baseline[r['seed']]['test']['f1_macro'] for r in group]
            summaries.append({'dataset': name, 'fraction': fraction, 'storage': storage, 'cap_percent': percent,
                'splits': len(group), 'validation_passes': sum(r['validation_quality_pass'] for r in group),
                'test_passes': sum(r['test_quality_pass'] for r in group),
                'joint_successes': sum(r['joint_success'] for r in group),
                'nonzero_choices': sum(r['candidate']['nnz'] > 0 for r in group),
                'median_macro_delta': float(np.median(deltas)),
                'wins_ties_losses': [sum(d > 1e-12 for d in deltas), sum(abs(d) <= 1e-12 for d in deltas),
                                     sum(d < -1e-12 for d in deltas)],
                'maximum_slack_bytes': max(r['slack_bytes'] for r in group),
                'original_joint_coo_passes': sum(r['original_joint_coo_pass'] for r in group)})
    return summaries


def rank_residual_map_study(config, report, output):
    started = time.perf_counter()
    def expired(signum, frame):
        raise TimeoutError('H-B1 approved CPU stage time budget exhausted')
    previous_signal = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-k', 'not gpu', 'tests/test_rank_residual_map.py',
                        'tests/test_uniform_threshold_core.py', 'tests/test_sparse_leaf_core.py',
                        'tests/test_prepared_leaf_inference.py'], cwd=ROOT, check=True, timeout=600)
        source_manifest = json.loads((ROOT / 'experiments/sparse_source_manifest.json').read_text())
        verify_sources(Path(config['source_dir']), source_manifest)
        parent_manifest = Path(config['inference_parent_dir']) / 'selection_manifest.json'
        if sha256(parent_manifest) != config['inference_parent_manifest_sha256']:
            raise RuntimeError('old sealed threshold manifest changed')
        old_records = {(r['dataset'], r['seed']): r for r in json.loads(parent_manifest.read_text())['selections']}
        locks = {str(Path(config['source_dir']) / p): h for p, h in source_manifest['files'].items()}
        locks[str(parent_manifest)] = sha256(parent_manifest)
        locks[config['environment_lock']] = config['expected_environment_sha256']
        for name in config['datasets']:
            data_path = Path(config['data_dir']) / (name + '.npz')
            locks[str(data_path)] = sha256(data_path)
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                old = Path(config['inference_parent_dir']) / f'{name}_{seed}'
                for filename in ('selection.json', 'development_splits.npz'):
                    locks[str(old / filename)] = sha256(old / filename)
        write_report(output / 'input_locks.json', locks)
        datasets = {name: read_dataset(name, config) for name in config['datasets']}
        prepared = []
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                prepared.append(prepare_case(name, seed, config, output, datasets[name], old_records[(name, seed)]))
                gc.collect()
        selection_path = output / 'selection_manifest.json'
        write_report(selection_path, {'selections': prepared, 'test_labels_used_in_selection': False,
                                     'tree_fits': 0, 'basis_fits': 0})
        seal = sha256(selection_path)
        check_locks(locks)
        emit('map_all_selections_sealed', sha256=seal, cases=len(prepared), test_scoring_started=False)
        records = []
        for p in prepared:
            if sha256(selection_path) != seal:
                raise RuntimeError('selection manifest changed after first test access')
            records.extend(evaluate_case(p, config, output, datasets[p['dataset']]))
        check_locks(locks)
        summaries = summarize(records)
        report.update(status='complete', tree_fits=0, basis_fits=0, gpu_runs=0, affine_fits=0,
                      records=records, summaries=summaries, cases=len(prepared),
                      input_locks_verified=len(locks), selection_manifest_sha256=seal,
                      threshold_fits=sum(p['reuse']['threshold_fits'] for p in prepared),
                      reused_variants=sum(p['reuse']['reused_variants'] for p in prepared),
                      new_variants=sum(p['reuse']['new_variants'] for p in prepared),
                      selection_policy='old strong rank choice; train-only coordinates; equal calibration; all choices sealed before tests',
                      model_cost='matched COO and prepared CSR separately; original reference denominators unchanged',
                      interpretation='exploratory reused tests; no training compactness, inference speed, or unseen-data guarantee',
                      elapsed_seconds=time.perf_counter()-started)
        write_report(output / 'report.json', report)
        emit('map_summary', tree_fits=0, basis_fits=0, gpu_runs=0, cases=len(prepared),
             decisions=len(records), threshold_fits=report['threshold_fits'],
             reused_variants=report['reused_variants'], new_variants=report['new_variants'],
             elapsed_seconds=report['elapsed_seconds'], summaries=summaries)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.perf_counter()-started)
        write_report(output / 'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_signal)
