"""Calibrated full/rank/sparse comparison on frozen teachers; no new tree fits."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scipy import sparse

from experiments.a100_runner import ROOT, emit, read_dataset, sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, compress, quality_guard, probabilities
from experiments.sparse_leaf_core import (Cost, Budget, budget_for, contrast, leaf_design,
    coordinate_scores, score_order, training_weights, residual_capacity, select_candidate, make_sparse_model)
from experiments.sparse_leaf_study import base_at_rank, verify_sources
from experiments.output_calibration_core import calibrated_quality
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants, choice_key, family_allows

FAMILIES = ('rank_only', 'sparse_01', 'sparse_02', 'sparse_05')
CAPS = {'sparse_01': 1, 'sparse_02': 2, 'sparse_05': 5}


def uniform_plan(dense, reference, mean, basis, train_leaves, train_target,
                 calibration_leaves, calibration_target, selection_leaves, selection_target,
                 reference_cost, config, progress=None, build_selections=True):
    """Development-only pure planner: test rows, labels and metrics cannot enter."""
    m, c = dense.coefficients.shape
    designs = {k: leaf_design(v, m) for k, v in
               [('train', train_leaves), ('calibration', calibration_leaves), ('selection', selection_leaves)]}
    full = compress(dense, mean, basis)
    latents = {k: d @ full.coefficients for k, d in designs.items()}
    train_teacher = contrast(designs['train'] @ dense.coefficients + dense.bias, dense.task)
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    budgets = [budget_for(reference_cost, k, f) for k in config['budget_kinds'] for f in config['budget_fractions']]
    rows, bases, orders, full_rows = [], {}, {}, []

    def append_variants(destination, kind, rank, selector, count, cost, raw_cal, raw_sel):
        variants = threshold_variants(raw_cal, calibration_target, raw_sel, selection_target,
                                     dense.task, config['decision_grid'], config['decision_shrinkage'])
        for v in variants:
            suffix = str(v['threshold_fraction']).replace('.', 'p')
            destination.append({'candidate_id': f'{kind}-r{rank}-{selector}-{count}-t{suffix}',
                                'kind': kind, 'rank': rank, 'selector': selector, 'nnz': count,
                                'parameters': cost.parameters, 'payload_bytes': cost.payload_bytes, **v})

    def reference_raw(design):
        raw = design @ reference.coefficients
        if reference.representation == 'reference_class':
            raw = np.column_stack((raw, np.zeros(len(raw))))
        return raw + reference.bias

    reference_cal = reference_raw(designs['calibration'])
    reference_sel = reference_raw(designs['selection'])
    np.testing.assert_allclose(probabilities(reference_sel, dense.task),
        probabilities(designs['selection'] @ dense.coefficients + dense.bias, dense.task), atol=2e-5, rtol=0)
    append_variants(full_rows, 'full', reference.coefficients.shape[1], 'full', 0,
                    Cost(reference.parameter_count+c, reference.payload_bytes+4*c), reference_cal, reference_sel)
    full_choice = min(full_rows, key=choice_key)
    validation_reference = full_choice['validation']
    rank_costs = {r: Cost((m+c)*r+2*c, dense.payload_bytes-4*m*c+4*(m+c)*r+4*c)
                  for r in range(1, basis.shape[1]+1)}
    allowed = [r for r, cost in rank_costs.items() if any(b.allows(cost) for b in budgets)]
    if not allowed and build_selections:
        raise RuntimeError('no calibrated rank fits the fixed budgets')
    for rank in allowed:
        base = base_at_rank(dense, mean, basis, rank, full)
        append_variants(rows, 'rank_only', rank, 'rank_only', 0, rank_costs[rank],
                        latents['calibration'][:, :rank] @ base.basis.T + base.bias,
                        latents['selection'][:, :rank] @ base.basis.T + base.bias)
        if progress and (rank % 10 == 0 or rank == allowed[-1]):
            progress('rank', rank, len(rows))
    sparse_ranks = set()
    for budget in budgets:
        feasible = [r for r in allowed if budget.allows(rank_costs[r])]
        if not feasible:
            if build_selections:
                raise RuntimeError('no calibrated rank fits a requested budget')
            continue
        maximum = max(feasible)
        sparse_ranks.update(max(1, int(maximum*f)) for f in config['base_rank_fractions'])
        sparse_ranks.add(maximum)
        for percent in CAPS.values():
            count = percent*reference_cost.parameters//100
            fits = [r for r in feasible if budget.allows(Cost(rank_costs[r].parameters+count,
                                                             rank_costs[r].payload_bytes+12*count))]
            if fits:
                sparse_ranks.add(max(fits))
    weights = {s: training_weights(train_target, train_teacher, dense.task, s, config['margin_scale'])
               for s in config['tiny_selectors']}
    for rank in sorted(sparse_ranks):
        base = base_at_rank(dense, mean, basis, rank, full)
        bases[rank] = base
        residual = (contrast(dense.coefficients, dense.task)
                    - contrast(base.coefficients.astype(float) @ base.basis.T, dense.task)).astype(np.float32)
        error = train_teacher-contrast(latents['train'][:, :rank] @ base.basis.T+base.bias, dense.task)
        for selector in config['tiny_selectors']:
            scores = coordinate_scores(designs['train'], residual, error, selector, weights[selector])
            order = score_order(scores, residual)
            orders[(rank, selector)] = order
            counts = {min(len(order), residual_capacity(rank_costs[rank], b),
                          int(np.floor((percent*reference_cost.parameters//100)*f)))
                      for b in budgets if b.allows(rank_costs[rank]) for percent in CAPS.values()
                      for f in config['residual_capacity_fractions']}
            for count in sorted(counts-{0}):
                flat = order[:count]
                rr, cc = (flat//c).astype(np.int32), (flat%c).astype(np.int32)
                values = residual.ravel()[flat]
                correction = sparse.coo_matrix((values, (rr, cc)), shape=(m, c)).tocsr()
                shift = np.bincount(cc, weights=values.astype(float)*frequency[rr], minlength=c)
                bias = (base.bias.astype(float)-shift).astype(np.float32)
                cost = Cost(rank_costs[rank].parameters+count, rank_costs[rank].payload_bytes+12*count)
                append_variants(rows, 'sparse', rank, selector, count, cost,
                    latents['calibration'][:, :rank] @ base.basis.T+bias+(designs['calibration']@correction).toarray(),
                    latents['selection'][:, :rank] @ base.basis.T+bias+(designs['selection']@correction).toarray())
        if progress:
            progress('sparse', rank, len(rows))
    selections = []
    for budget in (budgets if build_selections else []):
        for family in FAMILIES:
            candidates = [r for r in rows if family_allows(r, family, reference_cost.parameters)]
            chosen, passed = select_candidate(candidates, validation_reference, budget)
            selections.append({'budget': asdict(budget), 'family': family,
                               'residual_fraction_cap': CAPS.get(family, 0)/100,
                               'validation_quality_pass': passed, 'candidate': chosen.copy()})
    return {'rows': rows, 'full_candidates': full_rows, 'full_choice': full_choice,
            'validation_reference': validation_reference, 'selections': selections,
            'bases': bases, 'orders': orders, 'full': full, 'frequency': frequency}


def materialize_uniform(chosen, plan, dense, reference, mean, basis):
    shifts = np.array(chosen['decision_shifts'], np.float32)
    empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    if chosen['kind'] == 'full':
        return ThresholdModel(reference, *empty, shifts)
    rank = chosen['rank']
    base = plan['bases'].get(rank)
    if base is None:
        base = base_at_rank(dense, mean, basis, rank, plan['full'])
    if not chosen['nnz']:
        return ThresholdModel(base, *empty, shifts)
    residual = (contrast(dense.coefficients, dense.task)
                - contrast(base.coefficients.astype(float)@base.basis.T, dense.task)).astype(np.float32)
    leaf = make_sparse_model(base, residual, plan['orders'][(rank, chosen['selector'])],
                             chosen['nnz'], plan['frequency'])
    return ThresholdModel(leaf.base, leaf.rows, leaf.columns, leaf.values, shifts)


def prepare_uniform(name, seed, config, output, data, previous):
    x, y, source_indices, metadata = data
    started = time.perf_counter()
    source = Path(config['source_dir'])/f'{name}_{seed}'
    directory = output/f'{name}_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices, f['source_indices'])
        old_train, old_validation = f['train'].copy(), f['validation'].copy()
    split_path = Path(config['previous_dir'])/f'{name}_{seed}/development_splits.npz'
    if sha256(split_path) != previous['development_split_sha256']:
        raise RuntimeError('previous calibration/selection split changed')
    with np.load(split_path, allow_pickle=False) as f:
        train, cal, select = (f[k].copy() for k in ('train', 'calibration', 'selection'))
    np.testing.assert_array_equal(train, old_train)
    np.testing.assert_array_equal(np.sort(np.r_[cal, select]), np.sort(old_validation))
    if len(np.intersect1d(cal, select)) or len(np.intersect1d(train, np.r_[cal, select])):
        raise RuntimeError('development sets overlap')
    np.savez(directory/'development_splits.npz', train=train, calibration=cal, selection=select)
    dense, reference = (LeafEnsemble.load(source/f'{n}.npz') for n in ('dense_teacher', 'reference'))
    with np.load(source/'calibration_diagnostics.npz', allow_pickle=False) as f:
        mean, basis = f['mean'].copy(), f['basis'].copy()
    leaves = {k: dense.leaf_indices(x[v]) for k, v in [('train', train), ('calibration', cal), ('selection', select)]}
    plan = uniform_plan(dense, reference, mean, basis, leaves['train'], y[train], leaves['calibration'],
                        y[cal], leaves['selection'], y[select], Cost(reference.parameter_count, reference.payload_bytes),
                        config, lambda stage, rank, count: emit('uniform_preparing', dataset=name, seed=seed,
                                                               stage=stage, rank=rank, candidates=count))
    design = leaf_design(leaves['selection'], len(dense.coefficients))
    saved = {}
    all_choices = [{'candidate': plan['full_choice']}, *plan['selections']]
    for selected in all_choices:
        chosen = selected['candidate']
        cid = chosen['candidate_id']
        if cid not in saved:
            model = materialize_uniform(chosen, plan, dense, reference, mean, basis)
            assert model.parameter_count == chosen['parameters'] and model.payload_bytes == chosen['payload_bytes']
            path = directory/f'{cid}.npz'
            model.save(path)
            loaded = ThresholdModel.load(path)
            np.testing.assert_array_equal(loaded.raw_from_design(design), model.raw_from_design(design))
            q = calibrated_quality(y[select], loaded.raw_from_design(design), dense.task, loaded.decision_shifts)
            if any(abs(q[k]-chosen['validation'][k]) > 1e-7 for k in q):
                raise RuntimeError('serialization changed selected quality')
            saved[cid] = {'artifact': str(path.relative_to(output)).replace('\\', '/'),
                          'sha256': sha256(path), 'file_bytes': path.stat().st_size}
        selected.update(saved[cid])
    record = {'dataset': name, 'seed': seed, 'task': dense.task, 'data_sha256': metadata['sha256'],
              'development_split_sha256': sha256(directory/'development_splits.npz'),
              'original_development_split_sha256': previous['development_split_sha256'],
              'reference_cost': {'parameters': reference.parameter_count, 'payload_bytes': reference.payload_bytes},
              'calibration_rows': len(cal), 'selection_rows': len(select), 'full': all_choices[0],
              'validation_reference': plan['validation_reference'], 'full_candidates': plan['full_candidates'],
              'selections': plan['selections'], 'validation_candidates': plan['rows'],
              'prepare_seconds': time.perf_counter()-started}
    write_report(directory/'selection.json', record)
    emit('uniform_split_sealed', dataset=name, seed=seed, candidates=len(plan['rows']),
         decisions=len(plan['selections'])+1, unique_models=len(saved), seconds=record['prepare_seconds'])
    return {k: v for k, v in record.items() if k not in ('validation_candidates', 'full_candidates')}


def test_uniform(record, config, output, data):
    x, y, _, _ = data
    source = Path(config['source_dir'])/f'{record["dataset"]}_{record["seed"]}'
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        test = f['test'].copy()
    dense = LeafEnsemble.load(source/'dense_teacher.npz')
    design = leaf_design(dense.leaf_indices(x[test]), len(dense.coefficients))
    raw_teacher = design@dense.coefficients+dense.bias
    original_quality = calibrated_quality(y[test], raw_teacher, dense.task, np.empty(0, np.float32))
    cache = {}
    def replay(selected):
        path = output/selected['artifact']
        if sha256(path) != selected['sha256']:
            raise RuntimeError('sealed selected artifact changed')
        if selected['artifact'] not in cache:
            model = ThresholdModel.load(path)
            raw = model.raw_from_design(design)
            cache[selected['artifact']] = (calibrated_quality(y[test], raw, dense.task, model.decision_shifts),
                                           calibrated_quality(y[test], raw, dense.task, np.empty(0, np.float32)))
        return cache[selected['artifact']]
    full_quality, full_uncalibrated = replay(record['full'])
    full_record = {'dataset': record['dataset'], 'seed': record['seed'], 'task': dense.task,
                   **record['full'], 'test': full_quality, 'test_uncalibrated': full_uncalibrated,
                   'original_teacher_test': original_quality}
    records = []
    for selected in record['selections']:
        q, uncalibrated = replay(selected)
        chosen = selected['candidate']
        assert Budget(**selected['budget']).allows(Cost(chosen['parameters'], chosen['payload_bytes']))
        assert family_allows(chosen, selected['family'], record['reference_cost']['parameters'])
        records.append({'dataset': record['dataset'], 'seed': record['seed'], 'task': dense.task,
                        **selected, 'test': q, 'test_uncalibrated': uncalibrated, 'test_reference': full_quality,
                        'test_quality_pass': quality_guard(q, full_quality),
                        'joint_success': selected['validation_quality_pass'] and quality_guard(q, full_quality),
                        'macro_retained': q['f1_macro']/full_quality['f1_macro'],
                        'loss_ratio': q['loss']/full_quality['loss'],
                        'actual_residual_fraction': chosen['nnz']/record['reference_cost']['parameters'],
                        'parameter_reduction': 1-chosen['parameters']/record['reference_cost']['parameters'],
                        'payload_reduction': 1-chosen['payload_bytes']/record['reference_cost']['payload_bytes']})
    emit('uniform_test_finished', dataset=record['dataset'], seed=record['seed'], decisions=len(records)+1)
    return full_record, records


def summarize_uniform(records):
    summaries = []
    for dataset, kind, fraction in sorted({(r['dataset'], r['budget']['kind'], r['budget']['fraction']) for r in records}):
        group = [r for r in records if (r['dataset'], r['budget']['kind'], r['budget']['fraction']) == (dataset, kind, fraction)]
        control = {r['seed']: r for r in group if r['family'] == 'rank_only'}
        for family in FAMILIES:
            rows = [r for r in group if r['family'] == family]
            deltas = [r['test']['f1_macro']-control[r['seed']]['test']['f1_macro'] for r in rows]
            summaries.append({'dataset': dataset, 'kind': kind, 'fraction': fraction, 'family': family,
                'splits': len(rows), 'quality_passes': sum(r['test_quality_pass'] for r in rows),
                'joint_successes': sum(r['joint_success'] for r in rows),
                'nonzero_residual_models': sum(r['candidate']['nnz'] > 0 for r in rows),
                'mean_macro': float(np.mean([r['test']['f1_macro'] for r in rows])),
                'mean_full_macro': float(np.mean([r['test_reference']['f1_macro'] for r in rows])),
                'median_macro_retained': float(np.median([r['macro_retained'] for r in rows])),
                'median_parameter_reduction': float(np.median([r['parameter_reduction'] for r in rows])),
                'median_payload_reduction': float(np.median([r['payload_reduction'] for r in rows])),
                'mean_macro_delta_vs_rank_only': float(np.mean(deltas)),
                'wins_ties_losses_vs_rank_only': [sum(d > 1e-12 for d in deltas), sum(abs(d) <= 1e-12 for d in deltas), sum(d < -1e-12 for d in deltas)]})
    return summaries


def uniform_threshold_study(config, report, output):
    def expired(signum, frame):
        raise TimeoutError('one-hour approved CPU comparison exhausted')
    previous_signal = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    started = time.perf_counter()
    report.update(status='running', teacher_fits=0, gpu_runs=0, affine_fits=0,
                  selection_policy='same C-threshold calibration for full/rank/sparse; train-only coordinates; all selections sealed before test',
                  model_cost='original exact-reference denominators; C stored threshold floats for every model; full outside caps',
                  interpretation='Exploratory old test splits; post-training compression, not fewer trained trees')
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_uniform_threshold_core.py',
                        'tests/test_output_calibration_core.py', 'tests/test_sparse_leaf_core.py',
                        'tests/test_compactness_core.py'], cwd=ROOT, check=True)
        manifest = json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text())
        verify_sources(Path(config['source_dir']), manifest)
        previous_manifest = Path(config['previous_dir'])/'selection_manifest.json'
        if sha256(previous_manifest) != config['previous_manifest_sha256']:
            raise RuntimeError('previous sealed selection manifest changed')
        previous = {(r['dataset'], r['seed']): r for r in json.loads(previous_manifest.read_text())['selections']}
        datasets = {name: read_dataset(name, config) for name in config['datasets']}
        selections = []
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                selections.append(prepare_uniform(name, seed, config, output, datasets[name], previous[(name, seed)]))
                gc.collect()
        seal_path = output/'selection_manifest.json'
        write_report(seal_path, {'source_run': manifest['source_run'], 'previous_manifest_sha256': config['previous_manifest_sha256'],
                                'test_labels_used_in_selection': False, 'selections': selections})
        seal = sha256(seal_path)
        emit('uniform_all_selections_sealed', sha256=seal, decisions=sum(len(r['selections'])+1 for r in selections), test_scoring_started=False)
        full_records, records = [], []
        for selected in selections:
            if sha256(seal_path) != seal:
                raise RuntimeError('selection manifest changed')
            full_record, rows = test_uniform(selected, config, output, datasets[selected['dataset']])
            full_records.append(full_record)
            records.extend(rows)
        summaries = summarize_uniform(records)
        report.update(status='complete', full_records=full_records, records=records, summaries=summaries,
                      selection_manifest_sha256=seal, source_manifest_sha256=sha256(ROOT/'experiments/sparse_source_manifest.json'),
                      elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        for name in config['datasets']:
            emit('uniform_comparison_summary', dataset=name,
                 full_macro=[r['test']['f1_macro'] for r in full_records if r['dataset'] == name],
                 summaries=[s for s in summaries if s['dataset'] == name])
        emit('uniform_complete', decisions=len(records)+len(full_records), teacher_fits=0, gpu_runs=0,
             elapsed_seconds=report['elapsed_seconds'], selection_manifest_sha256=seal)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_signal)
