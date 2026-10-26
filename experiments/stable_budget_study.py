"""Frozen-model policy comparison: 500 paired draws, >=450 joint passes."""
from dataclasses import replace
from functools import lru_cache
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, read_dataset, sha256, write_report
from experiments.minimum_budget_study import copy_checked, test_minimum
from experiments.shared_leaf_basis import LeafEnsemble, compress, quality_guard
from experiments.sparse_leaf_core import (Budget, contrast, coordinate_scores, leaf_design,
    score_order, training_weights)
from experiments.sparse_leaf_study import base_at_rank, verify_sources
from experiments.stable_budget_core import (paired_indices, bootstrap_design, bootstrap_quality,
    paired_pass_mask, stable_minimum, stability_requests, StableMinimum)
from experiments.uniform_threshold_core import ThresholdModel
from experiments.uniform_threshold_study import FAMILIES, materialize_uniform

POLICIES = ('stable_minimum', 'minimum_bytes', 'maximum_f1')


class FrozenDevelopment:
    """I/O/reconstruction shell; bounded caches, no fit/calibration operations."""
    def __init__(self, source, control_root, control, x, y, train, select):
        self.dense, self.reference = (LeafEnsemble.load(source/f'{n}.npz') for n in ('dense_teacher', 'reference'))
        with np.load(source/'calibration_diagnostics.npz', allow_pickle=False) as f:
            self.mean, self.basis = f['mean'].copy(), f['basis'].copy()
        self.full = compress(self.dense, self.mean, self.basis)
        self.train = leaf_design(self.dense.leaf_indices(x[train]), len(self.dense.coefficients))
        self.selection = leaf_design(self.dense.leaf_indices(x[select]), len(self.dense.coefficients))
        self.teacher = contrast(self.train@self.dense.coefficients+self.dense.bias, self.dense.task)
        self.latent = self.train@self.full.coefficients
        self.frequency = np.asarray(self.train.mean(axis=0)).ravel()
        self.y_train = y[train].copy()
        self.control_root = control_root
        choices = [control['full'], *[s for s in control['selections'] if s['candidate']]]
        self.known = {s['candidate']['candidate_id']: s for s in choices}
        self.known_structure = {self.key(s['candidate']): s for s in choices}
        self.base = lru_cache(maxsize=8)(self._base)
        self.order = lru_cache(maxsize=8)(self._order)
        self.raw_model = lru_cache(maxsize=6)(self._raw_model)

    @staticmethod
    def key(row):
        return row['kind'], row['rank'], row['selector'], row['nnz']

    def _base(self, rank):
        return base_at_rank(self.dense, self.mean, self.basis, rank, self.full)

    def _order(self, rank, selector):
        base = self.base(rank)
        residual = (contrast(self.dense.coefficients, self.dense.task)
            - contrast(base.coefficients.astype(float)@base.basis.T, self.dense.task)).astype(np.float32)
        error = self.teacher-contrast(self.latent[:, :rank]@base.basis.T+base.bias, self.dense.task)
        weights = training_weights(self.y_train, self.teacher, self.dense.task, selector, .5)
        return score_order(coordinate_scores(self.train, residual, error, selector, weights), residual)

    def _raw_model(self, key):
        kind, rank, selector, count = key
        if key in self.known_structure:
            known = self.known_structure[key]
            path = self.control_root/known['artifact']
            if sha256(path) != known['sha256']:
                raise RuntimeError('frozen control artifact changed')
            model = ThresholdModel.load(path)
        else:
            chosen = {'kind': kind, 'rank': rank, 'selector': selector, 'nnz': count,
                      'decision_shifts': [0.]*len(self.dense.bias)}
            plan = {'full': self.full, 'bases': {rank: self.base(rank)},
                    'frequency': self.frequency, 'orders': {}}
            if count:
                plan['orders'][(rank, selector)] = self.order(rank, selector)
            model = materialize_uniform(chosen, plan, self.dense, self.reference, self.mean, self.basis)
        return model, model.raw_from_design(self.selection)

    def model(self, row):
        model, raw = self.raw_model(self.key(row))
        return replace(model, decision_shifts=np.asarray(row['decision_shifts'], np.float32)), raw


def prepare_stable(name, seed, config, output, data, control, input_manifest):
    started = time.perf_counter()
    x, y, source_indices, metadata = data
    source = Path(config['source_dir'])/f'{name}_{seed}'
    parent_dir = Path(config['minimum_parent_dir'])/f'{name}_{seed}'
    directory = output/f'{name}_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    parent = json.loads((parent_dir/'selection.json').read_text())
    copy_checked(parent_dir/'selection.json', directory/'source_selection.json',
                 input_manifest['files'][f'{name}_{seed}/selection.json'])
    copy_checked(parent_dir/'development_splits.npz', directory/'development_splits.npz', control['development_split_sha256'])
    with np.load(directory/'development_splits.npz', allow_pickle=False) as f:
        train, cal, select = (f[k].copy() for k in ('train', 'calibration', 'selection'))
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices, f['source_indices'])
        np.testing.assert_array_equal(train, f['train'])
        np.testing.assert_array_equal(np.sort(np.r_[cal, select]), np.sort(f['validation']))
        if len(np.unique(np.r_[train, cal, select, f['test']])) != len(x):
            raise RuntimeError('partition mismatch')
    if metadata['sha256'] != control['data_sha256']:
        raise RuntimeError('dataset differs from frozen control')
    development = FrozenDevelopment(source, Path(config['minimum_control_dir']), control, x, y, train, select)
    draw_seed = config['bootstrap_seed_base']+seed
    indices = paired_indices(len(select), config['bootstrap_repeats'], draw_seed)
    np.savez(directory/'bootstrap_indices.npz', indices=indices)
    # Column zero evaluates the original sample, followed by exactly 500 draws.
    # This is a correctness check, not a 501st bootstrap repeat.
    design = bootstrap_design(y[select], development.dense.task, len(development.dense.bias),
                              np.vstack((np.arange(len(select), dtype=np.int32), indices)))
    full_choice = dict(control['full'])
    _, reference_raw = development.model(full_choice['candidate'])
    reference_series = bootstrap_quality(design, reference_raw, full_choice['candidate']['decision_shifts'])
    if any(abs(float(getattr(reference_series, k)[0])-parent['validation_reference'][k]) > 1e-7
           for k in ('loss', 'f1_macro', 'f1_weighted')):
        raise RuntimeError('frozen full-reference validation quality changed')
    np.savez(directory/'bootstrap_reference.npz', **{k: getattr(reference_series, k) for k in
                                                  ('loss', 'f1_macro', 'f1_weighted')})
    counts, checks = {}, {}

    def score(row):
        cid = row['candidate_id']
        if cid in counts:
            return counts[cid]
        model, raw = development.model(row)
        if (model.parameter_count, model.payload_bytes) != (row['parameters'], row['payload_bytes']):
            raise RuntimeError('candidate cost changed')
        series = bootstrap_quality(design, raw, model.decision_shifts)
        point = {k: float(getattr(series, k)[0]) for k in ('loss', 'f1_macro', 'f1_weighted')}
        if any(abs(point[k]-row['validation'][k]) > 1e-7 for k in point):
            raise RuntimeError('frozen validation metric reconstruction mismatch: '+cid)
        mask = paired_pass_mask(series, reference_series)[1:]
        count = int(mask.sum())
        counts[cid] = count
        checks[cid] = {'candidate_id': cid, 'successes': count, 'repeats': len(mask), 'point_quality': point,
                       'pass_mask_hex': np.packbits(mask, bitorder='little').tobytes().hex()}
        return count

    scopes = [s for s in control['selections'] if s['policy'] == 'minimum_bytes']
    assert len(scopes) == 36
    pending = set(range(len(scopes)))
    rows = parent['validation_candidates']
    reference = parent['validation_reference']
    reference_parameters = parent['reference_cost']['parameters']
    for row, memberships in stability_requests(rows, reference, reference_parameters, scopes):
        relevant = pending.intersection(memberships)
        if relevant:
            if score(row) >= config['minimum_bootstrap_successes']:
                pending.difference_update(relevant)
            if len(checks) % 50 == 0:
                emit('stable_preparing', dataset=name, seed=seed, checked=len(checks), unresolved=len(pending))
    stable = []
    for scope in scopes:
        choice = stable_minimum(rows, reference, reference_parameters, scope['family'], Budget(**scope['budget']), counts)
        selected = isinstance(choice, StableMinimum)
        stable.append({'policy': 'stable_minimum', 'budget': scope['budget'], 'family': scope['family'],
            'residual_fraction_cap': scope['residual_fraction_cap'],
            'status': 'selected' if selected else ('no_qualified_candidate' if not choice.point_qualified_count else 'no_stable_candidate'),
            'validation_quality_pass': selected,
            'candidate': choice.candidate.copy() if selected else None,
            'point_qualified_count': choice.point_qualified_count, 'checked_count': choice.checked_count,
            'bootstrap_successes': choice.successes if selected else None,
            'bootstrap_repeats': config['bootstrap_repeats'], 'bootstrap_pass': selected})
        if selected and choice.candidate['payload_bytes'] < scope['candidate']['payload_bytes']:
            raise RuntimeError('stable minimum is cheaper than point minimum')
    decisions = [*stable, *[dict(s) for s in control['selections']]]
    for d in decisions:
        if d['candidate'] is not None:
            d['bootstrap_successes'] = score(d['candidate'])
            d['bootstrap_repeats'] = config['bootstrap_repeats']
            d['bootstrap_pass'] = bool(d['validation_quality_pass'] and d['bootstrap_successes'] >= 450)
    saved = {}
    for d in [full_choice, *[d for d in decisions if d['candidate']]]:
        row = d['candidate']
        cid = row['candidate_id']
        if cid not in saved:
            path = directory/f'{cid}.npz'
            if cid in development.known:
                old = development.known[cid]
                copy_checked(Path(config['minimum_control_dir'])/old['artifact'], path, old['sha256'])
            else:
                development.model(row)[0].save(path)
            loaded = ThresholdModel.load(path)
            assert (loaded.parameter_count, loaded.payload_bytes) == (row['parameters'], row['payload_bytes'])
            np.testing.assert_array_equal(loaded.decision_shifts, np.asarray(row['decision_shifts'], np.float32))
            saved[cid] = {'artifact': str(path.relative_to(output)).replace('\\', '/'), 'sha256': sha256(path),
                          'file_bytes': path.stat().st_size}
        d.update(saved[cid])
    for i, scope in enumerate(scopes):
        replay = stable_minimum(rows, reference, reference_parameters, scope['family'], Budget(**scope['budget']), counts)
        assert (replay.candidate if isinstance(replay, StableMinimum) else None) == stable[i]['candidate']
    check_path = directory/'candidate_stability.json'
    write_report(check_path, {'bootstrap_seed': draw_seed, 'repeats': 500, 'minimum_successes': 450,
        'checks': list(checks.values()), 'all_cheaper_eligible_candidates_checked': True})
    record = {'dataset': name, 'seed': seed, 'task': development.dense.task, 'data_sha256': metadata['sha256'],
        'reference_cost': parent['reference_cost'], 'validation_reference': reference, 'full': full_choice,
        'development_split_sha256': sha256(directory/'development_splits.npz'),
        'source_selection_sha256': sha256(directory/'source_selection.json'),
        'bootstrap_indices_sha256': sha256(directory/'bootstrap_indices.npz'),
        'bootstrap_reference_sha256': sha256(directory/'bootstrap_reference.npz'),
        'bootstrap_checks_sha256': sha256(check_path), 'bootstrap_seed': draw_seed,
        'bootstrap_repeats': 500, 'minimum_bootstrap_successes': 450,
        'calibration_rows': len(cal), 'selection_rows': len(select), 'selections': decisions,
        'prepare_seconds': time.perf_counter()-started, 'checked_candidates': len(checks)}
    write_report(directory/'selection.json', record)
    emit('stable_split_sealed', dataset=name, seed=seed, checked_candidates=len(checks),
         stable_models=sum(d['candidate'] is not None for d in stable), scopes=len(stable), seconds=record['prepare_seconds'])
    # Release bounded lru wrappers, which otherwise retain this I/O shell.
    for cache in (development.base, development.order, development.raw_model):
        cache.cache_clear()
    return record


def aggregate_stable(records):
    summaries = []
    for dataset, kind, fraction, family, policy in sorted({(r['dataset'], r['budget']['kind'], r['budget']['fraction'],
                                                           r['family'], r['policy']) for r in records}):
        rows = [r for r in records if (r['dataset'], r['budget']['kind'], r['budget']['fraction'], r['family'], r['policy'])
                == (dataset, kind, fraction, family, policy)]
        selected = [r for r in rows if r['candidate']]
        summaries.append({'dataset': dataset, 'kind': kind, 'fraction': fraction, 'family': family, 'policy': policy,
            'splits': len(rows), 'selected_count': len(selected), 'missing_count': len(rows)-len(selected),
            'joint_successes': sum(r['joint_success'] for r in rows),
            'bootstrap_passes': sum(r.get('bootstrap_pass', False) for r in rows),
            'mean_macro_selected': float(np.mean([r['test']['f1_macro'] for r in selected])) if selected else None,
            'median_payload_reduction_selected': float(np.median([r['payload_reduction'] for r in selected])) if selected else None,
            'median_parameter_reduction_selected': float(np.median([r['parameter_reduction'] for r in selected])) if selected else None})
    primary, pairs = [], []
    for name in sorted({r['dataset'] for r in records}):
        rows = [r for r in records if r['dataset'] == name and r['budget']['kind'] == 'joint'
                and r['budget']['fraction'] == .75 and r['family'] == 'sparse_05']
        lookup = {(r['seed'], r['policy']): r for r in rows}
        for seed in sorted({r['seed'] for r in rows}):
            new = lookup[(seed, 'stable_minimum')]
            for policy in ('minimum_bytes', 'maximum_f1'):
                old = lookup[(seed, policy)]
                comparable = bool(new['candidate'] and old['candidate'])
                safe = new['joint_success'] and old['joint_success']
                pairs.append({'dataset': name, 'seed': seed, 'control_policy': policy,
                    'stable_success': new['joint_success'], 'control_success': old['joint_success'],
                    'stable_selected': new['candidate'] is not None, 'comparable': comparable, 'both_safe': safe,
                    'stable_bytes': new['candidate']['payload_bytes'] if new['candidate'] else None,
                    'control_bytes': old['candidate']['payload_bytes'] if old['candidate'] else None,
                    'byte_saving_vs_control': 1-new['candidate']['payload_bytes']/old['candidate']['payload_bytes'] if comparable else None})
        quality_pairs = [p for p in pairs if p['dataset'] == name and p['control_policy'] == 'maximum_f1' and p['both_safe']]
        primary.append({'dataset': name, 'splits': 5,
            'policies': {p: {'selected': sum(r['candidate'] is not None for r in rows if r['policy'] == p),
                             'joint_successes': sum(r['joint_success'] for r in rows if r['policy'] == p)} for p in POLICIES},
            'safe_pairs_vs_max_f1': len(quality_pairs),
            'safe_byte_wins_vs_max_f1': sum(p['byte_saving_vs_control'] > 0 for p in quality_pairs),
            'median_byte_saving_safe_vs_max_f1': float(np.median([p['byte_saving_vs_control'] for p in quality_pairs])) if quality_pairs else None})
    return summaries, primary, pairs


def stable_budget_study(config, report, output):
    if (config['bootstrap_repeats'], config['minimum_bootstrap_successes'], config['quality_tolerance']) != (500, 450, .05):
        raise ValueError('approved 500/90/5 contract changed')
    def expired(signum, frame):
        raise TimeoutError('approved one-hour CPU comparison exhausted')
    old_signal = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    started = time.perf_counter()
    report.update(status='running', teacher_fits=0, basis_fits=0, threshold_fits=0, gpu_runs=0,
        selection_policy='original point guards and >=450/500 joint paired-object bootstrap passes; smallest bytes; explicit missing',
        interpretation='exploratory reused tests; bootstrap stability is not a future-success probability',
        model_cost='unchanged exact-reference denominators, all thresholds/COO indices/tree structures charged')
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_stable_budget_core.py',
            'tests/test_minimum_budget_core.py', 'tests/test_uniform_threshold_core.py',
            'tests/test_output_calibration_core.py', 'tests/test_sparse_leaf_core.py',
            'tests/test_compactness_core.py'], cwd=ROOT, check=True)
        verify_sources(Path(config['source_dir']), json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text()))
        input_manifest = json.loads((ROOT/'experiments/minimum_source_manifest.json').read_text())
        for relative, expected in input_manifest['files'].items():
            if sha256(Path(config['minimum_parent_dir'])/relative) != expected:
                raise RuntimeError('frozen development input changed: '+relative)
        control_path = Path(config['minimum_control_dir'])/'selection_manifest.json'
        if sha256(control_path) != config['minimum_control_manifest_sha256']:
            raise RuntimeError('frozen control choices changed')
        copy_checked(control_path, output/'control_manifest.json', config['minimum_control_manifest_sha256'])
        controls = {(r['dataset'], r['seed']): r for r in json.loads(control_path.read_text())['selections']}
        datasets = {name: read_dataset(name, config) for name in config['datasets']}
        selections = []
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                selections.append(prepare_stable(name, seed, config, output, datasets[name], controls[(name, seed)], input_manifest))
                gc.collect()
        seal_path = output/'selection_manifest.json'
        write_report(seal_path, {'test_labels_used_in_selection': False, 'bootstrap_repeats': 500,
            'minimum_bootstrap_successes': 450, 'quality_tolerance': .05,
            'control_manifest_sha256': config['minimum_control_manifest_sha256'], 'selections': selections})
        seal = sha256(seal_path)
        emit('stable_all_selections_sealed', sha256=seal, decisions=sum(len(r['selections'])+1 for r in selections),
             bootstrap_repeats=500, minimum_successes=450, test_scoring_started=False)
        full_records, records = [], []
        for selection in selections:
            if sha256(seal_path) != seal:
                raise RuntimeError('sealed decisions changed')
            full, rows = test_minimum(selection, config, output, datasets[selection['dataset']])
            full_records.append(full)
            records.extend(rows)
        # Access prior test metrics only after all new choices have been sealed.
        old_report = json.loads((Path(config['minimum_control_dir'])/'report.json').read_text())
        old = {(r['dataset'], r['seed'], r['policy'], r['family'], r['budget']['kind'], r['budget']['fraction']): r
               for r in old_report['records']}
        for r in records:
            if r['policy'] != 'stable_minimum':
                previous = old[(r['dataset'], r['seed'], r['policy'], r['family'], r['budget']['kind'], r['budget']['fraction'])]
                if (r['candidate'], r['test'], r['joint_success']) != (previous['candidate'], previous['test'], previous['joint_success']):
                    raise RuntimeError('prior policy replay changed')
        summaries, primary, pairs = aggregate_stable(records)
        report.update(status='complete', full_records=full_records, records=records, summaries=summaries,
            primary=primary, pairs=pairs, selection_manifest_sha256=seal,
            control_results_replayed=1080, elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        emit('stable_summary', primary=primary,
            policies={p: {'choices': sum(r['policy'] == p for r in records),
                'missing': sum(r['policy'] == p and r['candidate'] is None for r in records),
                'joint_successes': sum(r['policy'] == p and r['joint_success'] for r in records)} for p in POLICIES})
        emit('stable_complete', elapsed_seconds=report['elapsed_seconds'], teacher_fits=0, basis_fits=0,
             threshold_fits=0, gpu_runs=0, control_results_replayed=1080, selection_manifest_sha256=seal)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_signal)
