"""Approved S0-S3 MediaMill comparison; all derivative choices sealed before test."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
try:
    import resource
except ImportError:  # Windows can inspect and test the orchestration boundary.
    resource = None
import signal
import subprocess
import sys
import time
import numpy as np

from experiments.a100_runner import ROOT, emit, read_dataset, sha256, write_report
from experiments.coverage_calibration_core import (Partition, repair_support, fit_thresholds,
    decisions, decision_series, quality_from_series, point_quality, label_counts)
from experiments.minimum_budget_core import memory_key
from experiments.minimum_budget_study import copy_checked
from experiments.shared_leaf_basis import LeafEnsemble, quality_guard
from experiments.sparse_leaf_core import Budget, Cost, leaf_design
from experiments.sparse_leaf_study import verify_sources
from experiments.stable_budget_core import paired_indices, bootstrap_design, paired_pass_mask
from experiments.stable_budget_study import FrozenDevelopment
from experiments.uniform_threshold_core import ThresholdModel, family_allows

ARMS = ('S0', 'S1', 'S2', 'S3')


def load(path):
    return json.loads(Path(path).read_text())


def scope(rows):
    return next(r for r in rows if r['family'] == 'sparse_05' and r['budget']['kind'] == 'joint'
                and r['budget']['fraction'] == .75)


def primary_choices(control):
    return {policy: scope([r for r in control['selections'] if r['policy'] == policy])
            for policy in ('stable_minimum', 'minimum_bytes', 'maximum_f1')}


def diagnostics(y_train, y_v, partition, full_pred):
    rows = []
    for j in range(y_v.shape[1]):
        n = int(y_v[:, j].sum())
        hits_a = int((full_pred[partition.a, j] & y_v[partition.a, j].astype(bool)).sum())
        hits_b = int((full_pred[partition.b, j] & y_v[partition.b, j].astype(bool)).sum())
        rows.append({'label_index': j, 'train_positive': int(y_train[:, j].sum()),
            'pool_positive': n, 'a_positive': int(y_v[partition.a, j].sum()),
            'b_positive': int(y_v[partition.b, j].sum()), 'positive_quota': min(5, n//2),
            'negative_quota': min(5, (len(y_v)-n)//2),
            'a_negative': int((~y_v[partition.a, j].astype(bool)).sum()),
            'b_negative': int((~y_v[partition.b, j].astype(bool)).sum()),
            'support_status': 'unobserved' if n == 0 else 'impossible_two_parts' if n == 1 else 'low_support' if n < 10 else 'quota_target',
            'full_hits_a': hits_a, 'full_hits_b': hits_b,
            'no_reference_hits_a': hits_a == 0, 'no_reference_hits_b': hits_b == 0})
    return rows


def prepare(seed, config, output, data, control):
    started = time.perf_counter()
    timings = {'support': 0., 'model_and_raw': 0., 'calibration': 0., 'point_scoring': 0.,
               'bootstrap': 0., 'final_calibration': 0.}
    x, y, source_indices, metadata = data
    source = Path(config['source_dir'])/f'mediamill_{seed}'
    previous = Path(config['stable_control_dir'])
    old_dir = previous/f'mediamill_{seed}'
    directory = output/f'mediamill_{seed}'
    directory.mkdir(exist_ok=False)
    for filename, expected in [('development_splits.npz', control['development_split_sha256']),
            ('source_selection.json', control['source_selection_sha256']),
            ('bootstrap_indices.npz', control['bootstrap_indices_sha256']),
            ('candidate_stability.json', control['bootstrap_checks_sha256'])]:
        copy_checked(old_dir/filename, directory/filename, expected)
    parent = load(directory/'source_selection.json')
    with np.load(directory/'development_splits.npz', allow_pickle=False) as f:
        train, a0, b0 = (f[k].copy() for k in ('train', 'calibration', 'selection'))
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['source_indices'], source_indices)
        np.testing.assert_array_equal(f['train'], train)
        np.testing.assert_array_equal(np.sort(f['validation']), np.sort(np.r_[a0, b0]))
        assert len(np.unique(np.r_[train, a0, b0, f['test']])) == len(x)
        # No test labels or test quality are accessed during preparation.
    assert len(a0) == len(b0) == 1200 and metadata['sha256'] == control['data_sha256']
    pool = np.r_[a0, b0]
    y_v, y_train = y[pool].copy(), y[train].copy()
    original = Partition(np.arange(len(a0)), np.arange(len(a0), len(pool)))
    t0 = time.perf_counter()
    repaired = repair_support(y_v, original, config['support_quota'], config['support_seconds'])
    timings['support'] += time.perf_counter()-t0
    partitions = {'original': original, 'controlled': repaired.partition}
    emit('support_checked', seed=seed, status=repaired.status.value, before=repaired.before,
         after=repaired.after, optimal=repaired.optimal)
    np.savez(directory/'partitions.npz', pool=pool, train=train, a0=original.a, b0=original.b,
             a1=repaired.partition.a, b1=repaired.partition.b)
    support = {'status': repaired.status.value, 'before': repaired.before, 'after': repaired.after,
               'optimal': repaired.optimal, 'solver_statuses': repaired.solver_statuses}
    write_report(directory/'support.json', support)
    development = FrozenDevelopment(source, previous, control, x, y, train, pool)
    full_row = control['full']['candidate']
    def model_and_raw(row):
        tick = time.perf_counter()
        result = development.model(row)
        timings['model_and_raw'] += time.perf_counter()-tick
        return result
    full_model, full_raw = model_and_raw(full_row)
    old_choices = primary_choices(control)
    budget = Budget(**old_choices['stable_minimum']['budget'])
    reference_parameters = control['reference_cost']['parameters']
    old_rows = [r for r in parent['validation_candidates'] if family_allows(r, 'sparse_05', reference_parameters)
                and budget.allows(Cost(r['parameters'], r['payload_bytes']))]
    by_id = {r['candidate_id']: r for r in old_rows}
    structures = {}
    for row in old_rows:
        structures.setdefault(development.key(row), row)
    if set(r['threshold_fraction'] for r in old_rows) != {0., .25, .5, 1.}:
        raise RuntimeError('frozen shrinkage grid differs')
    with np.load(directory/'bootstrap_indices.npz', allow_pickle=False) as f:
        old_draws = f['indices'].copy()
    np.testing.assert_array_equal(old_draws, paired_indices(1200, 500, config['bootstrap_seed_base']+seed))
    cross_draws = paired_indices(2400, 500, config['bootstrap_seed_base']+seed)
    np.savez(directory/'cross_bootstrap_indices.npz', indices=cross_draws)
    unique = ['S0', 'S2'] if np.array_equal(repaired.partition.a, original.a) else list(ARMS)
    aliases = {'S1': 'S0', 'S3': 'S2'} if len(unique) == 2 else {}
    specs = {}
    for arm in unique:
        partition = original if arm in ('S0', 'S2') else repaired.partition
        cross = arm in ('S2', 'S3')
        positions = np.arange(2400) if cross else partition.b
        draws = cross_draws if cross else old_draws
        design = bootstrap_design(y_v[positions], 'multilabel', 101,
            np.vstack((np.arange(len(positions), dtype=np.int32), draws)))
        specs[arm] = (partition, cross, positions, design)
    calibrations, full_refs, full_series, arm_rows = {}, {}, {}, {arm: [] for arm in unique}
    fits = 0

    def calibrated(key, raw):
        nonlocal fits
        if key not in calibrations:
            result = {}
            for name, partition in partitions.items():
                if name == 'controlled' and np.array_equal(partition.a, original.a):
                    result[name] = result['original']
                else:
                    tick = time.perf_counter()
                    result[name] = (fit_thresholds(raw[partition.a], y_v[partition.a], config['decision_grid']),
                                    fit_thresholds(raw[partition.b], y_v[partition.b], config['decision_grid']))
                    timings['calibration'] += time.perf_counter()-tick
                    fits += 2
            calibrations[key] = result
        return calibrations[key]

    full_key = development.key(full_row)
    fc = calibrated(full_key, full_raw)
    np.testing.assert_array_equal(fc['original'][0], np.asarray(full_row['decision_shifts'], np.float32))
    for arm, (partition, cross, positions, design) in specs.items():
        ta, tb = fc['original' if arm in ('S0', 'S2') else 'controlled']
        pred = decisions(full_raw, partition, ta, tb, 1., cross)
        series = decision_series(design, full_raw[positions], pred)
        full_series[arm], full_refs[arm] = series, quality_from_series(series)
    assert max(abs(full_refs['S0'][k]-control['validation_reference'][k]) for k in full_refs['S0']) < 1e-12
    # Full OOF predictions carry a threshold trained on the complementary half.
    support['labels'] = diagnostics(y_train, y_v, original,
        decisions(full_raw, original, *fc['original'], 1., True))
    support['controlled_labels'] = diagnostics(y_train, y_v, repaired.partition,
        decisions(full_raw, repaired.partition, *fc['controlled'], 1., True))
    write_report(directory/'support.json', support)
    for number, (key, structural_row) in enumerate(structures.items(), 1):
        _, raw = model_and_raw(structural_row)
        cal = calibrated(key, raw)
        variants = [r for r in old_rows if development.key(r) == key]
        for old in variants:
            alpha = old['threshold_fraction']
            np.testing.assert_array_equal((alpha*cal['original'][0]).astype(np.float32),
                                          np.asarray(old['decision_shifts'], np.float32))
            for arm, (partition, cross, positions, _) in specs.items():
                pred = decisions(raw, partition, *cal['original' if arm in ('S0', 'S2') else 'controlled'], alpha, cross)
                tick = time.perf_counter()
                q = point_quality(y_v[positions], raw[positions], pred)
                timings['point_scoring'] += time.perf_counter()-tick
                if arm == 'S0' and max(abs(q[k]-old['validation'][k]) for k in q) > 1e-12:
                    raise RuntimeError('S0 point replay changed: '+old['candidate_id'])
                arm_rows[arm].append({**{k: old[k] for k in ('candidate_id', 'kind', 'rank', 'selector', 'nnz',
                    'parameters', 'payload_bytes', 'threshold_fraction')}, 'validation': q})
        if number % 50 == 0:
            emit('cross_points_prepared', seed=seed, structures=number, total=len(structures), threshold_fits=fits)
    old_checks = {r['candidate_id']: r for r in load(directory/'candidate_stability.json')['checks']}
    choices, checks = {}, {}
    for arm, (partition, cross, positions, design) in specs.items():
        qualified = sorted((r for r in arm_rows[arm] if quality_guard(r['validation'], full_refs[arm])), key=memory_key)
        arm_checks, chosen = [], None
        for row in qualified:
            _, raw = model_and_raw(by_id[row['candidate_id']])
            cal = calibrated(development.key(row), raw)['original' if arm in ('S0', 'S2') else 'controlled']
            pred = decisions(raw, partition, *cal, row['threshold_fraction'], cross)
            tick = time.perf_counter()
            series = decision_series(design, raw[positions], pred)
            timings['bootstrap'] += time.perf_counter()-tick
            assert max(abs(quality_from_series(series)[k]-row['validation'][k]) for k in row['validation']) < 1e-12
            mask = paired_pass_mask(series, full_series[arm])[1:]
            check = {'candidate_id': row['candidate_id'], 'successes': int(mask.sum()),
                     'pass_mask_hex': np.packbits(mask, bitorder='little').tobytes().hex()}
            arm_checks.append(check)
            if arm == 'S0':
                previous_check = old_checks[row['candidate_id']]
                assert (check['successes'], check['pass_mask_hex']) == (previous_check['successes'], previous_check['pass_mask_hex'])
            if check['successes'] >= 450:
                chosen = {**row, 'bootstrap_successes': check['successes'], 'pass_mask_hex': check['pass_mask_hex']}
                break
        choices[arm] = {'arm': arm, 'candidate': chosen, 'qualified_count': len(qualified),
            'checked_count': len(arm_checks), 'status': 'selected' if chosen else 'no_stable_candidate' if qualified else 'no_qualified_candidate'}
        checks[arm] = arm_checks
        emit('cross_choice', seed=seed, arm=arm, candidate=chosen['candidate_id'] if chosen else None,
             qualified=len(qualified), checked=len(arm_checks), successes=chosen['bootstrap_successes'] if chosen else None)
    assert choices['S0']['candidate']['candidate_id'] == old_choices['stable_minimum']['candidate']['candidate_id']
    assert choices['S0']['candidate']['bootstrap_successes'] == old_choices['stable_minimum']['bootstrap_successes']
    for alias, canonical in aliases.items():
        choices[alias] = {**choices[canonical], 'arm': alias, 'alias_of': canonical}
    write_report(directory/'development.json', {'rows': arm_rows, 'references': full_refs, 'checks': checks, 'aliases': aliases})
    keys = list(calibrations)
    arrays = {f'{i}_{name}_{side}': calibrations[key][name][side] for i, key in enumerate(keys)
              for name in ('original', 'controlled') for side in (0, 1)}
    np.savez(directory/'calibration_thresholds.npz', **arrays)
    write_report(directory/'calibration_keys.json', {'keys': keys})
    model_records, saved_refit = [], {}

    def save_model(role, row, final):
        nonlocal fits
        cid = row['candidate_id']
        model, raw = model_and_raw(row)
        if final:
            if cid not in saved_refit:
                tick = time.perf_counter()
                thresholds = (row['threshold_fraction']*fit_thresholds(raw, y_v, config['decision_grid'])).astype(np.float32)
                timings['final_calibration'] += time.perf_counter()-tick
                fits += 1
                model = replace(model, decision_shifts=thresholds)
                path = directory/f'{cid}-refit.npz'
                model.save(path)
                saved_refit[cid] = (model, path)
            model, path = saved_refit[cid]
        else:
            old = control['full'] if role == 'F0' else old_choices[role.removeprefix('legacy_')]
            path = directory/f'{cid}-legacy.npz'
            copy_checked(previous/old['artifact'], path, old['sha256'])
        assert (model.parameter_count, model.payload_bytes) == (row['parameters'], row['payload_bytes'])
        result = {'role': role, 'candidate_id': cid, 'rank': row['rank'], 'nnz': row['nnz'],
            'selector': row['selector'], 'threshold_fraction': row['threshold_fraction'],
            'parameters': model.parameter_count, 'payload_bytes': model.payload_bytes,
            'artifact': str(path.relative_to(output)).replace('\\', '/'), 'sha256': sha256(path),
            'decision_shifts': model.decision_shifts.tolist(), 'final_refit': final}
        model_records.append(result)
        return result

    save_model('F0', full_row, False)
    save_model('FV', full_row, True)
    for policy, choice in old_choices.items():
        save_model('legacy_'+policy, choice['candidate'], False)
        save_model('refit_'+policy, choice['candidate'], True)
    for arm in ARMS:
        if choices[arm]['candidate']:
            model = save_model(arm, by_id[choices[arm]['candidate']['candidate_id']], True)
            choices[arm]['model_sha256'] = model['sha256']
    row = {'dataset': 'mediamill', 'seed': seed, 'data_sha256': metadata['sha256'], 'support': support,
        'choices': choices, 'models': model_records, 'full_references': full_refs,
        'reference_cost': control['reference_cost'], 'threshold_fits': fits,
        'source_development_sha256': sha256(directory/'source_selection.json'),
        'partitions_sha256': sha256(directory/'partitions.npz'),
        'development_sha256': sha256(directory/'development.json'),
        'calibration_thresholds_sha256': sha256(directory/'calibration_thresholds.npz'),
        'calibration_keys_sha256': sha256(directory/'calibration_keys.json'),
        'timings_seconds': timings, 'prepare_seconds': time.perf_counter()-started}
    write_report(directory/'selection.json', row)
    for cache in (development.base, development.order, development.raw_model):
        cache.cache_clear()
    emit('cross_split_sealed', seed=seed, seconds=row['prepare_seconds'], threshold_fits=fits)
    return row


def evaluate(selection, config, output, data):
    x, y, _, _ = data
    seed = selection['seed']
    source = Path(config['source_dir'])/f'mediamill_{seed}'
    dense = LeafEnsemble.load(source/'dense_teacher.npz')
    with np.load(source/'splits.npz', allow_pickle=False) as f:
        test, train = f['test'].copy(), f['train'].copy()
    design = leaf_design(dense.leaf_indices(x[test]), len(dense.coefficients))
    rare = y[train].mean(axis=0) < .01
    records, per_label = [], []
    for record in selection['models']:
        path = output/record['artifact']
        assert sha256(path) == record['sha256']
        model = ThresholdModel.load(path)
        raw = model.raw_from_design(design)
        pred = raw >= model.decision_shifts
        q = point_quality(y[test], raw, pred)
        tp, fp, fn, f1 = label_counts(y[test], pred)
        records.append({**record, 'seed': seed, 'test': q, 'rare_macro': float(f1[rare].mean()),
            'payload_reduction': 1-record['payload_bytes']/selection['reference_cost']['payload_bytes'],
            'parameter_reduction': 1-record['parameters']/selection['reference_cost']['parameters']})
        for j in range(101):
            per_label.append({'seed': seed, 'role': record['role'], 'label_index': j,
                'rare': bool(rare[j]), 'support': int(y[test, j].sum()), 'tp': int(tp[j]),
                'fp': int(fp[j]), 'fn': int(fn[j]), 'f1': float(f1[j])})
    by_role = {r['role']: r for r in records}
    for record in records:
        record['test_pass_F0'] = quality_guard(record['test'], by_role['F0']['test'])
        record['test_pass_FV'] = quality_guard(record['test'], by_role['FV']['test'])
    decisions = []
    for arm, choice in selection['choices'].items():
        record = by_role.get(arm)
        decisions.append({'seed': seed, 'arm': arm, 'selected': record is not None,
            'test_pass_F0': record['test_pass_F0'] if record else False,
            'test_pass_FV': record['test_pass_FV'] if record else False,
            'false_accept': bool(record and not record['test_pass_F0']),
            'candidate_id': record['candidate_id'] if record else None,
            'payload_bytes': record['payload_bytes'] if record else None,
            'test': record['test'] if record else None, 'alias_of': choice.get('alias_of')})
    return records, per_label, decisions


def coverage_calibration_study(config, report, output):
    if config['datasets'] != ['mediamill'] or config['seeds_by_dataset']['mediamill'] != [1104, 1114, 1124, 1134, 1144]:
        raise ValueError('approved five-MediaMill scope changed')
    if (config['support_quota'], config['bootstrap_repeats'], config['minimum_bootstrap_successes']) != (5, 500, 450):
        raise ValueError('approved coverage/bootstrap contract changed')
    started = time.perf_counter()
    def expired(signum, frame):
        raise TimeoutError('approved one-hour limit exhausted')
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(config['walltime_seconds'])
    report.update(status='running', teacher_fits=0, basis_fits=0, gpu_runs=0)
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_coverage_calibration_core.py',
            'tests/test_stable_budget_core.py', 'tests/test_minimum_budget_core.py',
            'tests/test_uniform_threshold_core.py', 'tests/test_output_calibration_core.py',
            'tests/test_sparse_leaf_core.py', 'tests/test_compactness_core.py'], cwd=ROOT, check=True)
        prior = Path(config['stable_control_dir'])
        manifest_path = prior/'selection_manifest.json'
        assert sha256(manifest_path) == config['stable_control_manifest_sha256']
        controls = {(r['dataset'], r['seed']): r for r in load(manifest_path)['selections']}
        verify_sources(Path(config['source_dir']), load(ROOT/'experiments/sparse_source_manifest.json'))
        data = read_dataset('mediamill', config)
        selections = [prepare(seed, config, output, data, controls[('mediamill', seed)])
                      for seed in config['seeds_by_dataset']['mediamill']]
        seal_path = output/'selection_manifest.json'
        write_report(seal_path, {'test_labels_used_in_selection': False,
            'protocol_sha256': sha256(ROOT/'experiments/coverage_calibration_protocol.md'), 'selections': selections})
        seal = sha256(seal_path)
        emit('cross_all_selections_sealed', sha256=seal, test_scoring_started=False, decisions=20)
        records, labels, decisions_out = [], [], []
        for selection in selections:
            assert sha256(seal_path) == seal
            r, l, d = evaluate(selection, config, output, data)
            records.extend(r)
            labels.extend(l)
            decisions_out.extend(d)
            gc.collect()
        # Historical test evidence is read only after the complete new seal.
        old_report = load(prior/'report.json')
        for seed in config['seeds_by_dataset']['mediamill']:
            full = next(r for r in old_report['full_records'] if (r['dataset'], r['seed']) == ('mediamill', seed))
            previous = primary_choices({'selections': [r for r in old_report['records'] if r['dataset'] == 'mediamill' and r['seed'] == seed]})
            lookup = {r['role']: r for r in records if r['seed'] == seed}
            for role, prior_row in [('F0', full), *[('legacy_'+p, r) for p, r in previous.items()]]:
                assert lookup[role]['sha256'] == prior_row['sha256']
                assert max(abs(lookup[role]['test'][k]-prior_row['test'][k]) for k in prior_row['test']) < 1e-12
        summaries = {arm: {'selected': sum(d['selected'] for d in decisions_out if d['arm'] == arm),
            'successes_F0': sum(d['test_pass_F0'] for d in decisions_out if d['arm'] == arm),
            'successes_FV': sum(d['test_pass_FV'] for d in decisions_out if d['arm'] == arm),
            'false_accepts': sum(d['false_accept'] for d in decisions_out if d['arm'] == arm)} for arm in ARMS}
        pairs = []
        for seed in config['seeds_by_dataset']['mediamill']:
            new = next(d for d in decisions_out if (d['seed'], d['arm']) == (seed, 'S3'))
            old = next(r for r in records if (r['seed'], r['role']) == (seed, 'refit_maximum_f1'))
            safe = new['test_pass_F0'] and old['test_pass_F0']
            pairs.append({'seed': seed, 'both_safe': safe,
                'byte_saving': 1-new['payload_bytes']/old['payload_bytes'] if new['selected'] else None})
        safe = [p['byte_saving'] for p in pairs if p['both_safe']]
        promising = bool(summaries['S3']['successes_F0'] > summaries['S0']['successes_F0']
            and summaries['S3']['false_accepts'] <= summaries['S0']['false_accepts'] and safe and np.median(safe) > 0)
        report.update(status='complete', records=records, per_label=labels, decisions=decisions_out,
            summaries=summaries, safe_pairs=pairs, primary_promising=promising,
            threshold_fits=sum(s['threshold_fits'] for s in selections), elapsed_seconds=time.perf_counter()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
            selection_manifest_sha256=seal,
            support_statuses={str(s['seed']): s['support']['status'] for s in selections})
        write_report(output/'report.json', report)
        emit('cross_summary', summaries=summaries, primary_promising=promising, safe_pairs=pairs,
             support_statuses=report['support_statuses'], seconds=report['elapsed_seconds'],
             threshold_fits=report['threshold_fits'], teacher_fits=0, basis_fits=0, gpu_runs=0)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
