"""Frozen Birds selection: development barrier, then exploratory test audit.

This module never trains trees, fits a basis/residual, or fits thresholds.
Existing checkpoint arrays and existing threshold vectors are copied verbatim.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.birds_selection_core import (POOLS, as_quality, quality_vector, contributions,
    weighted_metrics, passes, pool_indices, select, bootstrap_weights, split_half_weights,
    leave_one_out_weights)
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.uniform_threshold_core import ThresholdModel
from experiments.output_calibration_core import calibrated_quality
from experiments.joint_budget_audit import array_cost, raw, csv_table
from experiments.label_budget_diagnostics import per_label


@contextmanager
def forbid_fitting():
    """Fail closed if future edits accidentally add an optimization call."""
    from contextlib import ExitStack
    targets = ('experiments.a100_runner.build_model', 'experiments.leaf_budget_runtime.fit_dense',
               'experiments.parallel_leaf_worker.fit_increment',
               'experiments.output_calibration_core.fit_decision_shifts',
               'experiments.uniform_threshold_core.fit_decision_shifts',
               'experiments.shared_leaf_basis.functional_basis')
    def forbidden(*args, **kwargs):
        raise RuntimeError('fitting is forbidden in the frozen Birds selection study')
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, forbidden))
        yield


def locked(path, expected, seal=None):
    path = Path(path)
    if sha256(path) != expected:
        raise RuntimeError('frozen Birds input changed: '+str(path))
    if seal is not None:
        seal[str(path)] = expected
    return path


def locked_json(path, expected, seal=None):
    path = locked(path, expected, seal)
    return json.loads(path.read_text()), path


def assert_close(actual, expected, tolerance=1e-7):
    error = float(np.max(np.abs(quality_vector(actual)-quality_vector(expected))))
    if error > tolerance:
        raise RuntimeError('frozen Birds selection metric mismatch')
    return error


def budgets_match(model, row, budget):
    if ((model.parameter_count, model.payload_bytes) != (row['parameters'], row['payload_bytes'])
            or model.parameter_count > budget['parameters'] or model.payload_bytes > budget['payload_bytes']):
        raise RuntimeError('frozen candidate violates actual joint budget')


def label_rows(y, ref_raw, ref_shifts, model_raw, model_shifts, supports, identity):
    full = per_label(y, ref_raw >= ref_shifts, 'multilabel', y.shape[1])
    current = per_label(y, model_raw >= model_shifts, 'multilabel', y.shape[1])
    return [dict(**identity, label=j, **{k:int(v[j]) for k,v in supports.items()},
        positive=int(full['support'][j]), full_tp=int(full['tp'][j]), tp=int(current['tp'][j]),
        full_fp=int(full['fp'][j]), fp=int(current['fp'][j]), full_fn=int(full['fn'][j]), fn=int(current['fn'][j]),
        full_f1=float(full['f1'][j]), f1=float(current['f1'][j]),
        delta_macro_contribution=float((current['f1'][j]-full['f1'][j])/y.shape[1])) for j in range(y.shape[1])]


def selection_diagnostics(rows, point, bootstrap, loo, choose_half, assess_half, selection_ids, target):
    """All metrics include the frozen full reference in column zero."""
    candidates, reference = point[1:], point[0]
    good_draws = passes(bootstrap[:, 1:], bootstrap[:, :1])
    rates = good_draws.mean(axis=0)
    decisions, stability, leaveouts, halfrows = [], [], [], []
    for pool in POOLS:
        indices = pool_indices(rows, pool)
        for rule in ('point', 'paired90'):
            decision = select(rows, candidates, reference, indices, None if rule == 'point' else rates)
            row = rows[decision['index']]
            decisions.append(dict(pool=pool, rule=rule, pool_size=len(indices), **decision,
                validation=as_quality(candidates[decision['index']]), reference=as_quality(reference),
                payload_bytes=row['payload_bytes'], parameters=row['parameters'],
                paired_pass_rate=float(rates[decision['index']])))
        original = decisions[-2]
        for draw, values in enumerate(bootstrap):
            chosen = select(rows, values[1:], values[0], indices)
            stability.append(dict(pool=pool, draw=draw, **chosen))
        for removed, values in enumerate(loo):
            chosen = select(rows, values[1:], values[0], indices)
            q = values[1+original['index']]
            leaveouts.append(dict(pool=pool, removed_local=removed, removed_source=int(selection_ids[removed]),
                removed_positive_labels=np.flatnonzero(target[removed]).tolist(), **chosen,
                choice_changed=chosen['candidate_id'] != original['candidate_id'],
                decision_changed=chosen['accepted'] != original['accepted'],
                original_candidate_pass=bool(passes(q, values[0])),
                original_candidate_quality=as_quality(q), reference=as_quality(values[0])))
        for direction, (first, second) in enumerate(zip(choose_half, assess_half)):
            chosen = select(rows, first[1:], first[0], indices)
            test_pass = bool(passes(second[1+chosen['index']], second[0]))
            pool_good = passes(second[1:], second[0])[indices]
            halfrows.append(dict(pool=pool, direction=direction, **chosen, assessment_pass=test_pass,
                false_accept=bool(chosen['accepted'] and not test_pass),
                rejected_good_candidate=bool(not chosen['accepted'] and test_pass),
                missed_available=bool(not chosen['accepted'] and pool_good.any()),
                assessment_quality=as_quality(second[1+chosen['index']]),
                assessment_reference=as_quality(second[0])))
    return decisions, rates, stability, leaveouts, halfrows


def birds_selection_development(config, report, output):
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_birds_selection_core.py',
                    'tests/test_birds_selection_study.py'], cwd=ROOT, check=True, timeout=300)
    seal = {}
    data = expanded_manifest(config)
    locked(config['expanded_manifest'], config['expanded_manifest_sha256'], seal)
    static, static_path = locked_json(config['joint_development_manifest'], config['joint_development_manifest_sha256'], seal)
    strict, _ = locked_json(config['birds_strict_manifest'], config['birds_strict_manifest_sha256'], seal)
    if static['test_evaluated'] or strict['test_evaluated']:
        raise RuntimeError('selection sources must be development artifacts')
    x, y = load_expanded('birds', data)
    if y.shape != (645, 19):
        raise RuntimeError('frozen Birds output domain changed')
    locked(data['datasets']['birds']['path'], data['datasets']['birds']['sha256'], seal)
    paths, cases, diagnostics = [], [], []
    max_error = 0.
    with forbid_fitting(), threadpool_limits(limits=1):
        for seed in config['birds_seeds']:
            check_deadline(config, 300)
            case = next(c for c in data['cases'] if c['dataset'] == 'birds' and c['seed'] == seed)
            roles = load_roles(case, len(x))
            locked(case['split_file'], case['split_sha256'], seal)
            info = next(r for r in static['records'] if r['dataset'] == 'birds' and r['seed'] == seed)
            full_path = locked(static_path.parent/info['full']['artifact'], info['full']['sha256'], seal)
            full = ThresholdModel.load(full_path)
            if (array_cost(full_path) != (full.parameter_count, full.payload_bytes)
                    or (full.parameter_count-19, full.payload_bytes-4*19) !=
                       (info['reference_cost']['parameters'], info['reference_cost']['payload_bytes'])):
                raise RuntimeError('Birds full-reference denominator changed')
            case_dir = output/f'birds_{seed}'
            case_dir.mkdir()
            full_copy = case_dir/'full.npz'
            shutil.copyfile(full_path, full_copy)
            seal[str(full_copy)] = sha256(full_copy)
            ids = roles['selection']
            ref_raw = raw(full, x[ids])
            ref_q = calibrated_quality(y[ids], ref_raw, 'multilabel', full.decision_shifts)
            max_error = max(max_error, assert_close(ref_q, info['validation_reference']))
            supports = {f'{role}_positive':y[roles[role]].sum(axis=0) for role in
                        ('train', 'calibration', 'control', 'selection')}
            w = bootstrap_weights(len(ids), 500, seed+700000)
            a, b = split_half_weights(len(ids), 100, seed+800000)
            jack = leave_one_out_weights(len(ids))
            cases.append(dict(seed=seed, full_artifact=full_copy.relative_to(output).as_posix(),
                full_sha256=sha256(full_copy), full_source=str(full_path), reference_cost=info['reference_cost'],
                reference_quality=info['validation_reference'], role_sizes={k:len(v) for k,v in roles.items()},
                split_file=case['split_file'], split_sha256=case['split_sha256'],
                label_supports={k:v.astype(int).tolist() for k,v in supports.items()}))
            for allocation in ('periodic_fixed', 'joint_adaptive'):
                source = next(e for e in strict['records'] if e['trajectory']['dataset'] == 'birds'
                              and e['trajectory']['seed'] == seed and e['allocation_policy'] == allocation)
                trajectory, root = source['trajectory'], Path(source['artifact_root'])
                directory = root/f'birds_{seed}'/trajectory['method']
                trajectory_path = directory/'trajectory.json'
                seal[str(trajectory_path)] = sha256(trajectory_path)
                if json.loads(trajectory_path.read_text()) != trajectory:
                    raise RuntimeError('trajectory differs from the frozen strict manifest')
                rows = trajectory['choice']['candidates']
                budget = trajectory['budget']
                if (budget['parameters'], budget['payload_bytes']) != (
                    int(.75*info['reference_cost']['parameters']), int(.75*info['reference_cost']['payload_bytes'])):
                    raise RuntimeError('Birds hard budget changed')
                target_dir = case_dir/allocation
                target_dir.mkdir()
                frozen, signals, shifts, cache = [], [], [], {}
                for row in rows:
                    block = row['block']
                    if block not in cache:
                        cycle = next(c for c in trajectory['cycles'] if c['block'] == block)
                        checkpoint = locked(root/cycle['artifact'], cycle['sha256'], seal)
                        base_model = ThresholdModel.load(checkpoint)
                        if array_cost(checkpoint) != (base_model.parameter_count, base_model.payload_bytes):
                            raise RuntimeError('checkpoint actual-array cost mismatch')
                        cache[block] = (base_model, raw(base_model, x[ids]), checkpoint)
                    model, signal, checkpoint = cache[block]
                    # The vector already existed in the old candidate manifest.
                    model = replace(model, decision_shifts=np.array(row['decision_shifts'], np.float32))
                    budgets_match(model, row, budget)
                    quality = calibrated_quality(y[ids], signal, 'multilabel', model.decision_shifts)
                    max_error = max(max_error, assert_close(quality, row['validation']))
                    artifact = target_dir/(row['candidate_id']+'.npz')
                    model.save(artifact)
                    seal[str(artifact)] = sha256(artifact)
                    frozen.append(dict(**row, artifact=artifact.relative_to(output).as_posix(),
                        sha256=sha256(artifact), source_checkpoint=str(checkpoint),
                        checkpoint_sha256=sha256(checkpoint)))
                    signals.append(signal)
                    shifts.append(model.decision_shifts)
                    diagnostics.extend(label_rows(y[ids], ref_raw, full.decision_shifts, signal,
                        model.decision_shifts, supports, dict(seed=seed, allocation=allocation,
                        candidate_id=row['candidate_id'], role='selection')))
                zs = np.stack([ref_raw, *signals])
                ts = np.stack([full.decision_shifts, *shifts])
                evidence = contributions(y[ids], zs, ts)
                measured = weighted_metrics(evidence, np.ones((1, len(ids)), np.int64))[0]
                frozen_metrics = np.array([quality_vector(info['validation_reference']),
                                          *[quality_vector(r['validation']) for r in rows]])
                max_error = max(max_error, float(np.max(np.abs(measured-frozen_metrics))))
                if max_error > 1e-7:
                    raise RuntimeError('vectorized selection metrics differ')
                bootstrap = weighted_metrics(evidence, w)
                jack_q = weighted_metrics(evidence, jack)
                qa, qb = weighted_metrics(evidence, a), weighted_metrics(evidence, b)
                decisions, rates, stability, leaveouts, halfrows = selection_diagnostics(
                    rows, frozen_metrics, bootstrap, jack_q, qa, qb, ids, y[ids])
                baseline = next(d for d in decisions if d['pool'] == 'all_all' and d['rule'] == 'point')
                old = trajectory['choice']
                if (baseline['candidate_id'] != old['candidate']['candidate_id'] or
                    baseline['accepted'] != (old['state'] == 'accepted')):
                    raise RuntimeError('original broad Birds selector was not reproduced')
                for i, row in enumerate(frozen):
                    row['bootstrap_pass_rate'] = float(rates[i])
                cache_path = target_dir/'selection_evidence.npz'
                np.savez(cache_path, target=y[ids], raw=zs, shifts=ts, selection_ids=ids,
                    bootstrap_weights=w, half_choose_weights=a, half_assess_weights=b,
                    leave_one_out_weights=jack)
                seal[str(cache_path)] = sha256(cache_path)
                details = target_dir/'internal_diagnostics.json'
                write_report(details, dict(bootstrap_choices=stability, leave_one_out=leaveouts,
                    split_half=halfrows, test_evaluated=False))
                seal[str(details)] = sha256(details)
                path = dict(seed=seed, allocation=allocation, budget=budget, candidates=frozen,
                    decisions=decisions, original_choice=old['candidate']['candidate_id'], original_state=old['state'],
                    evidence=cache_path.relative_to(output).as_posix(),
                    diagnostics=details.relative_to(output).as_posix(),
                    unique_decisions=len({(z >= t).tobytes() for z,t in zip(signals, shifts)}))
                paths.append(path)
                for d in decisions:
                    emit('birds_development_decision', seed=seed, allocation=allocation, **d)
                emit('birds_internal_summary', seed=seed, allocation=allocation,
                    selection_samples=len(ids), candidates=len(rows), unique_decisions=path['unique_decisions'],
                    pools=[dict(pool=pool, bootstrap_acceptances=sum(r['accepted'] for r in stability if r['pool']==pool),
                        bootstrap_repeats=500, loo_choice_changes=sum(r['choice_changed'] for r in leaveouts if r['pool']==pool),
                        loo_decision_changes=sum(r['decision_changed'] for r in leaveouts if r['pool']==pool),
                        half_directions=200, half_acceptances=sum(r['accepted'] for r in halfrows if r['pool']==pool),
                        half_false_acceptances=sum(r['false_accept'] for r in halfrows if r['pool']==pool),
                        half_missed_available=sum(r['missed_available'] for r in halfrows if r['pool']==pool)) for pool in POOLS])
    csv_table(output/'selection_per_label.csv', diagnostics)
    write_report(output/'sealed_inputs.json', seal)
    manifest = dict(version=1, status='birds_selection_development_complete', cases=cases, paths=paths,
        config=config, test_evaluated=False, tree_fits=0, threshold_fits=0, basis_fits=0,
        input_seal_sha256=sha256(output/'sealed_inputs.json'),
        maximum_selection_replay_error=max_error, elapsed_seconds=time.perf_counter()-started)
    write_report(output/'selection_manifest.json', manifest)
    report.update(manifest)
    emit('birds_development_complete', cases=len(cases), trajectories=len(paths), decisions=sum(len(p['decisions']) for p in paths),
        models=sum(len(p['candidates']) for p in paths), tree_fits=0, threshold_fits=0, basis_fits=0,
        maximum_selection_replay_error=max_error, manifest_sha256=sha256(output/'selection_manifest.json'),
        input_seal_sha256=manifest['input_seal_sha256'], test_evaluated=False)


def replay_development_choices(manifest, root):
    """Audit cached row statistics/decisions before any test data access."""
    maximum = 0.
    for path in manifest['paths']:
        with np.load(root/path['evidence'], allow_pickle=False) as f:
            evidence = contributions(f['target'], f['raw'], f['shifts'])
            measured = weighted_metrics(evidence, np.ones((1,len(f['target'])), np.int64))[0]
            boot = weighted_metrics(evidence, f['bootstrap_weights'])
        case = next(c for c in manifest['cases'] if c['seed'] == path['seed'])
        frozen = np.array([quality_vector(case['reference_quality']),
                          *[quality_vector(r['validation']) for r in path['candidates']]])
        maximum = max(maximum, float(np.max(np.abs(measured-frozen))))
        if maximum > 1e-7:
            raise RuntimeError('cached selection contributions changed')
        rates = passes(boot[:,1:], boot[:,:1]).mean(axis=0)
        for decision in path['decisions']:
            actual = select(path['candidates'], frozen[1:], frozen[0],
                pool_indices(path['candidates'], decision['pool']), None if decision['rule']=='point' else rates)
            if any(actual[k] != decision[k] for k in actual):
                raise RuntimeError('frozen Birds final decision changed')
    return maximum


def birds_selection_test(config, report, output):
    started = time.perf_counter()
    manifest, source = locked_json(config['birds_selection_manifest'], config['birds_selection_manifest_sha256'])
    if manifest['status'] != 'birds_selection_development_complete' or manifest['test_evaluated']:
        raise RuntimeError('successful non-test Birds development must complete first')
    root = source.parent
    seal, _ = locked_json(root/'sealed_inputs.json', manifest['input_seal_sha256'])
    for path, digest in seal.items():
        locked(path, digest)
    if manifest['config']['expected_environment_sha256'] != report['environment_sha256']:
        raise RuntimeError('Birds test environment changed')
    with forbid_fitting(), threadpool_limits(limits=1):
        error = replay_development_choices(manifest, root)
        # This is the first test scoring in this study, after all decisions froze.
        data = expanded_manifest(manifest['config'])
        x, y = load_expanded('birds', data)
        records, candidate_rows, labels, summaries = [], [], [], []
        for case in manifest['cases']:
            check_deadline(config, 180)
            seed = case['seed']
            original = next(c for c in data['cases'] if c['dataset']=='birds' and c['seed']==seed)
            roles = load_roles(original, len(x))
            full = ThresholdModel.load(locked(root/case['full_artifact'], case['full_sha256']))
            ref_raw = raw(full, x[roles['test']])
            ref_q = calibrated_quality(y[roles['test']], ref_raw, 'multilabel', full.decision_shifts)
            supports = {k:np.asarray(v) for k,v in case['label_supports'].items()}
            for path in (p for p in manifest['paths'] if p['seed'] == seed):
                qualities, success = [], []
                for row in path['candidates']:
                    model = ThresholdModel.load(locked(root/row['artifact'], row['sha256']))
                    budgets_match(model, row, path['budget'])
                    if array_cost(root/row['artifact']) != (model.parameter_count, model.payload_bytes):
                        raise RuntimeError('test candidate cost changed')
                    z = raw(model, x[roles['test']])
                    q = calibrated_quality(y[roles['test']], z, 'multilabel', model.decision_shifts)
                    passed = bool(passes(quality_vector(q), quality_vector(ref_q)))
                    qualities.append(q); success.append(passed)
                    identity = dict(seed=seed, allocation=path['allocation'], candidate_id=row['candidate_id'])
                    candidate_rows.append(dict(**identity, test_quality_pass=passed, **q,
                        reference_loss=ref_q['loss'], reference_macro=ref_q['f1_macro'], reference_weighted=ref_q['f1_weighted'],
                        selection_quality_pass=bool(passes(quality_vector(row['validation']),quality_vector(case['reference_quality']))),
                        paired_pass_rate=row['bootstrap_pass_rate'], bytes=row['payload_bytes'], parameters=row['parameters'],
                        block=row['block'], threshold_fraction=row['threshold_fraction']))
                    labels.extend(label_rows(y[roles['test']], ref_raw, full.decision_shifts, z,
                        model.decision_shifts, supports, dict(**identity, role='test')))
                for d in path['decisions']:
                    indices = pool_indices(path['candidates'], d['pool'])
                    any_good = any(success[i] for i in indices)
                    q, passed = qualities[d['index']], success[d['index']]
                    result = dict(seed=seed, allocation=path['allocation'], **d,
                        test_quality_pass=passed, joint_success=bool(d['accepted'] and passed),
                        false_accept=bool(d['accepted'] and not passed),
                        rejected_good_candidate=bool(not d['accepted'] and passed),
                        missed_available=bool(not d['accepted'] and any_good),
                        pool_has_test_feasible=any_good, broad_has_test_feasible=any(success),
                        lost_pool_coverage=bool(any(success) and not any_good),
                        bytes_fraction=d['payload_bytes']/case['reference_cost']['payload_bytes'],
                        parameters_fraction=d['parameters']/case['reference_cost']['parameters'],
                        test_quality=q, test_reference=ref_q,
                        loss_ratio=q['loss']/ref_q['loss'], macro_retained=q['f1_macro']/ref_q['f1_macro'],
                        weighted_retained=q['f1_weighted']/ref_q['f1_weighted'])
                    records.append(result)
                    emit('birds_test_decision', **result)
        for pool in POOLS:
            for rule in ('point', 'paired90'):
                for allocation in ('periodic_fixed', 'joint_adaptive', 'both'):
                    selected = [r for r in records if r['pool']==pool and r['rule']==rule
                                and (allocation=='both' or r['allocation']==allocation)]
                    summary = dict(pool=pool, rule=rule, allocation=allocation, decisions=len(selected),
                        independent_outer_splits=3, **{key:sum(r[key] for r in selected) for key in
                        ('accepted','joint_success','false_accept','rejected_good_candidate','missed_available',
                         'pool_has_test_feasible','lost_pool_coverage')})
                    summaries.append(summary)
                    emit('birds_test_summary', **summary)
    flat = [{**{k:v for k,v in r.items() if not isinstance(v,dict)},
             **{'test_'+k:v for k,v in r['test_quality'].items()}} for r in records]
    csv_table(output/'decisions.csv', flat)
    csv_table(output/'candidate_test_diagnostics.csv', candidate_rows)
    csv_table(output/'test_per_label.csv', labels)
    csv_table(output/'summaries.csv', summaries)
    report.update(status='birds_selection_test_complete', records=records, summaries=summaries,
        development_manifest=str(source), development_manifest_sha256=sha256(source),
        input_seal_sha256=manifest['input_seal_sha256'], sealed_files_verified=len(seal),
        maximum_selection_replay_error=error, test_evaluated=True, exploratory=True,
        tree_fits=0, threshold_fits=0, basis_fits=0, elapsed_seconds=time.perf_counter()-started)
    emit('birds_test_complete', decisions=len(records), candidates=len(candidate_rows), sealed_files=len(seal),
         tree_fits=0, threshold_fits=0, basis_fits=0, maximum_selection_replay_error=error)
