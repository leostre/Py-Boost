"""Bounded native SR2 rollout. Selection follows training; test follows a seal."""
from dataclasses import asdict, replace
from pathlib import Path
import gc, json, subprocess, sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked, digest, write, event, native_load
from abl_quality_worker import check_native_environment, memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected, block_cost
from abl_quality_core import full_output_base, fingerprint, fitted_control
from abl_increment_diagnostic_core import task_quality, projection_candidates, accumulation, fidelity
from abl_training_core import calibrated_local_variants
from abl_stop_rule_core import Candidate, Phase, Accepted, decide, quality_failures
from abl_stop_rule_replay import qvalue, raw_signature, choose_final
from abl_stop_continuation_core import Job, plan, reserve, state_from_json, initial_arrays
from experiments.absolute_history_core import AbsoluteHistory
from experiments.parallel_leaf_core import prefix
from experiments.shared_leaf_basis import reference_class_model, functional_basis
from experiments.sparse_leaf_core import leaf_design, contrast


def pin(ref, locks=None, artifact=False):
    key = 'artifact' if artifact else 'path'
    if digest(ref[key]) != ref['sha256']:
        raise RuntimeError('immutable SR2 source changed: '+str(ref[key]))
    if locks is not None:
        locks[ref[key]] = ref['sha256']
    return ref[key]


def old_case(sr, locks=None):
    pin(sr['original'], locks)
    return locked(sr['original']['path'], sr['original']['sha256'])


def history_arrays(history, estimate):
    s = history.state
    return dict(mean=np.empty((0, 0)) if s.mean is None else s.mean,
        gram=np.empty((0, 0)) if s.gram is None else s.gram,
        halves=np.empty((0, 0, 0)) if s.half_grams is None else np.asarray(s.half_grams),
        mass=np.array(s.mass), observations=np.array(s.observations), basis=estimate.basis)


def verify_history_file(history, estimate, ref):
    pin(ref)
    with np.load(ref['path'], allow_pickle=False) as a:
        for k, value in history_arrays(history, estimate).items():
            np.testing.assert_allclose(a[k], value,
                atol=2e-7 if k == 'basis' else 2e-13, rtol=2e-6 if k == 'basis' else 2e-12)


def restore_history(sr, path, source, data, locks=None):
    """Old snapshot is BEFORE rejected block b; observe its SR1 model once."""
    if path['path'] == 'functional':
        return None, None
    kind = path['path']; b = path['attempts'][-1]['block']
    c = source['record']['case']['outputs']; task = source['task']; rank = max(1, (c-int(task == 'multiclass'))//2)
    pin(source['warmup'], locks); warm = locked(source['warmup']['path'], source['warmup']['sha256'])
    pin(warm['warmup_threshold'], locks, True)
    teacher = native_load(warm['warmup_threshold']['artifact'])
    history = AbsoluteHistory(kind, c, task, .8)
    for trees in (0, 10, 20, 30, 40):
        z = (np.broadcast_to(teacher.base.bias, (len(data['train_y']), c)).copy() if trees == 0
            else prefix(teacher, trees).prepare_inference().raw(data['train_x']))
        estimate = history.observe(data['train_y'], z, 1., rank)
    ref = next(t for t in source['trajectories'] if t['mode'] == kind)
    pin(ref, locks); trajectory = locked(ref['path'], ref['sha256'])
    for block in range(1, b):
        meta = source['metas'][kind+f'/b{block:02d}']; pin(meta, locks, True)
        estimate = history.observe(data['train_y'], load_protected(meta['artifact']).raw(data['train_x']), 1., rank)
    cycle = next(t for t in trajectory['cycles'] if t['block'] == b)
    ref = dict(path=cycle['history_state_path'], sha256=cycle['history_state_sha256'])
    pin(ref, locks); verify_history_file(history, estimate, ref)
    snapshot = path['last_observed']; pin(snapshot, locks, True)
    estimate = history.observe(data['train_y'], load_protected(snapshot['artifact']).raw(data['train_x']), 1., rank)
    if history.state.observations != 5+b:
        raise RuntimeError('restored history observation count changed')
    return history, estimate


def raw_data(source):
    data = data_before_test(source['record']); nc, nt, _ = data['sizes']
    # The interpreter passes only these arrays to projection/calibration/fit.
    features = dict(train=data['train_x'], calibration=data['dev_x'][:nc], control=data['dev_x'][nc:nc+nt])
    labels = dict(train=data['train_y'], calibration=data['dev_y'][:nc], control=data['dev_y'][nc:nc+nt])
    return data, features, labels


def preflight(sr, job, cfg, directory, locks):
    source = old_case(sr, locks); data, features, labels = raw_data(source)
    path = next(p for p in sr['paths'] if p['path'] == job.kind)
    initial = path['last_observed']; pin(initial, locks, True)
    model = load_protected(initial['artifact']); state = state_from_json(path['attempts'][-1]['after'])
    assert state.anchor.token == sr['anchor']['token'] and state.anchor.quality == qvalue(sr['anchor']['control'])
    assert state.budget_bytes == sr['budget_bytes'] == source['budget_bytes']
    assert state.current.uid == initial['uid']
    assert model.n_trees == 40*(job.start_block+1) and model.n_trees < 200
    assert block_cost(model)['cost_bytes'] == state.current.cost_bytes == initial['cost_bytes']
    assert raw_signature(model) == initial['raw_signature']
    q = task_quality(labels['control'], model.raw(features['control']), source['task'], **dict(
        temperature=initial['policy']['temperature'], shifts=initial['policy']['shifts']))
    for k in q:
        assert abs(q[k]-initial['control'][k]) < 2e-8
    assert abs(task_quality(labels['control'], model.raw(features['control']), source['task'])['loss']-state.current.raw_nll) < 2e-8
    # Independently preserve the actual frozen full reference and its old policy.
    mode = 'temperature_shifts' if source['task'] == 'multiclass' else 'shifts'
    reference = next(c for c in source['choices'] if c['group'] == 'full_unbounded' and
        c['mode'] == mode and c['objective'] == 'f1_macro')
    assert reference == sr['anchor']['reference_choice']
    meta = source['metas'][reference['chosen']['uid']]; pin(meta, locks, True)
    cp = asdict(fitted_control(next(r for r in source['rows'] if r['uid'] == meta['uid']), reference['chosen']['policy']))
    cp['shifts'] = list(cp['shifts'])
    assert cp == sr['anchor']['policy']
    full_q = task_quality(labels['control'], load_protected(meta['artifact']).raw(features['control']), source['task'],
        cp['temperature'], cp['shifts'])
    for k in full_q:
        assert abs(full_q[k]-sr['anchor']['control'][k]) < 2e-8
    h, estimate = restore_history(sr, path, source, data, locks)
    result = dict(job=asdict(job), initial=initial, state=asdict(state), raw_signature=raw_signature(model),
        starting_trees=model.n_trees, budget_bytes=state.budget_bytes, history_observations=None if h is None else h.state.observations,
        training_roles=['train', 'calibration', 'control'], status='passed')
    if h is not None:
        p = directory/'restored_history.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **history_arrays(h, estimate)); result['history'] = dict(path=str(p), sha256=digest(p))
    return result


def save_snapshot(model, policy, uid, block, directory, labels, features, task, anchor):
    model = model.with_shifts(np.asarray(policy['shifts'], np.float32))
    policy = dict(temperature=float(policy['temperature']), shifts=model.decision_shifts.tolist())
    z = model.raw(features['control'])
    q = task_quality(labels['control'], z, task, policy['temperature'], policy['shifts'])
    directory.mkdir(parents=True, exist_ok=True); file = directory/'model.npz'; model.save(file)
    result = dict(uid=uid, block=block, policy=policy, artifact=str(file), sha256=digest(file),
        raw_signature=raw_signature(model), cost_bytes=block_cost(model)['cost_bytes'], trees=model.n_trees,
        control=q, raw_nll=task_quality(labels['control'], z, task)['loss'],
        control_guard=not quality_failures(qvalue(q), anchor.quality))
    write(directory/'policy.json', dict(temperature=policy['temperature'], model_sha256=result['sha256'],
        prediction_contract='raw/T - model.decision_shifts'))
    return result


def trajectory(request, output, fit_function=None):
    from abl_training_increment import fit_training_increment
    fit_function = fit_training_increment if fit_function is None else fit_function
    started = perf_counter(); job = Job(**request['job']); cfg = request['config']
    sr = locked(request['source']['path'], request['source']['sha256']); source = old_case(sr)
    data, features, labels = raw_data(source); task = source['task']; c = source['record']['case']['outputs']
    path = next(p for p in sr['paths'] if p['path'] == job.kind); initial = path['last_observed']
    pin(initial, artifact=True); model = load_protected(initial['artifact'])
    state = state_from_json(path['attempts'][-1]['after']); current = initial
    h, estimate = restore_history(sr, path, source, data)
    if h is not None:
        verify_history_file(h, estimate, request['preflight']['history'])
    output.mkdir(parents=True, exist_ok=True)
    work = output/'worker_data.npz'; np.savez(work, train_x=features['train'], train_y=labels['train'],
        validation_x=features['calibration'], validation_y=labels['calibration'])
    ledger_path = Path(request['ledger_path']); attempts = []; snapshots = []; terminal = 'approved_horizon_exhausted'
    rank = max(1, (c-int(task == 'multiclass'))//2)
    for block in job.blocks:
        directory = output/f'block{block:02d}'; directory.mkdir(parents=True, exist_ok=True)
        initial_path = directory/'initial.npz'; np.savez(initial_path, **initial_arrays(model, features['train'], features['calibration']))
        ledger = reserve(json.loads(ledger_path.read_text()), job, block); write(ledger_path, ledger)
        index = len(ledger['attempts'])-1
        fit = fit_function(dict(data_path=str(work), initial_path=str(initial_path), config={**cfg, 'max_depth':3},
            task=task, seed=source['record']['case']['seed']+50000+40*block, trees=40, lr=cfg['lr'],
            subsample=1., colsample=1., minimum_leaf=10, output=str(directory/'increment.npz'),
            minimum_free_mib=8192, own_pool_limit_bytes=16*1024**3))
        ledger = json.loads(ledger_path.read_text()); ledger['attempts'][index].update(state='complete', fit=fit); write(ledger_path, ledger)
        inc = native_load(pin(fit, artifact=True))
        if task == 'multiclass':
            inc = replace(inc, base=reference_class_model(inc.base))
        dense = full_output_base(inc)
        designs = {k: leaf_design(dense.leaf_indices(x), len(dense.coefficients)) for k, x in features.items()}
        protected = {k:model.raw(x) for k, x in features.items()}
        contributions = {k:np.asarray(d@dense.coefficients.astype(float)) for k, d in designs.items()}
        mean = contributions['train'].mean(0)
        if job.kind == 'functional':
            mean, basis, _ = functional_basis(contributions['train'], task)
        else:
            basis = estimate.basis
        teacher = {k:protected[k]+z+dense.bias for k, z in contributions.items()}
        rows = []; candidates = []
        for cand in projection_candidates(model, inc, designs, protected, labels, basis, mean, teacher, rank):
            variants = calibrated_local_variants(cand.raw['calibration'], labels['calibration'], cand.raw['control'],
                labels['control'], task, cfg['decision_grid'], cfg['decision_shrinkage'])
            raw_loss = task_quality(labels['control'], cand.raw['control'], task)['loss']
            for variant in variants:
                i = len(rows); uid = f'{job.kind}/sr2-b{block:02d}/v{i:02d}'
                candidate = Candidate(uid, f'residual-{cand.percent}', raw_loss, qvalue(variant['validation']),
                    cand.cost['cost_bytes'], cand.train_weighted_mse, i)
                rows.append(dict(percent=cand.percent, temperature=variant['temperature'],
                    shifts=variant['decision_shifts'], threshold_fraction=variant['threshold_fraction']))
                candidates.append(candidate)
        decision = decide(state, candidates)
        attempt = dict(block=block, before=asdict(state), candidate_rows=[asdict(c) for c in candidates],
            variants=rows, gates=[asdict(g) for g in decision.gates], fit=fit,
            initial_path=str(initial_path), initial_sha256=digest(initial_path), initial_model=current,
            full_step_raw_nll=task_quality(labels['control'], teacher['control'], task)['loss'],
            full_step_quality_previous_policy=task_quality(labels['control'], teacher['control'], task,
                current['policy']['temperature'], current['policy']['shifts']), rank=rank,
            history_diagnostics=None if estimate is None else estimate.diagnostics)
        if h is not None:
            file = directory/'history_before_projection.npz'; np.savez(file, **history_arrays(h, estimate))
            attempt['history'] = dict(path=str(file), sha256=digest(file))
        if not isinstance(decision, Accepted):
            terminal = decision.reason.value; attempt.update(action='stopped', reason=terminal)
            attempts.append(attempt); break
        state = decision.state; chosen = rows[state.current.ordinal]
        selected = next(cand for cand in projection_candidates(model, inc, designs, protected, labels, basis,
            mean, teacher, rank) if cand.percent == chosen['percent'])
        accepted = selected.combined; cp = dict(temperature=chosen['temperature'], shifts=chosen['shifts'])
        snapshot = save_snapshot(accepted, cp, state.current.uid, block, directory, labels, features, task, state.anchor)
        assert snapshot['cost_bytes'] == state.current.cost_bytes <= state.budget_bytes
        assert snapshot['trees'] <= 200 and abs(snapshot['raw_nll']-state.current.raw_nll) < 2e-8
        for k in snapshot['control']:
            assert abs(snapshot['control'][k]-asdict(state.current.quality)[k]) < 2e-8
        assert [fingerprint(b) for b in model.blocks] == [fingerprint(b) for b in accepted.blocks[:-1]]
        attempt.update(action='accepted', selected_uid=state.current.uid, selected_percent=selected.percent,
            after=asdict(state), entered_qualification=decision.entered_qualification, snapshot=snapshot,
            local_control_fidelity=fidelity(selected.raw['control']-teacher['control'],
                teacher['control']-protected['control'], task))
        attempts.append(attempt); snapshots.append(snapshot); model = load_protected(snapshot['artifact']); current = snapshot
        if h is not None:
            estimate = h.observe(labels['train'], model.raw(features['train']), cfg['lambda_l2'], rank)
        write(output/'progress.json', dict(job=asdict(job), attempts=attempts, snapshots=snapshots))
        event('sr2_checkpoint', case=job.case, kind=job.kind, block=block, trees=model.n_trees,
            bytes=snapshot['cost_bytes'], phase=state.phase.value, control=snapshot['control'])
        del designs, contributions, protected, teacher, inc, dense, selected, accepted
        gc.collect(); memory_guard()
    result = dict(status='complete', job=asdict(job), source=request['source'], preflight=request['preflight'],
        initial=initial, attempts=attempts, snapshots=snapshots, terminal=terminal, final_state=asdict(state),
        training_roles=['train', 'calibration', 'control'], selection_used_for_training=False, test_accessed=False,
        cpu_peak_rss_bytes=memory_guard(), elapsed_seconds=perf_counter()-started)
    write(output/'trajectory.json', result)
    event('sr2_trajectory_complete', case=job.case, kind=job.kind, terminal=terminal,
        new_trees=40*len(attempts), accepted_steps=len(snapshots), phase=state.phase.value)


def enrich(snapshot, source, data, sr):
    """Selection is evaluated only after every requested tree fit is finished."""
    model = load_protected(pin(snapshot, artifact=True)); n = sum(data['sizes'][:2]); cp = snapshot['policy']
    q = task_quality(data['dev_y'][n:], model.raw(data['dev_x'][n:]), source['task'], cp['temperature'], cp['shifts'])
    return dict(snapshot, trees=model.n_trees, selection=q,
        selection_guard=not quality_failures(qvalue(q), qvalue(sr['anchor']['selection'])))


def development(sr_ref, trajectories):
    sr = locked(sr_ref['path'], sr_ref['sha256']); source = old_case(sr); data = data_before_test(source['record'])
    paths = []
    for old in sr['paths']:
        t = trajectories.get((sr['case'], old['path']))
        saved = [old['initial'], *[a['snapshot'] for a in old['attempts'] if a['action'] == 'accepted']]
        new = [] if t is None else t['snapshots']; items = [enrich(v, source, data, sr) for v in [*saved, *new]]
        qualified = [s for s in items if s['control_guard']]
        matched = old['matched_old_final']
        old_meta = matched['source']
        matched = enrich(dict(matched, artifact=old_meta['artifact'], sha256=old_meta['sha256']), source, data, sr)
        historical = old['historical_old_selection']['chosen']; uid = historical['uid']; meta = source['metas'][uid]
        policy = asdict(fitted_control(next(r for r in source['rows'] if r['uid'] == uid), historical['policy']))
        policy['shifts'] = list(policy['shifts'])
        historical = enrich(dict(uid=uid, policy=policy, artifact=meta['artifact'], sha256=meta['sha256'],
            cost_bytes=meta['cost_bytes']), source, data, sr)
        paths.append(dict(path=old['path'], continued=t is not None,
            terminal=old['terminal'] if t is None else t['terminal'],
            snapshots=items, final_qualified=choose_final(qualified), diagnostic_best=choose_final(items),
            start=items[len(saved)-1], last=items[-1], matched_old=matched, historical_old_pooled=historical,
            new_trees=0 if t is None else 40*len(t['attempts'])))
    uid = sr['anchor']['uid']; meta = source['metas'][uid]
    full = dict(uid=uid, policy=sr['anchor']['policy'], artifact=meta['artifact'], sha256=meta['sha256'],
        cost_bytes=meta['cost_bytes'], control=sr['anchor']['control'], selection=sr['anchor']['selection'])
    return dict(case=sr['case'], task=sr['task'], source=sr_ref, original=sr['original'],
        budget_bytes=sr['budget_bytes'], anchor=sr['anchor'], full=full, paths=paths, test_accessed=False)


def evaluate_test(case, seal_sha, output):
    source = locked(case['original']['path'], case['original']['sha256']); record = source['record']
    pin(dict(path=record['case']['split_file'], sha256=record['case']['split_sha256']))
    pin(record['dataset']); directory = output/'test'/case['case']; directory.mkdir(parents=True, exist_ok=True)
    with np.load(record['case']['split_file'], allow_pickle=False) as a:
        ids = a['test'].copy()
    with np.load(record['dataset']['path'], allow_pickle=False) as a:
        x, y = a['features'][ids].astype(np.float32), a['target'][ids].copy()
    target = directory/'targets.npz'; np.savez_compressed(target, target=y, indices=ids)
    points = [('full', 'full', case['full'])]
    for p in case['paths']:
        for name in ('final_qualified', 'diagnostic_best', 'start', 'last', 'matched_old', 'historical_old_pooled'):
            if p[name] is not None:
                points.append((p['path'], name, p[name]))
    predictions = {}; values = []; raw = {}
    for kind, name, s in points:
        sha = s['sha256']; cp = s['policy']; model = load_protected(pin(s, artifact=True))
        if sha not in raw:
            z = model.raw(x); file = directory/(sha+'.npz'); np.savez_compressed(file, raw=z)
            raw[sha] = z; predictions[sha] = dict(path=str(file), sha256=digest(file))
        q = task_quality(y, raw[sha], case['task'], cp['temperature'], cp['shifts'])
        values.append(dict(path=kind, point=name, snapshot=s, test=q, trees=model.n_trees))
    full_q = values[0]['test']
    for v in values:
        v['test_guard'] = not quality_failures(qvalue(v['test']), qvalue(full_q))
    result = dict(case=case['case'], task=case['task'], points=values, predictions=predictions,
        target=dict(path=str(target), sha256=digest(target)), pretest_seal_sha256=seal_sha,
        test_used_for_selection=False, new_tree_fits=0)
    file = directory/'evaluation.json'; write(file, result)
    return dict(case=case['case'], path=str(file), sha256=digest(file))


def run(cfg, output):
    start = perf_counter(); env = check_native_environment(cfg)
    contract = dict(maximum_new_trees=1320, trajectories=16, block_trees=40, total_tree_horizon=200,
        tolerance=.05, minimum_free_mib=8192, own_pool_limit_bytes=16*1024**3,
        candidates='same_own_basis_half_rank_residuals_0_1_2_5', controller='unchanged_SR1',
        final_test='sealed_after_training_and_development_audit', stop_after_results=True)
    if cfg['abl_stop_continuation'] != contract or cfg['lr'] != .1 or cfg['lambda_l2'] != 1:
        raise RuntimeError('outside approved bounded SR2 contract')
    ref = cfg['abl_stop_rule_replay_completed']; root = locked(ref['path'], ref['sha256'])
    assert root['status'] == 'fixed_anchor_saved_rule_complete'
    assert locked(root['audit']['path'], root['audit']['sha256'])['status'] == 'passed'
    selected_refs = [r for r in root['cases'] if r['case'].startswith(('mediamill_', 'mnist_'))]
    cases = [locked(r['path'], r['sha256']) for r in selected_refs]; jobs = plan(cases)
    locks = {ref['path']:ref['sha256']}; preflights = {}; requests = []
    ledger_path = output/'training_ledger.json'; write(ledger_path, dict(reserved_rounds=0, attempts=[], phase_limit=1320))
    native = json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    worker_cfg = {k:cfg[k] for k in ('lr', 'lambda_l2', 'ntrees', 'decision_grid', 'decision_shrinkage', 'own_pool_limit_bytes')}
    for job in jobs:
        sr = next(c for c in cases if c['case'] == job.case); sr_ref = next(r for r in selected_refs if r['case'] == job.case)
        pin(sr_ref, locks); directory = output/'trajectories'/job.case/job.kind
        pre = preflight(sr, job, cfg, directory, locks); preflights[job.case+'/'+job.kind] = pre
        request = dict(job=asdict(job), source=sr_ref, preflight=pre, config=worker_cfg, ledger_path=str(ledger_path))
        file = directory/'request.json'; write(file, request); requests.append((job, file, directory))
        event('sr2_preflight', case=job.case, kind=job.kind, starting_trees=pre['starting_trees'], history=pre['history_observations'])
        gc.collect(); memory_guard()
    write(output/'preflight.json', dict(status='passed', jobs=[asdict(j) for j in jobs], maximum_new_trees=1320,
        checks=preflights, immutable_inputs=locks))
    entries = []; trajectories = {}
    for job, request_file, directory in requests:
        subprocess.run([native, __file__, '--trajectory', str(request_file), str(directory)], check=True, timeout=3600)
        file = directory/'trajectory.json'; ref_t = dict(path=str(file), sha256=digest(file))
        t = locked(file, ref_t['sha256']); assert t['status'] == 'complete'
        trajectories[job.case, job.kind] = t; entries.append(ref_t)
        write(output/'progress.json', dict(trajectories=entries, new_tree_fits=json.loads(ledger_path.read_text())['reserved_rounds']))
    from audit_abl_stop_continuation import audit_development, audit_choices, audit_final
    ledger = json.loads(ledger_path.read_text()); checked = audit_development(entries, ledger)
    dev = [development(r, trajectories) for r in selected_refs]
    checked['choices'] = audit_choices(dev)
    audit_file = output/'development_audit.json'; write(audit_file, checked)
    seal = output/'pretest_choices.json'; write(seal, dict(status='sealed_before_test', cases=dev,
        development_audit_sha256=digest(audit_file), source=ref, test_used_for_selection=False))
    seal_sha = digest(seal)
    evaluated = [evaluate_test(case, seal_sha, output) for case in dev]
    final_check = audit_final(dev, evaluated, seal_sha); final_file = output/'final_audit.json'; write(final_file, final_check)
    assert digest(seal) == seal_sha
    for path, sha in locks.items():
        assert digest(path) == sha
    from collections import Counter
    all_paths = [p for c in dev for p in c['paths']]
    tests = [locked(e['path'], e['sha256']) for e in evaluated]
    result = dict(status='bounded_native_continuation_complete', contract=contract, source=ref,
        trajectories=entries, evaluations=evaluated, training_ledger=ledger,
        pretest_seal=dict(path=str(seal), sha256=seal_sha),
        development_audit=dict(path=str(audit_file), sha256=digest(audit_file)),
        final_audit=dict(path=str(final_file), sha256=digest(final_file)),
        immutable_inputs=locks, environment=env, elapsed_seconds=perf_counter()-start, cpu_peak_rss_bytes=memory_guard(),
        summary=dict(paths=len(all_paths), continued=16, new_trees=ledger['reserved_rounds'],
            accepted_new_blocks=sum(len(t['snapshots']) for t in trajectories.values()),
            terminals=dict(Counter(p['terminal'] for p in all_paths)),
            qualified=sum(p['final_qualified'] is not None for p in all_paths),
            qualified_selection_pass=sum(p['final_qualified'] is not None and p['final_qualified']['selection_guard'] for p in all_paths),
            qualified_test_pass=sum(p['point']=='final_qualified' and p['test_guard'] for c in tests for p in c['points'])),
        limitations=['old research data/test reused; not independent confirmation',
            'full anchor retains historical pooled calibration; new candidates use calibration/control',
            'temporary full derivatives and training teachers remain; no total-training-memory savings claim'])
    write(output/'report.json', result)
    event('sr2_complete', path=str(output/'report.json'), sha256=digest(output/'report.json'), summary=result['summary'])


def main():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1] == '--trajectory':
            trajectory(json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
        else:
            run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')), Path(sys.argv[2]))


if __name__ == '__main__':
    main()
