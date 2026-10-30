"""CPU replay of fixed-anchor decisions, censored at the first raw-path change."""
from dataclasses import asdict, replace
from pathlib import Path
import gc, hashlib, json, sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked, write, digest, event, native_load
from abl_quality_worker import check_native_environment, memory_guard
from abl_quality_core import fitted_control, full_output_base, fingerprint
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected, block_cost
from abl_training_core import calibrated_local_variants
from abl_increment_diagnostic_core import task_quality, projection_candidates
from abl_stop_rule_core import (Quality, Anchor, Candidate, Phase, Accepted, Stopped,
    initialize, decide, quality_failures, NLL_EPSILON)
from experiments.shared_leaf_basis import functional_basis, reference_class_model
from experiments.sparse_leaf_core import leaf_design

KINDS = ('functional', 'current', 'mean', 'history')


def qvalue(row):
    return Quality(*(float(row[k]) for k in ('loss', 'f1_macro', 'f1_weighted')))


def raw_signature(model):
    h = hashlib.sha256(model.bias.tobytes())
    for b in model.blocks:
        h.update(fingerprint(b).encode())
    return h.hexdigest()  # Global decision policy cannot affect training logits.


def choose_final(items):
    if not items:
        return None
    best = max(x['selection']['f1_macro'] for x in items)
    rows = [x for x in items if x['selection']['f1_macro'] >= best-1e-12]
    loss = min(x['selection']['loss'] for x in rows)
    return min((x for x in rows if x['selection']['loss'] <= loss+1e-6),
        key=lambda x: (x['cost_bytes'], x['uid']))


def typed_rows(rows, prefix):
    # Recover uncalibrated NLL from a T=1 policy of each representation.
    raw = {}
    for r in rows:
        if r['temperature'] == 1.:
            raw[r['percent']] = r['validation']['loss']
    return tuple(Candidate(prefix+f'/v{i:02d}', f"residual-{r['percent']}",
        float(raw[r['percent']]), qvalue(r['validation']), int(r['cost_bytes']),
        float(r['train_weighted_mse']), i) for i, r in enumerate(rows))


def case_replay(entry, cfg, out, locks):
    def pin(path, sha):
        if path not in locks:
            if digest(path) != sha:
                raise RuntimeError('frozen stopping-rule source changed: '+str(path))
            locks[path] = sha
    pin(entry['path'], entry['sha256'])
    diagnostic = locked(entry['path'], entry['sha256'])
    original_ref = diagnostic['input']
    pin(original_ref['path'], original_ref['sha256'])
    source = locked(original_ref['path'], original_ref['sha256'])
    data = data_before_test(source['record'])
    for p, sha in ((source['record']['dataset']['path'], source['record']['dataset']['sha256']),
                   (source['record']['case']['split_file'], source['record']['case']['split_sha256'])):
        pin(p, sha)
    cid, task, budget = source['case_id'], source['task'], source['budget_bytes']
    nc, nt, _ = data['sizes']
    slices = dict(calibration=slice(0, nc), control=slice(nc, nc+nt), selection=slice(nc+nt, None))
    labels = dict(train=data['train_y'], **{k: data['dev_y'][s] for k, s in slices.items()})
    features = dict(train=data['train_x'], **{k: data['dev_x'][s] for k, s in slices.items()})
    def quality(z, cp, role):
        return task_quality(labels[role], z, task, cp['temperature'], cp['shifts'])
    def cache(uid):
        meta = source['metas'][uid]
        pin(meta['artifact'], meta['sha256'])
        f = meta['development_cache']; pin(f['path'], f['sha256'])
        with np.load(f['path'], allow_pickle=False) as a:
            return a['raw'].copy()
    mode = 'temperature_shifts' if task == 'multiclass' else 'shifts'
    reference_choice = next(c for c in source['choices'] if c['mode'] == mode and
        c['objective'] == 'f1_macro' and c['group'] == 'full_unbounded')
    full_uid = reference_choice['chosen']['uid']
    full_row = next(r for r in source['rows'] if r['uid'] == full_uid)
    full_policy = asdict(fitted_control(full_row, reference_choice['chosen']['policy']))
    full_z = cache(full_uid)
    full_control = quality(full_z[slices['control']], full_policy, 'control')
    full_selection = quality(full_z[slices['selection']], full_policy, 'selection')
    for k, v in full_selection.items():
        if abs(v-reference_choice['chosen']['validation'][k]) > 2e-10:
            raise RuntimeError('frozen full reference policy changed')
    token = full_uid+'/'+reference_choice['chosen']['policy']+'/'+source['metas'][full_uid]['sha256']
    anchor = Anchor(token, qvalue(full_control))
    result = dict(case=cid, task=task, input=entry, original=original_ref, budget_bytes=budget,
        anchor=dict(token=token, uid=full_uid, policy_name=reference_choice['chosen']['policy'],
            policy=full_policy, control=full_control, selection=full_selection,
            source=source['metas'][full_uid]['source'], reference_choice=reference_choice),
        one_step=[], paths=[], test_labels_used=False)
    case_dir = out/'cases'/cid; case_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(model, cp, uid, directory, old_meta=None, row=None):
        # Store one decision vector inside the model; temperature stays external.
        m = model.with_shifts(np.asarray(cp['shifts'], np.float32))
        z = m.raw(data['dev_x'])
        value = dict(uid=uid, policy=cp, control=quality(z[slices['control']], cp, 'control'),
            selection=quality(z[slices['selection']], cp, 'selection'),
            raw_nll=task_quality(labels['control'], z[slices['control']], task)['loss'],
            raw_signature=raw_signature(m), cost_bytes=block_cost(m)['cost_bytes'])
        value['control_guard'] = not quality_failures(qvalue(value['control']), anchor.quality)
        value['selection_guard'] = not quality_failures(qvalue(value['selection']), qvalue(full_selection))
        directory.mkdir(parents=True, exist_ok=True)
        file = directory/'model.npz'; m.save(file)
        value['artifact'] = str(file); value['sha256'] = digest(file)
        write(directory/'policy.json', dict(temperature=cp['temperature'], model_sha256=value['sha256'],
            prediction_contract='raw/T - model.decision_shifts; policy is already fitted',
            note='BlockLeafModel.predict does not apply an external temperature automatically'))
        if row is not None:
            assert value['cost_bytes'] == row['cost_bytes']
            for k in value['control']:
                assert abs(value['control'][k]-row['validation'][k]) < 2e-8
                assert abs(value['selection'][k]-row['selection'][k]) < 2e-8
        return value

    def construct(previous_uid, cycle, kind, row):
        meta = source['metas'][previous_uid]
        previous = load_protected(meta['artifact'])
        fit = cycle['fit']; pin(fit['artifact'], fit['sha256'])
        inc = native_load(fit['artifact'])
        if task == 'multiclass':
            inc = replace(inc, base=reference_class_model(inc.base))
        dense = full_output_base(inc)
        designs = {k: leaf_design(dense.leaf_indices(x), len(dense.coefficients)) for k, x in features.items()}
        protected = {k: previous.raw(x) for k, x in features.items()}
        contributions = {k: np.asarray(d@dense.coefficients.astype(float)) for k, d in designs.items()}
        mean = contributions['train'].mean(0)
        if kind == 'functional':
            mean, basis, _ = functional_basis(contributions['train'], task)
        else:
            pin(cycle['history_state_path'], cycle['history_state_sha256'])
            with np.load(cycle['history_state_path'], allow_pickle=False) as a:
                basis = a['basis'].copy()
        teacher = {k: protected[k]+z+dense.bias for k, z in contributions.items()}
        rank = max(1, (len(dense.bias)-int(task == 'multiclass'))//2)
        for cand in projection_candidates(previous, inc, designs, protected, labels, basis, mean, teacher, rank):
            if cand.percent == row['percent']:
                return cand.combined
        raise RuntimeError('selected saved representation absent')

    for kind in KINDS:
        warm_uid = kind+'/b00'; warm_meta = source['metas'][warm_uid]; warm_z = cache(warm_uid)
        warm_rows = calibrated_local_variants(warm_z[:nc], labels['calibration'],
            warm_z[nc:nc+nt], labels['control'], task, cfg['decision_grid'], cfg['decision_shrinkage'])
        warm_raw_nll = task_quality(labels['control'], warm_z[nc:nc+nt], task)['loss']
        initial_candidates = tuple(Candidate(kind+f'/warm/v{i}', 'warmup', warm_raw_nll,
            qvalue(v['validation']), warm_meta['cost_bytes'], 0., i) for i, v in enumerate(warm_rows))
        state = initialize(initial_candidates, anchor, budget)
        initial_policy = warm_rows[state.current.ordinal]
        cp = dict(temperature=initial_policy['temperature'], shifts=initial_policy['decision_shifts'])
        warm_model = load_protected(warm_meta['artifact'])
        initial = snapshot(warm_model, cp, state.current.uid, case_dir/kind/'warmup')
        certificates = [initial] if state.phase is Phase.QUALIFIED else []
        old_warm = min(warm_rows, key=lambda v: (-v['validation']['f1_macro'], v['validation']['loss'], v['threshold_fraction']))
        old_cp = dict(temperature=old_warm['temperature'], shifts=old_warm['decision_shifts'])
        baseline = [dict(uid=warm_uid, policy=old_cp, control=quality(warm_z[nc:nc+nt], old_cp, 'control'),
            selection=quality(warm_z[nc+nt:], old_cp, 'selection'), cost_bytes=warm_meta['cost_bytes'],
            source=warm_meta['source'])]
        ref = next(r for r in source['trajectories'] if r['mode'] == kind)
        pin(ref['path'], ref['sha256']); trajectory = locked(ref['path'], ref['sha256'])
        attempts = []; route_open = True; terminal = 'saved_horizon_exhausted'; last = initial
        for cycle in trajectory['cycles']:
            block = cycle['block']; previous_uid = kind+f'/b{block-1:02d}'
            previous_z = cache(previous_uid)
            previous_raw_nll = task_quality(labels['control'], previous_z[nc:nc+nt], task)['loss']
            diagnostic_row = next(r for r in diagnostic['basis_comparisons'] if r['path'] == kind and
                r['basis'] == kind and r['block'] == block)
            rows = diagnostic_row['candidates']
            candidates = typed_rows(rows, kind+f'/b{block:02d}')
            # All-111 summaries are conditional on the OLD saved state, not a new rollout.
            gates = [dict(uid=c.uid, memory=c.cost_bytes <= budget,
                useful=previous_raw_nll-c.raw_nll > NLL_EPSILON,
                fixed_quality=not quality_failures(c.quality, anchor.quality)) for c in candidates]
            result['one_step'].append(dict(path=kind, block=block, original_action=cycle['action'],
                previous_raw_nll=previous_raw_nll, candidates=[asdict(c) for c in candidates], gates=gates,
                scope='conditional on original saved prefix; not evidence of a reachable new trajectory'))
            if cycle['action'] != 'rejected':
                chosen = cycle['projection']['chosen']
                old_row = next(r for r in rows if r['percent'] == chosen['percent'] and
                    r['temperature'] == chosen['temperature'] and r['threshold_fraction'] == chosen['threshold_fraction'])
                meta = source['metas'][kind+f'/b{block:02d}']
                baseline.append(dict(uid=kind+f'/b{block:02d}', policy=dict(temperature=old_row['temperature'],
                    shifts=old_row['decision_shifts']), control=old_row['validation'], selection=old_row['selection'],
                    cost_bytes=old_row['cost_bytes'], source=meta['source']))
            if not route_open:
                continue
            if abs(state.current.raw_nll-previous_raw_nll) > 2e-10:
                raise RuntimeError('continued on a mismatched training prefix')
            before = state
            decision = decide(state, candidates)
            attempt = dict(block=block, previous_uid=previous_uid, before=asdict(before),
                candidate_rows=[asdict(c) for c in candidates], gates=[asdict(g) for g in decision.gates])
            if isinstance(decision, Stopped):
                terminal = decision.reason.value; route_open = False
                attempt.update(action='stopped', reason=terminal)
                attempts.append(attempt); continue
            state = decision.state
            row = rows[state.current.ordinal]
            model = construct(previous_uid, cycle, kind, row)
            cp = dict(temperature=row['temperature'], shifts=row['decision_shifts'])
            observed = snapshot(model, cp, state.current.uid, case_dir/kind/f'block{block:02d}', row=row)
            assert abs(observed['raw_nll']-state.current.raw_nll) < 2e-8
            if state.phase is Phase.QUALIFIED:
                certificates.append(observed)
            compatibility = False
            if cycle['action'] != 'rejected':
                old_meta = source['metas'][kind+f'/b{block:02d}']
                pin(old_meta['artifact'], old_meta['sha256'])
                compatibility = raw_signature(model) == raw_signature(load_protected(old_meta['artifact']))
            attempt.update(action='accepted', selected_uid=state.current.uid, after=asdict(state),
                selected_percent=row['percent'], snapshot=observed, raw_path_compatible=compatibility,
                entered_qualification=decision.entered_qualification, original_action=cycle['action'])
            attempts.append(attempt); last = observed
            if not compatibility:
                route_open = False
                terminal = ('censored_old_rejected_block_now_accepted' if cycle['action'] == 'rejected'
                    else 'censored_changed_raw_function')
            del model
            gc.collect(); memory_guard()
        final = choose_final(certificates)
        old_final = choose_final(baseline)
        historical = next(c for c in source['choices'] if c['mode'] == mode and
            c['objective'] == 'f1_macro' and c['group'] == kind)['chosen']
        result['paths'].append(dict(path=kind, initial=initial, initial_state=asdict(initialize(initial_candidates, anchor, budget)),
            attempts=attempts, terminal=terminal, censored=terminal.startswith('censored'),
            qualified_snapshots=certificates, final_observed=final, last_observed=last,
            matched_old_final=dict(old_final,
                control_guard=not quality_failures(qvalue(old_final['control']), anchor.quality),
                selection_guard=not quality_failures(qvalue(old_final['selection']), qvalue(full_selection))),
            historical_old_selection=dict(chosen=historical,
                guard=not quality_failures(qvalue(historical['validation']), qvalue(full_selection)))))
        event('stop_rule_path', case=cid, path=kind, observed_steps=len(attempts), terminal=terminal,
            observed_certificate=final is not None,
            certificate_selection_guard=None if final is None else final['selection_guard'])
    file = case_dir/'case.json'; write(file, result)
    return dict(case=cid, path=str(file), sha256=digest(file))


def run(cfg, out):
    expected = dict(tree_training_allowed=False, gpu_execution=False, test_labels_used=False,
        tolerance=.05, nll_epsilon=1e-6, anchor_mode='frozen_calibrated_macro_reference',
        continuation='exact_raw_signature_only', candidates='same_own_basis_half_rank_residuals_0_1_2_5',
        qualification_priority=True)
    if cfg['abl_stop_rule'] != expected:
        raise RuntimeError('outside approved stopping-rule replay contract')
    start = perf_counter(); environment = check_native_environment(cfg)
    ref = cfg['abl_increment_diagnostic_completed']; previous = locked(ref['path'], ref['sha256'])
    if previous['status'] != 'saved_increment_diagnostic_complete' or len(previous['cases']) != 9:
        raise RuntimeError('nine audited saved cases required')
    if locked(previous['audit']['path'], previous['audit']['sha256'])['status'] != 'passed':
        raise RuntimeError('HD1 audit required')
    locks = {ref['path']: ref['sha256']}; cases = []
    event('stop_rule_start', contract=expected, source=ref, new_tree_fits=0)
    for entry in previous['cases']:
        cases.append(case_replay(entry, cfg, out, locks))
        write(out/'progress.json', dict(cases=cases, new_tree_fits=0))
    from audit_abl_stop_rule import audit
    checked = audit(cases)
    write(out/'audit_report.json', checked)
    for path, sha in locks.items():
        if digest(path) != sha:
            raise RuntimeError('source changed during rule replay')
    if any(k.split('.')[0] in ('cupy', 'torch', 'py_boost') for k in sys.modules):
        raise RuntimeError('training/GPU runtime imported')
    from collections import Counter
    result = dict(status='fixed_anchor_saved_rule_complete', cases=cases, source=ref,
        immutable_inputs=locks, audit=dict(path=str(out/'audit_report.json'), sha256=digest(out/'audit_report.json')),
        new_tree_fits=0, gpu_execution=False, test_labels_used=False, environment=environment,
        cpu_peak_rss_bytes=memory_guard(), elapsed_seconds=perf_counter()-start)
    values = [locked(e['path'], e['sha256']) for e in cases]
    result['summary'] = dict(conditional_steps=sum(len(c['one_step']) for c in values),
        paths=sum(len(c['paths']) for c in values),
        terminal_counts=dict(Counter(p['terminal'] for c in values for p in c['paths'])),
        observed_certificates=sum(p['final_observed'] is not None for c in values for p in c['paths']),
        observed_selection_guard_passes=sum(p['final_observed'] is not None and p['final_observed']['selection_guard'] for c in values for p in c['paths']),
        matched_old_selection_guard_passes=sum(p['matched_old_final']['selection_guard'] for c in values for p in c['paths']),
        censored=sum(p['censored'] for c in values for p in c['paths']))
    write(out/'report.json', result)
    event('stop_rule_complete', path=str(out/'report.json'), sha256=digest(out/'report.json'),
        summary=result['summary'], new_tree_fits=0, cpu_peak_rss_bytes=result['cpu_peak_rss_bytes'])


def main():
    from threadpoolctl import threadpool_limits
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    with threadpool_limits(limits=1):
        run(cfg, Path(sys.argv[2]))


if __name__ == '__main__':
    main()
