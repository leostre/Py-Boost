"""Replay saved H1 increments on CPU; no future trajectories are invented."""
from dataclasses import replace
from pathlib import Path
import gc
import json
import sys
from time import perf_counter
import numpy as np

from budget_quality_worker import digest, locked, write, event, native_load
from abl_quality_worker import check_native_environment, memory_guard
from abl_quality_core import full_output_base, fingerprint
from abl_transfer_study import data_before_test
from abl_training_core import calibrated_local_variants
from abl_protected_core import load_protected
from abl_increment_diagnostic_core import (
    projection_candidates, guard_details, task_quality, fidelity, accumulation, stop_cause)
from experiments.shared_leaf_basis import functional_basis, reference_class_model
from experiments.sparse_leaf_core import leaf_design, contrast
from experiments.absolute_history_core import AbsoluteHistory
from experiments.gradient_basis_core import captured_energy
from experiments.parallel_leaf_core import prefix

KINDS = ('functional', 'current', 'mean', 'history')
ROLES = ('train', 'calibration', 'control', 'selection')


def policy(v):
    return dict(temperature=v['temperature'], shifts=v['decision_shifts'])


def quality(y, z, task, v):
    p = policy(v)
    return task_quality(y, z, task, p['temperature'], p['shifts'])


def variants(raw, labels, task, cfg):
    return calibrated_local_variants(raw['calibration'], labels['calibration'],
        raw['control'], labels['control'], task, cfg['decision_grid'], cfg['decision_shrinkage'])


def best_policy(raw, labels, task, cfg):
    return min(variants(raw, labels, task, cfg), key=lambda v: (
        -v['validation']['f1_macro'], v['validation']['loss'], v['threshold_fraction']))


def pick(rows, budget, reference=None):
    ids = [i for i, r in enumerate(rows) if r['cost_bytes'] <= budget and
        (reference is None or guard_details(r['validation'], reference)['passed'])]
    return min(ids, key=lambda i: (rows[i]['train_weighted_mse'],
        -rows[i]['validation']['f1_macro'], rows[i]['validation']['loss'],
        rows[i]['cost_bytes'], rows[i]['percent'])) if ids else None


def record_choice(rows, index):
    if index is None:
        return None
    r = rows[index]
    return dict(index=index, percent=r['percent'], nnz=r['nnz'], cost_bytes=r['cost_bytes'],
        train_weighted_mse=r['train_weighted_mse'], policy=r['policy'],
        temperature=r['temperature'], threshold_fraction=r['threshold_fraction'],
        validation=r['validation'], selection=r['selection'],
        control_guard=r['control_guard'], selection_guard=r['selection_guard'],
        previous_control_guard=r['previous_control_guard'], previous_selection_guard=r['previous_selection_guard'])


def verify_rows(actual, expected):
    if len(actual) != len(expected):
        raise RuntimeError('original projection candidate count changed')
    maximum = 0.
    for a, b in zip(actual, expected):
        for k in ('percent', 'rank', 'nnz', 'temperature', 'threshold_fraction'):
            if a[k] != b[k]:
                raise RuntimeError('original projection scalar changed: '+k)
        if a['cost_bytes'] != b['cost']['cost_bytes']:
            raise RuntimeError('original projection storage changed')
        np.testing.assert_array_equal(a['decision_shifts'], b['decision_shifts'])
        maximum = max(maximum, abs(a['train_weighted_mse']-b['train_weighted_mse']),
            *(abs(a['validation'][k]-b['validation'][k]) for k in b['validation']))
    if maximum > 2e-8:
        raise RuntimeError('original projection score changed')
    return maximum


def diagnose_case(entry, cfg, out, locks):
    def pin(path, sha):
        if path not in locks:
            if digest(path) != sha:
                raise RuntimeError('immutable diagnostic input changed: '+path)
            locks[path] = sha
    pin(entry['path'], entry['sha256'])
    case = locked(entry['path'], entry['sha256'])
    for p, h in ((case['record']['dataset']['path'], case['record']['dataset']['sha256']),
                 (case['record']['case']['split_file'], case['record']['case']['split_sha256'])):
        pin(p, h)
    data = data_before_test(case['record'])
    ncal, nctrl, _ = data['sizes']
    spans = dict(calibration=slice(0, ncal), control=slice(ncal, ncal+nctrl),
        selection=slice(ncal+nctrl, None))
    features = dict(train=data['train_x'], **{k: data['dev_x'][s] for k, s in spans.items()})
    labels = dict(train=data['train_y'], **{k: data['dev_y'][s] for k, s in spans.items()})
    task, cid = case['task'], case['case_id']
    c = case['record']['case']['outputs']
    rank = max(1, (c-int(task == 'multiclass'))//2)
    warm_ref = case['warmup']
    pin(warm_ref['path'], warm_ref['sha256'])
    warm = locked(warm_ref['path'], warm_ref['sha256'])
    w = warm['models'][0]
    pin(w['artifact'], w['sha256'])
    warm_model = load_protected(w['artifact'])
    warm_raw = {k: warm_model.raw(x) for k, x in features.items()}
    warm_teacher_ref = warm['warmup_threshold']
    pin(warm_teacher_ref['artifact'], warm_teacher_ref['sha256'])
    warm_teacher = native_load(warm_teacher_ref['artifact'])
    warm_states = []
    for trees in (0, 10, 20, 30, 40):
        warm_states.append(np.broadcast_to(warm_teacher.base.bias, (len(labels['train']), c)).copy()
            if not trees else prefix(warm_teacher, trees).prepare_inference().raw(features['train']))
    result = dict(case=cid, task=task, rank=rank, budget_bytes=case['budget_bytes'],
        role_sizes={k: len(y) for k, y in labels.items()}, input=entry,
        basis_comparisons=[], accumulation=[], stop_decisions=[], original_replay=[],
        path_scope='same saved increment and actual preceding checkpoint; no refit/no unseen continuation',
        test_labels_used=False, selection_role='diagnostic only; never used for candidate or policy choice')
    for ref in case['trajectories']:
        if ref['mode'] == 'full':
            continue
        path_kind = ref['mode']
        pin(ref['path'], ref['sha256'])
        trajectory = locked(ref['path'], ref['sha256'])
        histories = {kind: AbsoluteHistory(kind, c, task, .8) for kind in KINDS[1:]}
        estimates = {}
        for scores in warm_states:
            for kind, history in histories.items():
                estimates[kind] = history.observe(labels['train'], scores, 1., rank)
        checkpoint, protected = warm_model, {k: a.copy() for k, a in warm_raw.items()}
        shadow = {k: a.copy() for k, a in warm_raw.items()}
        sum_local_mse = {k: 0. for k in ROLES}
        for cycle in trajectory['cycles']:
            block = cycle['block']
            fit = cycle['fit']
            pin(fit['artifact'], fit['sha256'])
            pin(cycle['initial_path'], cycle['initial_sha256'])
            inc = native_load(fit['artifact'])
            if task == 'multiclass':
                inc = replace(inc, base=reference_class_model(inc.base))
            dense = full_output_base(inc)
            designs = {k: leaf_design(dense.leaf_indices(x), len(dense.coefficients)) for k, x in features.items()}
            contributions = {k: np.asarray(d@dense.coefficients.astype(float)) for k, d in designs.items()}
            teacher = {k: protected[k]+z+dense.bias for k, z in contributions.items()}
            mean, functional, singular = functional_basis(contributions['train'], task)
            centered = contrast(contributions['train']-mean, task)
            bases = dict(functional=functional, **{k: e.basis for k, e in estimates.items()})
            if path_kind != 'functional':
                pin(cycle['history_state_path'], cycle['history_state_sha256'])
                with np.load(cycle['history_state_path'], allow_pickle=False) as saved:
                    np.testing.assert_allclose(bases[path_kind], saved['basis'], atol=2e-7, rtol=2e-6)
            reference = best_policy(teacher, labels, task, cfg)
            prior = best_policy(protected, labels, task, cfg)
            reference_selection = quality(labels['selection'], teacher['selection'], task, reference)
            prior_selection = quality(labels['selection'], protected['selection'], task, prior)
            own = None
            own_rows = None
            replay_error = 0.
            for basis_kind, basis in bases.items():
                rows, pure = [], None
                for candidate in projection_candidates(checkpoint, inc, designs, protected, labels,
                        basis, mean, teacher, rank):
                    fixed = {k: quality(labels[k], candidate.raw[k], task, reference)
                        for k in ('control', 'selection')}
                    f = {k: fidelity(candidate.raw[k]-teacher[k], contributions[k], task)
                        for k in ('train', 'control', 'selection')}
                    if candidate.percent == 0:
                        pure = dict(cost_bytes=candidate.cost['cost_bytes'], fidelity=f,
                            train_weighted_mse=candidate.train_weighted_mse)
                    for v in variants(candidate.raw, labels, task, cfg):
                        sel = quality(labels['selection'], candidate.raw['selection'], task, v)
                        row = dict(percent=candidate.percent, rank=rank, nnz=len(candidate.increment.values),
                            denominator=inc.parameter_count, cost_bytes=candidate.cost['cost_bytes'],
                            train_weighted_mse=candidate.train_weighted_mse, **v,
                            selection=sel, fidelity=f,
                            control_guard=guard_details(v['validation'], reference['validation']),
                            selection_guard=guard_details(sel, reference_selection),
                            previous_control_guard=guard_details(v['validation'], prior['validation']),
                            previous_selection_guard=guard_details(sel, prior_selection),
                            frozen_reference_policy=fixed,
                            frozen_control_guard=guard_details(fixed['control'], reference['validation']),
                            frozen_selection_guard=guard_details(fixed['selection'], reference_selection))
                        rows.append(row)
                    if basis_kind == path_kind and cycle['action'] != 'rejected' and \
                            candidate.percent == cycle['projection']['chosen']['percent']:
                        own = candidate
                chosen = pick(rows, case['budget_bytes'], reference['validation'])
                fallback = pick(rows, case['budget_bytes'])
                previous_guard_choice = pick(rows, case['budget_bytes'], prior['validation'])
                mode_result = dict(path=path_kind, block=block, basis=basis_kind,
                    increment_sha256=fit['sha256'],
                    increment_effective_rank90=int(np.searchsorted(np.cumsum(singular**2), .9*np.sum(singular**2))+1),
                    increment_capture=captured_energy(centered, basis[:, :rank]),
                    signal_capture=captured_energy(
                        __import__('abl_history_state_audit').independent_signal(
                            labels['train'], protected['train'], task, 1.), basis[:, :rank]),
                    pca_retained=float(np.sum(singular[:rank]**2)/np.sum(singular**2)) if np.sum(singular**2) else None,
                    pure=pure, original_budget_bytes=case['budget_bytes'],
                    stop=stop_cause(rows, case['budget_bytes'], reference['validation']).value,
                    chosen=record_choice(rows, chosen),
                    without_local_guard=record_choice(rows, fallback),
                    guard_against_previous=record_choice(rows, previous_guard_choice),
                    reference_control=reference['validation'], reference_selection=reference_selection,
                    previous_control=prior['validation'], previous_selection=prior_selection, candidates=rows)
                result['basis_comparisons'].append(mode_result)
                if basis_kind == path_kind:
                    own_rows = rows
                    feasible = [r for r in rows if r['cost_bytes'] <= case['budget_bytes']]
                    replay_error = verify_rows(feasible, cycle['projection']['candidates'])
                    expected_accepted = cycle['action'] != 'rejected'
                    if (chosen is not None) != expected_accepted:
                        raise RuntimeError('original acceptance changed')
                    if expected_accepted:
                        a, b = rows[chosen], cycle['projection']['chosen']
                        for k in ('percent', 'temperature', 'threshold_fraction'):
                            if a[k] != b[k]:
                                raise RuntimeError('original chosen policy changed')
                    decision = {k: v for k, v in mode_result.items() if k not in ('candidates', 'pure')}
                    decision.update(original_action=cycle['action'], original_stop=trajectory['stop_reason'],
                        reference_policy=reference, previous_policy=prior,
                        feasible_variants=len(feasible), replay_max_error=replay_error)
                    result['stop_decisions'].append(decision)
            if own_rows is None:
                raise RuntimeError('original basis absent')
            result['original_replay'].append(dict(path=path_kind, block=block,
                variants=len(cycle['projection']['candidates']), maximum_error=replay_error,
                status='passed'))
            if cycle['action'] == 'rejected':
                break
            item = next(m for m in trajectory['models'] if m['block'] == block)
            pin(item['artifact'], item['sha256'])
            accepted = load_protected(item['artifact'])
            actual = {k: accepted.raw(x) for k, x in features.items()}
            for k in ROLES:
                np.testing.assert_allclose(actual[k], own.raw[k], atol=2e-8, rtol=2e-10)
            if [fingerprint(b) for b in checkpoint.blocks] != [fingerprint(b) for b in accepted.blocks[:-1]]:
                raise RuntimeError('old protected block changed')
            stats = {}
            for k in ROLES:
                previous_error = protected[k]-shadow[k]
                next_shadow = shadow[k]+contributions[k]+dense.bias
                local = actual[k]-teacher[k]
                total = actual[k]-next_shadow
                s = accumulation(previous_error, local, total, task)
                if s['identity_max_abs'] > 2e-8:
                    raise RuntimeError('additive error identity failed')
                sum_local_mse[k] += s['local_mse']
                s['sum_local_mse'] = sum_local_mse[k]
                s['total_over_sum_local'] = s['cumulative_mse']/sum_local_mse[k] if sum_local_mse[k] else None
                s['local_relative_to_increment'] = fidelity(local, contributions[k], task)['relative_centered_mse']
                stats[k] = s
                shadow[k] = next_shadow
            chosen_policy = cycle['projection']['chosen']
            result['accumulation'].append(dict(path=path_kind, block=block, roles=stats,
                selection_actual=quality(labels['selection'], actual['selection'], task, chosen_policy),
                selection_shadow=quality(labels['selection'], shadow['selection'], task, chosen_policy),
                selection_actual_raw=task_quality(labels['selection'], actual['selection'], task),
                selection_shadow_raw=task_quality(labels['selection'], shadow['selection'], task),
                old_blocks_unchanged=True,
                scope='path-conditional dense sum of the SAME saved increments; not an independently trained full model'))
            checkpoint, protected = accepted, actual
            for kind, history in histories.items():
                estimates[kind] = history.observe(labels['train'], protected['train'], 1., rank)
            memory_guard()
            del designs, contributions, teacher, bases, rows, own, own_rows
            gc.collect()
        event('increment_diagnostic_path', case=cid, path=path_kind,
            blocks=len(trajectory['cycles']), stop=trajectory['stop_reason'])
    path = out/'cases'/(cid+'.json')
    write(path, result)
    event('increment_diagnostic_case', case=cid, path=str(path),
        paired_blocks=len(result['stop_decisions']), sha256=digest(path))
    return dict(case=cid, path=str(path), sha256=digest(path))


def summarize(cases):
    from collections import Counter
    all_cases = [locked(e['path'], e['sha256']) for e in cases]
    summary = {}
    for dataset in ('yeast', 'mediamill', 'mnist'):
        group = [c for c in all_cases if c['case'].startswith(dataset+'_')]
        first = [r for c in group for r in c['basis_comparisons']
            if r['path'] == 'functional' and r['block'] == 1]
        primary = {}
        for kind in KINDS:
            rows = [r for r in first if r['basis'] == kind]
            primary[kind] = dict(cases=len(rows),
                mean_increment_capture=float(np.mean([r['increment_capture'] for r in rows])),
                mean_signal_capture=float(np.mean([r['signal_capture'] for r in rows])),
                mean_weighted_mse=float(np.mean([r['pure']['train_weighted_mse'] for r in rows])),
                local_guard_pass=sum(r['chosen'] is not None for r in rows),
                mean_rank90=float(np.mean([r['increment_effective_rank90'] for r in rows])))
        paths = {}
        for kind in KINDS:
            decisions = [r for c in group for r in c['stop_decisions'] if r['path'] == kind]
            terminal = [r for c in group for r in c['stop_decisions'] if r['path'] == kind and
                r['block'] == max(a['block'] for a in c['stop_decisions'] if a['path'] == kind)]
            acc = [r for c in group for r in c['accumulation'] if r['path'] == kind and
                r['block'] == max(a['block'] for a in c['accumulation'] if a['path'] == kind)]
            rejected = [r for r in terminal if r['stop'] == 'local_quality']
            paths[kind] = dict(blocks=len(decisions), terminal_stop=dict(Counter(r['stop'] for r in terminal)),
                rejected_with_fallback_improving_selection=sum(r['without_local_guard'] is not None and
                    r['without_local_guard']['selection']['f1_macro'] > r['previous_selection']['f1_macro'] for r in rejected),
                rejected_with_fallback_passing_selection_teacher=sum(r['without_local_guard'] is not None and
                    r['without_local_guard']['selection_guard']['passed'] for r in rejected),
                rejected_with_fallback_passing_selection_previous=sum(r['without_local_guard'] is not None and
                    r['without_local_guard']['previous_selection_guard']['passed'] for r in rejected),
                chosen_control_pass_selection_fail=sum(r['chosen'] is not None and
                    not r['chosen']['selection_guard']['passed'] for r in decisions),
                accepted_blocks=sum(r['chosen'] is not None for r in decisions),
                accumulated_cases=len(acc),
                mean_terminal_total_over_sum_local=(float(np.mean([r['roles']['selection']['total_over_sum_local']
                    for r in acc])) if acc else None),
                mean_terminal_f1_actual_minus_shadow_pp=(float(np.mean([100*(r['selection_actual']['f1_macro']-
                    r['selection_shadow']['f1_macro']) for r in acc])) if acc else None))
        summary[dataset] = dict(first_shared_increment=primary, path_diagnostics=paths)
    return dict(datasets=summary, original_blocks=sum(len(c['original_replay']) for c in all_cases),
        basis_blocks=sum(len(c['basis_comparisons']) for c in all_cases),
        accepted_prefixes=sum(len(c['accumulation']) for c in all_cases))


def run(cfg, out):
    if cfg['phase'] != 'abl_increment_diagnostic' or cfg['abl_increment_diagnostic'] != dict(
            tree_training_allowed=False, gpu_execution=False, test_labels_used=False,
            rank='original_half', residual_percentages=[0, 1, 2, 5], tolerance=.05,
            basis_kinds=list(KINDS), include_cases=9):
        raise RuntimeError('diagnostic contract changed')
    started = perf_counter()
    environment = check_native_environment(cfg)
    ref = cfg['abl_protected_development_completed']
    source = locked(ref['path'], ref['sha256'])
    locks = {ref['path']: ref['sha256']}
    if len(source['cases']) != 9:
        raise RuntimeError('all nine saved cases required')
    cases = []
    event('increment_diagnostic_start', contract=cfg['abl_increment_diagnostic'], source=ref,
        native_runtime=sys.executable, tree_fits=0)
    for entry in source['cases']:
        cases.append(diagnose_case(entry, cfg, out, locks))
        write(out/'progress.json', dict(cases=cases, tree_fits=0, test_labels_used=False))
    for p, sha in locks.items():
        if digest(p) != sha:
            raise RuntimeError('source changed during diagnostics: '+p)
    if any(k.split('.')[0] in ('cupy', 'torch', 'py_boost') for k in sys.modules):
        raise RuntimeError('diagnostic unexpectedly imported a training/GPU runtime')
    summary = summarize(cases)
    from audit_abl_increment_diagnostic import audit
    audit_result = audit(cases, cfg)
    write(out/'audit_report.json', audit_result)
    report = dict(status='saved_increment_diagnostic_complete', cases=cases, summary=summary,
        audit=dict(path=str(out/'audit_report.json'), sha256=digest(out/'audit_report.json')),
        source=ref, immutable_inputs=locks, new_tree_fits=0, gpu_execution=False,
        test_labels_used=False, environment=environment, cpu_peak_rss_bytes=memory_guard(),
        elapsed_seconds=perf_counter()-started,
        selection_scope='previously used development selection; diagnostic, not an independent new confirmation',
        limitations=['No continuation beyond the saved first rejected block.',
            'Dense shadow sums the same compact-path increments; it is not a full-trained counterfactual.',
            'Pure bases have exactly equal byte costs; hybrids share the same cap and record actual nnz/cost.',
            'Repeated blocks and splits are dependent; no dataset-level significance claim.'])
    write(out/'report.json', report)
    event('increment_diagnostic_complete', cases=len(cases), new_tree_fits=0, path=str(out/'report.json'),
        sha256=digest(out/'report.json'), cpu_peak_rss_bytes=report['cpu_peak_rss_bytes'], summary=summary)


def main():
    from threadpoolctl import threadpool_limits
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(sys.argv[2])
    with threadpool_limits(limits=1):
        run(cfg, out)


if __name__ == '__main__':
    main()
