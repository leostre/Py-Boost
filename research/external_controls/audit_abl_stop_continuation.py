"""Independent SR2 decisions, moments, physical bytes, gradients and test audit."""
from pathlib import Path
import numpy as np
from budget_quality_worker import locked, digest, native_load
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from audit_abl_protected import physical_cost
from audit_abl_transfer import independent_task_quality
from audit_abl_stop_rule import passed, final_choice
from abl_history_state_audit import independent_signal
from experiments.parallel_leaf_core import prefix


def assert_metrics(actual, expected):
    for key in ('loss', 'f1_macro', 'f1_weighted'):
        assert abs(actual[key]-expected[key]) < 2e-8, (key, actual[key], expected[key])


def check_state_moments(file, kind, scores, y, task):
    """Recompute from score history, independently of AbsoluteHistory/state loader."""
    mass = 0.; mean = None; gram = None; halves = None
    for score in scores:
        z = independent_signal(y, score, task, 1.)
        if kind == 'current':
            gram = z.T@z; halves = [v.T@v for v in (z[::2], z[1::2])]; mass = 1.
        elif kind == 'mean':
            mean = .8*(np.zeros_like(z) if mean is None else mean)+.2*z
            mass = .8*mass+.2
        else:
            g = z.T@z; hh = [v.T@v for v in (z[::2], z[1::2])]
            gram = .8*(np.zeros_like(g) if gram is None else gram)+.2*g
            halves = [.8*a+.2*b for a, b in zip([np.zeros_like(g)]*2 if halves is None else halves, hh)]
            mass = .8*mass+.2
    assert digest(file['path']) == file['sha256']
    with np.load(file['path'], allow_pickle=False) as a:
        assert int(a['observations']) == len(scores) and abs(float(a['mass'])-mass) < 1e-14
        for name, value in (('mean', mean), ('gram', gram), ('halves', halves)):
            absent = np.empty((0, 0, 0)) if name == 'halves' else np.empty((0, 0))
            np.testing.assert_allclose(a[name], absent if value is None else value, atol=2e-13, rtol=2e-12)
        basis = a['basis'].copy()
    g = (mean/mass).T@(mean/mass) if kind == 'mean' else gram/mass
    c = len(g)
    q = np.linalg.qr(np.vstack((np.eye(c-1), -np.ones(c-1))))[0] if task == 'multiclass' else np.eye(c)
    values = np.linalg.eigvalsh(q.T@g@q)[::-1]
    np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=2e-6)
    np.testing.assert_allclose(np.diag(basis.T@g@basis), values, atol=2e-7, rtol=2e-6)


def old_scores(sr, source, path, x):
    warm = locked(source['warmup']['path'], source['warmup']['sha256'])
    teacher = native_load(warm['warmup_threshold']['artifact'])
    scores = [(np.broadcast_to(teacher.base.bias, (len(x), len(teacher.base.bias))).copy() if trees == 0
        else prefix(teacher, trees).prepare_inference().raw(x)) for trees in (0, 10, 20, 30, 40)]
    block = path['attempts'][-1]['block']
    for b in range(1, block):
        meta = source['metas'][path['path']+f'/b{b:02d}']
        assert digest(meta['artifact']) == meta['sha256']
        scores.append(load_protected(meta['artifact']).raw(x))
    scores.append(load_protected(path['last_observed']['artifact']).raw(x))
    return scores


def check_decision(attempt):
    before = attempt['before']; rows = attempt['candidate_rows']; ref = before['anchor']['quality']
    feasible = [r for r in rows if r['cost_bytes'] <= before['budget_bytes']]
    useful = [r for r in feasible if before['current']['raw_nll']-r['raw_nll'] > 1e-6]
    qualified = [r for r in useful if passed(r['quality'], ref)]
    reason = ('no_candidate_within_memory' if not feasible else 'no_useful_candidate' if not useful else
        'no_useful_candidate_within_fixed_quality_guard' if before['phase'] == 'qualified' and not qualified else None)
    for row, gate in zip(rows, attempt['gates']):
        assert gate['uid'] == row['uid']
        assert gate['memory'] == (row['cost_bytes'] <= before['budget_bytes'])
        assert gate['useful'] == (before['current']['raw_nll']-row['raw_nll'] > 1e-6)
        assert gate['fixed_quality'] == passed(row['quality'], ref)
    if reason is not None:
        assert attempt['action'] == 'stopped' and attempt['reason'] == reason
        return None
    winner = min(qualified or useful, key=lambda c:(c['reconstruction_mse'], -c['quality']['f1_macro'],
        c['quality']['loss'], c['cost_bytes'], c['ordinal'], c['uid']))
    assert attempt['action'] == 'accepted' and attempt['selected_uid'] == winner['uid']
    after = attempt['after']; assert after['current'] == winner
    assert after['anchor'] == before['anchor'] and after['budget_bytes'] == before['budget_bytes']
    assert after['phase'] == ('qualified' if passed(winner['quality'], ref) else 'seeking_quality')
    assert attempt['entered_qualification'] == (before['phase'] == 'seeking_quality' and after['phase'] == 'qualified')
    return winner


def audit_development(entries, ledger):
    observed = {}; checks = []; intervals = []
    assert len(entries) == 16
    assert ledger['reserved_rounds'] == 40*len(ledger['attempts']) <= 1320 and ledger['phase_limit'] == 1320
    assert all(a['state'] == 'complete' for a in ledger['attempts'])
    for entry in entries:
        t = locked(entry['path'], entry['sha256']); job = t['job']; cid, kind = job['case'], job['kind']
        sr = locked(t['source']['path'], t['source']['sha256'])
        source = locked(sr['original']['path'], sr['original']['sha256'])
        data = data_before_test(source['record']); nc, nt, _ = data['sizes']; task = source['task']
        old = next(p for p in sr['paths'] if p['path'] == kind)
        assert old['censored'] and old['attempts'][-1]['block'] == job['start_block']
        previous_snapshot = old['last_observed']; previous = load_protected(previous_snapshot['artifact'])
        state = old['attempts'][-1]['after']; scores = None
        if kind != 'functional':
            scores = old_scores(sr, source, old, data['train_x'])
            check_state_moments(t['preflight']['history'], kind, scores, data['train_y'], task)
        assert t['training_roles'] == ['train', 'calibration', 'control']
        assert not t['selection_used_for_training'] and not t['test_accessed']
        for i, attempt in enumerate(t['attempts']):
            b = attempt['block']; fit = attempt['fit']; key = (cid, kind, b)
            assert key not in observed; observed[key] = fit
            assert b == job['start_block']+i+1 <= 4
            assert attempt['before'] == state and attempt['initial_model'] == previous_snapshot
            assert state['anchor']['token'] == sr['anchor']['token']
            assert state['anchor']['quality'] == sr['anchor']['control']
            assert state['budget_bytes'] == sr['budget_bytes'] == source['budget_bytes']
            assert digest(attempt['initial_path']) == attempt['initial_sha256'] == fit['initial_sha256']
            with np.load(attempt['initial_path'], allow_pickle=False) as a:
                np.testing.assert_array_equal(a['train'], previous.raw(data['train_x']).astype(np.float32))
                np.testing.assert_array_equal(a['validation'], previous.raw(data['dev_x'][:nc]))
            assert fit['trees'] == 40 and fit['minimum_leaf'] == 10 and fit['lr'] == .1
            assert fit['observed_minimum_leaf_support'] >= 10
            assert fit['seed'] == source['record']['case']['seed']+50000+40*b
            assert fit['training_pool_peak_bytes'] <= 16*1024**3
            assert max(fit['train_replay_error'], fit['validation_replay_error']) <= 2e-5
            assert digest(fit['artifact']) == fit['sha256']
            assert native_load(fit['artifact']).base.n_trees == 40
            intervals.append((fit['fit_started_ns'], fit['fit_finished_ns']))
            if scores is not None:
                check_state_moments(attempt['history'], kind, scores, data['train_y'], task)
            # Audit all decision rows against actually reconstructed candidates.
            from dataclasses import replace
            from abl_quality_core import full_output_base
            from abl_increment_diagnostic_core import projection_candidates
            from experiments.sparse_leaf_core import leaf_design
            from experiments.shared_leaf_basis import reference_class_model, functional_basis
            inc = native_load(fit['artifact'])
            if task == 'multiclass':
                inc = replace(inc, base=reference_class_model(inc.base))
            dense = full_output_base(inc)
            xx = dict(train=data['train_x'], calibration=data['dev_x'][:nc], control=data['dev_x'][nc:nc+nt])
            yy = dict(train=data['train_y'], calibration=data['dev_y'][:nc], control=data['dev_y'][nc:nc+nt])
            designs = {k:leaf_design(dense.leaf_indices(x), len(dense.coefficients)) for k, x in xx.items()}
            protected = {k:previous.raw(x) for k, x in xx.items()}
            contrib = {k:np.asarray(d@dense.coefficients.astype(float)) for k, d in designs.items()}
            mean = contrib['train'].mean(0)
            if kind == 'functional':
                mean, basis, _ = functional_basis(contrib['train'], task)
            else:
                with np.load(attempt['history']['path'], allow_pickle=False) as a:
                    basis = a['basis'].copy()
            teacher = {k:protected[k]+z+dense.bias for k, z in contrib.items()}
            assert_metrics(independent_task_quality(yy['control'], teacher['control'], previous_snapshot['policy'], task),
                attempt['full_step_quality_previous_policy'])
            for cand in projection_candidates(previous, inc, designs, protected, yy, basis, mean, teacher, attempt['rank']):
                for row, variant in zip(attempt['candidate_rows'], attempt['variants']):
                    if variant['percent'] != cand.percent:
                        continue
                    assert physical_cost(cand.combined) == row['cost_bytes']
                    cp = dict(temperature=variant['temperature'], shifts=variant['shifts'])
                    assert_metrics(independent_task_quality(yy['control'], cand.raw['control'], cp, task), row['quality'])
                    raw = independent_task_quality(yy['control'], cand.raw['control'],
                        dict(temperature=1., shifts=np.zeros(len(dense.bias))), task)
                    assert abs(raw['loss']-row['raw_nll']) < 2e-8
            winner = check_decision(attempt)
            if winner is None:
                assert i == len(t['attempts'])-1
                assert t['terminal'] == attempt['reason']; continue
            s = attempt['snapshot']; assert digest(s['artifact']) == s['sha256']
            model = load_protected(s['artifact']); assert model.n_trees == previous.n_trees+40 <= 200
            assert physical_cost(model) == s['cost_bytes'] == winner['cost_bytes'] <= sr['budget_bytes']
            np.testing.assert_array_equal(model.decision_shifts, s['policy']['shifts'])
            from abl_quality_core import fingerprint
            assert [fingerprint(v) for v in model.blocks[:-1]] == [fingerprint(v) for v in previous.blocks]
            q = independent_task_quality(yy['control'], model.raw(xx['control']), s['policy'], task)
            assert_metrics(q, s['control']); assert_metrics(q, winner['quality'])
            assert s['control_guard'] == passed(q, sr['anchor']['control'])
            state = attempt['after']; previous_snapshot = s; previous = model
            if scores is not None:
                scores.append(model.raw(data['train_x']))
        assert state == t['final_state']
        assert t['snapshots'] == [a['snapshot'] for a in t['attempts'] if a['action'] == 'accepted']
        assert t['cpu_peak_rss_bytes'] <= 32*1024**3
        checks.append(dict(case=cid, kind=kind, new_trees=40*len(t['attempts']), status='passed'))
    assert observed == {(a['case'], a['kind'], a['block']):a['fit'] for a in ledger['attempts']}
    intervals.sort()
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    return dict(status='passed', checks=checks, new_trees=ledger['reserved_rounds'],
        test_accessed=False, gpu_execution=False, training_repeated=False,
        scope='independent temporal moments, raw initial scores, all candidate metrics and physical costs, '
              'controller transitions, protected prefixes, paid fit ledger and sequential GPU intervals')


def audit_choices(cases):
    checks = []
    for case in cases:
        source = locked(case['original']['path'], case['original']['sha256'])
        data = data_before_test(source['record']); nc = sum(data['sizes'][:2]); task = case['task']
        for path in case['paths']:
            for s in path['snapshots']:
                assert digest(s['artifact']) == s['sha256']
                model = load_protected(s['artifact']); assert physical_cost(model) == s['cost_bytes'] <= case['budget_bytes']
                raw = model.raw(data['dev_x'])
                q = independent_task_quality(data['dev_y'][nc:], raw[nc:], s['policy'], task)
                assert_metrics(q, s['selection'])
                assert s['selection_guard'] == passed(q, case['anchor']['selection'])
                ncal = data['sizes'][0]
                qc = independent_task_quality(data['dev_y'][ncal:nc], raw[ncal:nc], s['policy'], task)
                assert_metrics(qc, s['control'])
                assert s['control_guard'] == passed(qc, case['anchor']['control'])
            assert path['final_qualified'] == final_choice([s for s in path['snapshots'] if s['control_guard']])
            assert path['diagnostic_best'] == final_choice(path['snapshots'])
        checks.append(dict(case=case['case'], status='passed', paths=len(case['paths'])))
    return dict(status='passed', cases=checks, test_accessed=False,
        scope='actual control/selection metrics, unchanged byte budgets and independently chosen final snapshots')


def audit_final(cases, evaluations, seal_sha):
    checks = []
    for case, entry in zip(cases, evaluations):
        source = locked(case['original']['path'], case['original']['sha256']); task = case['task']
        evaluation = locked(entry['path'], entry['sha256']); assert evaluation['pretest_seal_sha256'] == seal_sha
        assert not evaluation['test_used_for_selection'] and evaluation['new_tree_fits'] == 0
        assert digest(evaluation['target']['path']) == evaluation['target']['sha256']
        with np.load(evaluation['target']['path'], allow_pickle=False) as a:
            y = a['target'].copy(); ids = a['indices'].copy()
        record = source['record']
        with np.load(record['case']['split_file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(ids, a['test'])
        with np.load(record['dataset']['path'], allow_pickle=False) as a:
            np.testing.assert_array_equal(y, a['target'][ids]); x = a['features'][ids].astype(np.float32)
        raw = {}
        for sha, r in evaluation['predictions'].items():
            assert digest(r['path']) == r['sha256']
            with np.load(r['path'], allow_pickle=False) as a:
                raw[sha] = a['raw'].copy()
        full = next(p for p in evaluation['points'] if p['point'] == 'full')['test']
        for p in evaluation['points']:
            expected = case['full'] if p['point'] == 'full' else next(v for v in case['paths'] if v['path'] == p['path'])[p['point']]
            assert expected == p['snapshot']
            s = expected; assert digest(s['artifact']) == s['sha256']
            m = load_protected(s['artifact']); assert m.n_trees == p['trees']
            np.testing.assert_allclose(m.raw(x), raw[s['sha256']], atol=2e-10, rtol=1e-12)
            q = independent_task_quality(y, raw[s['sha256']], s['policy'], task)
            assert_metrics(q, p['test']); assert p['test_guard'] == passed(q, full)
        checks.append(dict(case=case['case'], points=len(evaluation['points']), status='passed'))
    return dict(status='passed', cases=checks, pretest_seal_sha256=seal_sha, training_repeated=False,
        scope='actual selection metrics and unfiltered final choice; sealed test points, exact old test '
              'indices/targets, replayed scores, independent metrics and fixed-full quality guard')
