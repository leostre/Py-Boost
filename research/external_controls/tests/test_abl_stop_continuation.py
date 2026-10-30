from dataclasses import asdict, replace
import json
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from abl_stop_continuation_core import (Job, EXPECTED_STARTS, plan, reserve, state_from_json, initial_arrays)
from abl_stop_rule_core import Anchor, Candidate, Quality, State, Phase, decide, Accepted


def sources():
    return [dict(case=c, paths=[dict(path=k, censored=True,
        terminal='censored_old_rejected_block_now_accepted', last_observed={'uid':k},
        attempts=[dict(block=b, action='accepted', snapshot={'uid':k})]) for k, b in kinds.items()])
        for c, kinds in EXPECTED_STARTS.items()]


def test_approved_plan_is_exactly_16_paths_1320_new_trees():
    jobs = plan(sources())
    assert len(jobs) == 16
    assert sum(j.maximum_new_trees for j in jobs) == 1320
    assert sum(j.maximum_new_trees for j in jobs if j.case.startswith('mediamill')) == 680
    assert sum(j.maximum_new_trees for j in jobs if j.case.startswith('mnist')) == 640
    with pytest.raises(ValueError):
        plan(sources()[:-1])
    rows = sources(); rows[0]['paths'][0]['attempts'][-1]['block'] = 1
    with pytest.raises(ValueError):
        plan(rows)


@settings(max_examples=25, deadline=None)
@given(st.permutations(tuple((c, k, b) for c, d in EXPECTED_STARTS.items() for k, b in d.items())))
def test_fit_ledger_conserves_paid_trees_under_job_reordering(order):
    ledger = dict(phase_limit=1320, attempts=[], reserved_rounds=0)
    seen = set()
    for c, k, b in order:
        job = Job(c, k, b)
        for block in job.blocks:
            before = ledger
            ledger = reserve(ledger, job, block)
            seen.add((c, k, block))
            assert len(before['attempts'])+1 == len(ledger['attempts'])
            assert ledger['reserved_rounds'] == 40*len(seen) <= 1320
            with pytest.raises(ValueError, match='duplicate'):
                reserve(ledger, job, block)
    assert ledger['reserved_rounds'] == 1320 and len(seen) == 33


def test_ledger_rejects_outside_horizon_or_inconsistent_paid_total():
    job = Job('mnist_3104', 'functional', 3)
    ledger = dict(phase_limit=1320, attempts=[], reserved_rounds=0)
    with pytest.raises(ValueError, match='horizon'):
        reserve(ledger, job, 3)
    with pytest.raises(ValueError, match='inconsistent'):
        reserve(dict(ledger, reserved_rounds=40), job, 4)
    with pytest.raises(ValueError, match='cap'):
        reserve(dict(ledger, phase_limit=1360), job, 4)


@pytest.mark.parametrize('qualified', [False, True])
def test_restored_controller_preserves_anchor_budget_and_certificate(qualified):
    q = Quality(1., .8, .9); anchor = Anchor('same-full-sha-and-policy', q)
    current = Candidate('saved', 'residual-5', 1., q, 100, .2, 7)
    state = State(anchor, 101, current, Phase.QUALIFIED if qualified else Phase.SEEKING,
        current if qualified else None)
    restored = state_from_json(json.loads(json.dumps(asdict(state))))
    assert restored == state
    new = replace(current, uid='next', raw_nll=.9)
    answer = decide(restored, [new]); assert isinstance(answer, Accepted)
    assert answer.state.anchor == anchor and answer.state.budget_bytes == 101


def test_initial_training_scores_ignore_external_policy():
    from test_abl_quality_core import fixture
    from abl_protected_core import as_protected
    m, x, _, _ = fixture(); m = as_protected(m)
    a = initial_arrays(m, x[:20], x[20:])
    b = initial_arrays(m.with_shifts(np.full(4, 2., np.float32)), x[:20], x[20:])
    np.testing.assert_array_equal(a['train'], b['train'])
    np.testing.assert_array_equal(a['validation'], b['validation'])
    assert a['train'].dtype == np.float32 and a['validation'].dtype == np.float64


@pytest.mark.parametrize('kind', ['current', 'mean', 'history'])
def test_history_resume_adds_newly_accepted_point_exactly_once(tmp_path, kind):
    from test_abl_quality_core import fixture
    from abl_protected_core import as_protected, append_protected
    from abl_stop_continuation import history_arrays, restore_history
    from audit_abl_stop_continuation import check_state_moments
    from budget_quality_worker import write, digest
    from experiments.absolute_history_core import AbsoluteHistory
    from experiments.parallel_leaf_core import prefix
    old, x, _, _ = fixture(11, 4, 40); inc, _, _, _ = fixture(22, 4, 40)
    base = as_protected(old); combined = append_protected(base, inc)
    y = np.arange(20)%4; xx = x[:20]
    oldp = tmp_path/'warm.npz'; old.save(oldp)
    warm = tmp_path/'warm.json'; write(warm, dict(warmup_threshold=dict(artifact=str(oldp), sha256=digest(oldp))))
    h = AbsoluteHistory(kind, 4, 'multiclass', .8); scores = []
    for trees in (0, 10, 20, 30, 40):
        score = np.broadcast_to(old.base.bias, (20, 4)).copy() if trees == 0 else prefix(old, trees).prepare_inference().raw(xx)
        scores.append(score); est = h.observe(y, score, 1., 1)
    before = tmp_path/'history.npz'; np.savez(before, **history_arrays(h, est))
    tr = tmp_path/'trajectory.json'; write(tr, dict(cycles=[dict(block=1, history_state_path=str(before), history_state_sha256=digest(before))]))
    checkpoint = tmp_path/'accepted.npz'; combined.save(checkpoint)
    source = dict(record=dict(case=dict(outputs=4)), task='multiclass',
        warmup=dict(path=str(warm), sha256=digest(warm)), trajectories=[dict(mode=kind, path=str(tr), sha256=digest(tr))])
    path = dict(path=kind, attempts=[dict(block=1)], last_observed=dict(artifact=str(checkpoint), sha256=digest(checkpoint)))
    restored, estimate = restore_history({}, path, source, dict(train_x=xx, train_y=y))
    assert restored.state.observations == 6
    resumed = tmp_path/'resumed.npz'; np.savez(resumed, **history_arrays(restored, estimate))
    scores.append(combined.raw(xx))
    check_state_moments(dict(path=str(resumed), sha256=digest(resumed)), kind, scores, y, 'multiclass')
    # A missing or double observation must fail the independent moment check.
    with pytest.raises(AssertionError):
        check_state_moments(dict(path=str(resumed), sha256=digest(resumed)), kind, scores[:-1], y, 'multiclass')


def test_trajectory_shell_reserves_before_one_fake_fit_and_keeps_source(tmp_path, monkeypatch):
    from test_abl_stop_rule import test_replay_boundary_with_synthetic_saved_models_and_independent_audit
    from budget_quality_worker import write, digest
    from abl_stop_continuation import trajectory
    from abl_protected_core import load_protected
    test_replay_boundary_with_synthetic_saved_models_and_independent_audit(tmp_path)
    srp = tmp_path/'rule/cases/fixture_1/case.json'; sr = json.loads(srp.read_text())
    source = json.loads((tmp_path/'case.json').read_text()); source['record']['case']['seed'] = 1
    write(tmp_path/'case.json', source); sr['original']['sha256'] = digest(tmp_path/'case.json')
    path = next(p for p in sr['paths'] if p['path'] == 'functional')
    # Choose a synthetic starting state; only block4 can be paid in this wiring test.
    snapshot = path['last_observed']; state = path['attempts'][-1].get('after', path['initial_state'])
    path['attempts'] = [dict(block=3, action='accepted', after=state, snapshot=snapshot)]
    write(srp, sr)
    ledger = tmp_path/'ledger.json'; write(ledger, dict(phase_limit=1320, attempts=[], reserved_rounds=0))
    start_sha = snapshot['sha256']; calls = []
    def fake_fit(request):
        paid = json.loads(ledger.read_text()); assert paid['reserved_rounds'] == 40
        assert paid['attempts'][-1]['state'] == 'reserved'
        with np.load(request['initial_path']) as a:
            actual = load_protected(snapshot['artifact']).raw(__import__('abl_transfer_study').data_before_test(source['record'])['train_x'])
            np.testing.assert_array_equal(a['train'], actual.astype(np.float32))
        calls.append(request)
        from test_abl_quality_core import fixture
        inc, _, _, _ = fixture(39, 4, 40); inc.save(request['output'])
        return dict(artifact=request['output'], sha256=digest(request['output']), trees=40)
    # Force rejection after real projection/calibration to test paid rejected block.
    from abl_stop_rule_core import Stopped, StopReason
    def stopped(state, candidates):
        return Stopped(state, StopReason.PROGRESS, ())
    monkeypatch.setattr('abl_stop_continuation.decide', stopped)
    request = dict(job=asdict(Job('fixture_1', 'functional', 3)), source=dict(path=str(srp), sha256=digest(srp)),
        config=dict(lr=.1, lambda_l2=1., ntrees=800, decision_grid=[0, .5, -.5], decision_shrinkage=[.5, 1.]),
        ledger_path=str(ledger), preflight={})
    trajectory(request, tmp_path/'continuation', fake_fit)
    result = json.loads((tmp_path/'continuation/trajectory.json').read_text())
    assert len(calls) == 1 and len(result['attempts']) == 1
    assert result['snapshots'] == [] and result['terminal'] == 'no_useful_candidate'
    assert digest(snapshot['artifact']) == start_sha
    assert json.loads(ledger.read_text())['reserved_rounds'] == 40
