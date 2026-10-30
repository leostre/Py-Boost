from dataclasses import replace
import numpy as np
import pytest
import budget_quality_worker
from test_abl_quality_core import fixture
from abl_increment_diagnostic_core import (
    accumulation, guard_details, stop_cause, StopCause, projection_candidates, task_quality)
from abl_protected_core import as_protected, project_new_increment
from abl_quality_core import full_output_base
from abl_training_core import calibrated_local_variants
from experiments.shared_leaf_basis import functional_basis
from experiments.sparse_leaf_core import leaf_design


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
def test_accumulation_identifies_alignment_and_cancellation(task):
    a = np.array([[1., -2., 1.], [-1., 2., -1.]])
    same = accumulation(a, a, 2*a, task)
    cancel = accumulation(a, -a, np.zeros_like(a), task)
    assert same['identity_max_abs'] == 0
    assert same['amplification'] == pytest.approx(2)
    assert same['cosine'] == pytest.approx(1)
    assert cancel['amplification'] == 0
    assert cancel['cosine'] == pytest.approx(-1)
    assert same['cumulative_mse'] == pytest.approx(
        same['previous_mse']+same['local_mse']+same['cross_term'])


def test_softmax_common_shift_is_not_compression_error():
    a = np.arange(4.)[:, None]*np.ones((1, 3))
    row = accumulation(a, a, 2*a, 'multiclass')
    assert row['cumulative_mse'] == 0
    assert row['cosine'] is None


def test_stop_reason_separates_storage_and_metric_failures():
    ref = dict(loss=1., f1_macro=.5, f1_weighted=.7)
    bad = dict(loss=1., f1_macro=.4, f1_weighted=.7)
    rows = [dict(cost_bytes=12, validation=bad), dict(cost_bytes=20, validation=ref)]
    assert stop_cause(rows, 10, ref) is StopCause.MEMORY
    assert stop_cause(rows, 15, ref) is StopCause.QUALITY
    assert stop_cause(rows, 20, ref) is StopCause.ACCEPTED
    assert guard_details(bad, ref)['failures'] == ['f1_macro']
    assert guard_details(ref, dict(loss=1., f1_macro=0., f1_weighted=0.))['passed']


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
def test_diagnostic_replays_existing_projection_without_changing_parent(task):
    old, x, _, _ = fixture(11, 4, 3)
    inc, _, _, _ = fixture(21, 4, 2)
    if task == 'multilabel':
        old = replace(old, base=replace(old.base, task=task))
        inc = replace(inc, base=replace(inc.base, task=task))
    checkpoint = as_protected(old)
    features = dict(train=x[:20], calibration=x[20:28], control=x[28:])
    target = np.arange(len(x)) % 4 if task == 'multiclass' else (
        np.arange(len(x))[:, None]+np.arange(4)[None, :]) % 3 == 0
    labels = dict(train=target[:20], calibration=target[20:28], control=target[28:])
    development = {k: (features[k], labels[k]) for k in features}
    dense = full_output_base(inc)
    designs = {k: leaf_design(dense.leaf_indices(v), len(dense.coefficients)) for k, v in features.items()}
    protected = {k: checkpoint.raw(v) for k, v in features.items()}
    teacher = {k: protected[k]+np.asarray(d@dense.coefficients.astype(float))+dense.bias for k, d in designs.items()}
    mean, basis, _ = functional_basis(np.asarray(designs['train']@dense.coefficients.astype(float)), task)
    cfg = dict(margin_scale=.5, decision_grid=[0, .5, -.5, 1, -1, 2, -2], decision_shrinkage=[.25, .5, 1.])
    expected, info = project_new_increment(checkpoint, inc, development, 100000, cfg)
    candidates = list(projection_candidates(checkpoint, inc, designs, protected, labels, basis, mean, teacher,
        max(1, (4-int(task == 'multiclass'))//2)))
    rows = []
    for item in candidates:
        for v in calibrated_local_variants(item.raw['calibration'], labels['calibration'],
                item.raw['control'], labels['control'], task, cfg['decision_grid'], cfg['decision_shrinkage']):
            rows.append((item, v))
    actual = [dict(percent=i.percent, train_weighted_mse=i.train_weighted_mse,
        validation=v['validation']) for i, v in rows]
    assert len(actual) == len(info['candidates'])
    for a, b in zip(actual, info['candidates']):
        assert a['percent'] == b['percent']
        assert a['train_weighted_mse'] == pytest.approx(b['train_weighted_mse'], abs=1e-14)
        assert a['validation'] == pytest.approx(b['validation'], abs=1e-14)
    if expected is not None:
        item = next(i for i in candidates if i.percent == info['chosen']['percent'])
        np.testing.assert_allclose(item.raw['control'], expected.raw(features['control']), atol=2e-8)
    np.testing.assert_array_equal(checkpoint.raw(x), as_protected(old).raw(x))


def test_vectorized_multilabel_quality_has_explicit_zero_support_domain():
    from audit_abl_transfer import independent_task_quality
    y = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
    z = np.array([[1., -1., -2.], [-1., 1., -2.], [.5, -1., -2.]])
    q = task_quality(y, z, 'multilabel', shifts=np.array([.2, .3, 0]))
    expected = independent_task_quality(y, z, dict(temperature=1., shifts=[.2, .3, 0]), 'multilabel')
    assert q == pytest.approx(expected, abs=1e-14)


def test_saved_artifact_diagnostic_boundary_and_independent_audit(tmp_path):
    from budget_quality_worker import digest, write
    from abl_increment_diagnostic import diagnose_case
    from audit_abl_increment_diagnostic import audit
    from experiments.shared_leaf_basis import reference_class_model
    from experiments.absolute_history_core import AbsoluteHistory
    from experiments.parallel_leaf_core import prefix
    old, x, _, _ = fixture(22, 4, 40)
    old = replace(old, base=reference_class_model(old.base))
    checkpoint = as_protected(old)
    inc, _, _, _ = fixture(33, 4, 40)
    inc_path = tmp_path/'increment.npz'
    inc.save(inc_path)
    inc = replace(inc, base=reference_class_model(inc.base))
    target = checkpoint.raw(x).argmax(1)
    target[:4] = np.arange(4)  # Explicit training domain.
    roles = dict(train=np.arange(20), calibration=np.arange(20, 28), control=np.arange(28, 32),
        selection=np.arange(32, 37), test=np.empty(0, dtype=int))
    data_path, split_path = tmp_path/'data.npz', tmp_path/'roles.npz'
    np.savez(data_path, features=x, target=target)
    np.savez(split_path, **roles)
    warm_path, teacher_path = tmp_path/'warmup.npz', tmp_path/'teacher.npz'
    checkpoint.save(warm_path)
    old.save(teacher_path)
    warm = dict(models=[dict(block=0, artifact=str(warm_path), sha256=digest(warm_path))],
        warmup_threshold=dict(artifact=str(teacher_path), sha256=digest(teacher_path)))
    warm_json = tmp_path/'warmup.json'
    write(warm_json, warm)
    cfg = dict(margin_scale=.5, decision_grid=[0, .5, -.5, 1, -1, 2, -2],
        decision_shrinkage=[.25, .5, 1.])
    development = {k: (x[roles[k]], target[roles[k]]) for k in ('train', 'calibration', 'control')}
    refs = []
    for mode in ('functional', 'current', 'mean', 'history'):
        estimate = None
        history = None
        if mode != 'functional':
            history = AbsoluteHistory(mode, 4, 'multiclass', .8)
            for count in (0, 10, 20, 30, 40):
                scores = np.broadcast_to(old.base.bias, (20, 4)).copy() if not count else (
                    prefix(old, count).prepare_inference().raw(x[:20]))
                estimate = history.observe(target[:20], scores, 1., 1)
        accepted, info = project_new_increment(checkpoint, inc, development, 1000000, cfg, estimate)
        initial = tmp_path/(mode+'-initial.npz')
        np.savez(initial, train=checkpoint.raw(x[:20]).astype(np.float32), validation=checkpoint.raw(x[20:28]))
        cycle = dict(block=1, fit=dict(artifact=str(inc_path), sha256=digest(inc_path)),
            initial_path=str(initial), initial_sha256=digest(initial), projection=info,
            action='rejected' if accepted is None else 'projected_increment')
        if history is not None:
            hp = tmp_path/(mode+'-history.npz')
            np.savez(hp, basis=estimate.basis)
            cycle.update(history_state_path=str(hp), history_state_sha256=digest(hp))
        models = list(warm['models'])
        if accepted is not None:
            path = tmp_path/(mode+'-accepted.npz')
            accepted.save(path)
            models.append(dict(block=1, artifact=str(path), sha256=digest(path)))
        tp = tmp_path/(mode+'.json')
        write(tp, dict(models=models, cycles=[cycle], stop_reason='horizon' if accepted else info['status']))
        refs.append(dict(mode=mode, path=str(tp), sha256=digest(tp)))
    cp = tmp_path/'case.json'
    write(cp, dict(case_id='fixture_1', task='multiclass', budget_bytes=1000000,
        record=dict(case=dict(task='multiclass', outputs=4, split_file=str(split_path), split_sha256=digest(split_path)),
            dataset=dict(path=str(data_path), sha256=digest(data_path))),
        warmup=dict(path=str(warm_json), sha256=digest(warm_json)), trajectories=refs))
    result = diagnose_case(dict(case_id='fixture_1', path=str(cp), sha256=digest(cp)), cfg, tmp_path/'out', {})
    checked = audit([result], cfg)
    assert checked['status'] == 'passed'
    assert checked['cases'][0]['paired_blocks'] == 4
    assert checked['new_tree_fits'] == 0
