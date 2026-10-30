import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.metrics import f1_score
from residual_gate_core import (ALPHAS, CalibrationMode, blend_raw, quality, sufficient,
    bootstrap_scores, decide, assess, fit_policies)


@given(st.lists(st.tuples(st.floats(-20, 20, allow_nan=False, allow_infinity=False), st.floats(-20, 20, allow_nan=False, allow_infinity=False)), min_size=3, max_size=15), st.sampled_from(ALPHAS))
@settings(max_examples=35, deadline=None)
def test_scalar_path_endpoints_and_no_source_mutation(values, alpha):
    base = np.asarray(values).reshape(-1, 2); corrected = base[::-1].copy(); before = base.copy()
    result = blend_raw(base, corrected, alpha)
    np.testing.assert_allclose(result, (1-alpha)*base+alpha*corrected, atol=1e-13)
    np.testing.assert_array_equal(base, before)
    assert not np.shares_memory(result, base)
    if alpha == 0: np.testing.assert_array_equal(result, base)
    if alpha == 1: np.testing.assert_array_equal(result, corrected)


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
def test_weighted_bootstrap_matches_explicit_row_resampling(task):
    rng = np.random.default_rng(5); n, c, repeats = 13, 4, 20
    y = rng.integers(0, 2, (n, c)) if task == 'multilabel' else rng.integers(0, c, n)
    raws = rng.normal(size=(3, n, c)); p = dict(temperature=1., shifts=[0.]*c)
    parts = [sufficient(y, z, task, p) for z in raws]
    predictions = np.stack([v[0] for v in parts]); losses = np.stack([v[1] for v in parts])
    bf, bl = bootstrap_scores(y, predictions, losses, task, c, repeats, 91)
    counts = np.random.default_rng(91).multinomial(n, np.full(n, 1/n), size=repeats)
    for b, weights in enumerate(counts):
        ids = np.repeat(np.arange(n), weights)
        for i in range(3):
            args = {'labels': np.arange(c)} if task == 'multiclass' else {}
            assert bf[b, i] == pytest.approx(f1_score(y[ids], predictions[i, ids], average='macro', zero_division=0, **args))
            assert bl[b, i] == pytest.approx(losses[i, ids].mean())


def test_gate_requires_benefit_and_nll_guard_and_prefers_less_amplitude():
    scores = [dict(f1_macro=a, loss=b) for a, b in zip([.5, .57, .6, .61, .9], [.5, .48, .47, .46, .8])]
    wave = np.linspace(-1, 1, 80)
    bf = np.asarray([q['f1_macro'] for q in scores])[None, :] + wave[:, None]*np.asarray([0, .15, .06, .03, .01])
    bl = np.tile(np.asarray([q['loss'] for q in scores]), (80, 1))
    d = decide(ALPHAS, scores, bf, bl)
    assert d.alpha == .25 and d.best_index == 3
    assert not d.candidates[4]['nll_guard_passed']
    assert not d.candidates[1]['benefit_passed']
    same = [scores[0]]*5
    assert decide(ALPHAS, same, np.full((80, 5), .5), np.full((80, 5), .5)).alpha == 0


def test_gate_skips_clear_harm_even_if_macro_f1_improves():
    scores = [dict(f1_macro=.5+i*.02, loss=.3+i*.01) for i in range(5)]
    bf = np.tile([q['f1_macro'] for q in scores], (10, 1)); bl = np.tile([q['loss'] for q in scores], (10, 1))
    assert decide(ALPHAS, scores, bf, bl).reason == 'no_admissible_benefit'


@pytest.mark.parametrize('mode', list(CalibrationMode))
def test_zero_correction_cannot_create_gate_benefit(mode):
    rng = np.random.default_rng(17); y = rng.integers(0, 2, (24, 3)); base = rng.normal(size=y.shape)
    raws = np.stack([base]*5); folds = [(tuple(range(12)), tuple(range(12, 24))), (tuple(range(12, 24)), tuple(range(12)))]
    result = assess(raws, y, 'multilabel', folds, mode, [0, -.5, .5], [.25, .5, 1.], 19, 30)
    assert result['decision'].alpha == 0
    for i in range(1, 5): np.testing.assert_array_equal(result['prediction'][i], result['prediction'][0])


def test_leaking_or_missing_validation_is_rejected():
    z = np.zeros((5, 8, 2)); y = np.zeros((8, 2))
    with pytest.raises(ValueError, match='leaking'):
        assess(z, y, 'multilabel', [((0, 1), (1, 2))], 'recalibrated', [0], [1.], 1)
    with pytest.raises(ValueError, match='cover'):
        assess(z, y, 'multilabel', [((0, 1), (2, 3))], 'recalibrated', [0], [1.], 1)


def test_multilabel_threshold_fit_matches_frozen_reference():
    import budget_quality_worker
    from experiments.output_calibration_core import fit_decision_shifts
    rng = np.random.default_rng(43); raw = rng.normal(size=(31, 6)); y = rng.integers(0, 2, raw.shape); y[:, 4] = 0
    grid = [0, .25, -.25, .5, -.5, 1., -1., 2., -2.]
    controls = fit_policies(raw, y, 'multilabel', grid, [.25, .5, 1.])
    np.testing.assert_array_equal(controls['raw_shift_1']['shifts'], fit_decision_shifts(raw, y, 'multilabel', grid))


def test_bootstrap_and_gate_do_not_change_global_rng():
    np.random.seed(44); before = np.random.get_state(); y = np.eye(3, dtype=int).repeat(4, axis=0)
    bootstrap_scores(y, np.stack([y, 1-y]), np.zeros((2, len(y))), 'multilabel', 3, 10, 4)
    after = np.random.get_state(); assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1]); assert before[2:] == after[2:]


def test_validation_labels_cannot_change_their_own_fitted_controls():
    rng = np.random.default_rng(8); base = rng.normal(size=(30, 4)); y = rng.integers(0, 2, base.shape)
    raws = np.stack([base+a*rng.normal(size=base.shape) for a in ALPHAS])
    folds = [(tuple(range(15)), tuple(range(15, 30))), (tuple(range(15, 30)), tuple(range(15)))]
    original = assess(raws, y, 'multilabel', folds, 'recalibrated', [-1., 0., 1.], [.25, .5, 1.], 2, 20)
    changed = y.copy(); changed[15:] = 1-changed[15:]
    counterfactual = assess(raws, changed, 'multilabel', folds, 'recalibrated', [-1., 0., 1.], [.25, .5, 1.], 2, 20)
    assert original['states'][0]['controls'] == counterfactual['states'][0]['controls']


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
@pytest.mark.parametrize('mode', list(CalibrationMode))
def test_nested_shell_on_toy_scores_preserves_outer_boundary_and_pooled_metrics(tmp_path, task, mode):
    from residual_gate_study import nested_case
    rng = np.random.default_rng(33); n, c = 48, 3
    y = rng.integers(0, 2, (n, c)) if task == 'multilabel' else np.tile(np.arange(c), n//c)
    base = rng.normal(size=(n, c)); corrected = base+rng.normal(size=base.shape)*.3
    raws = np.stack([blend_raw(base, corrected, a) for a in ALPHAS])
    cfg = dict(decision_grid=[0, -.5, .5], decision_shrinkage=[.25, .5, 1.])
    result = nested_case(raws, y, task, mode, cfg, 83, tmp_path, np.arange(n)+100, [10, 20, 20, 20, 20])
    for state in result['states']:
        assert set(state['inner']['fitting_rows']) == set(state['fitting'])
        assert not set(state['inner']['fitting_rows']) & set(state['validation'])
        for inner in state['inner']['states']:
            assert not set(np.asarray(state['fitting'])[inner['fitting']]) & set(state['validation'])
    with np.load(result['outer_arrays']['path'], allow_pickle=False) as f:
        ids = np.arange(n); pred = f['prediction'][f['selected'], ids]
        args = {'labels': np.arange(c)} if task == 'multiclass' else {}
        assert result['outer_gate']['quality']['f1_macro'] == pytest.approx(f1_score(y, pred, average='macro', zero_division=0, **args))
