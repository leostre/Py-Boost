import numpy as np

from experiments.revision_forecast import (
    SubsetTrajectoryRecorder, build_forecast_rows, future_risks,
    energy_features, spectral_features, window_moments,
)


def test_future_targets_exclude_current_and_stop_at_next_control():
    trace = {t: t / 100 for t in range(1, 101)}
    assert future_risks(trace, 16, 'next_tree') == .17
    assert future_risks(trace, 16, 'next_control') == .25
    assert future_risks(trace, 25, 'next_control') == .35
    assert future_risks(trace, 96, 'next_control') is None
    trace[17] = None
    assert future_risks(trace, 16, 'next_control') is None


def test_recording_uses_fixed_training_subset_and_resets():
    recorder = SubsetTrajectoryRecorder(representation='gh', subset_rows=3, subset_seed=12)
    G = np.arange(20.).reshape(5, 4)
    H = np.ones_like(G)
    info = {'num_iter': 0, 'data': {'train': {'grad': G, 'hess': H}}}
    recorder.before_train(info)
    recorder.before_iteration(info)
    selected = recorder.row_indices.copy()
    np.testing.assert_equal(recorder.snapshots[0]['gradient'], G[selected])
    assert recorder.cheap_trace[0]['log_energy'] == energy_features((G ** 2).sum(axis=0))['log_energy']
    info['num_iter'] = 1
    recorder.before_iteration(info)
    np.testing.assert_equal(selected, recorder.row_indices)
    recorder.before_train(info)
    assert recorder.row_indices is None and not recorder.cheap_trace


def test_future_mutations_cannot_change_causal_features():
    G = np.random.default_rng(42).normal(size=(35, 10, 8))
    H = np.ones_like(G)
    cheap = [{'iteration': t + 1, **energy_features((g ** 2).sum(axis=0))} for t, g in enumerate(G)]
    spectra = spectral_features(window_moments(G, H))
    traces = {b: {t: .02 for t in range(1, 36)} for b in (25, 50, 75)}
    before = [r for r in build_forecast_rows(cheap, spectra, traces, 'test', 1, 100) if r['iteration'] == 16]
    G[16:] *= 100
    changed = [{'iteration': t + 1, **energy_features((g ** 2).sum(axis=0))} for t, g in enumerate(G)]
    changed_spectra = spectral_features(window_moments(G, H))
    traces[25][17] = .9
    after = [r for r in build_forecast_rows(changed, changed_spectra, traces, 'test', 1, 100) if r['iteration'] == 16]
    assert before and after
    for a, b in zip(before, after):
        for key in a:
            if key != 'budget_label' and not key.startswith('risk_'):
                assert a[key] == b[key]
        assert a['risk_25'] != b['risk_25']


def test_centered_constant_hessian_has_zero_contribution():
    G = np.random.default_rng(42).normal(size=(4, 10, 8))
    np.testing.assert_allclose(window_moments(G, np.ones_like(G), True),
                               window_moments(G, np.zeros_like(G), True))
