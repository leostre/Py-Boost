"""Public callback checks: causal output energy EMA and run-state reset."""

import numpy as np
import pytest

from py_boost.gpu.accumulation import history_callback


def test_ema_energy_uses_each_round_once_including_warmup(monkeypatch):
    monkeypatch.setattr(history_callback, 'cp', np)
    callback = history_callback.EmaGradientEnergySampling(
        smoothing_alpha=.5, scheduler_mode='fixed_warmup', min_warmup_rounds=2)
    callback.before_train({})
    G1 = np.array([[1., 2.], [3., 4.]])
    G2 = np.array([[2., 1.], [4., 3.]])
    for iteration, G in enumerate([G1, G2]):
        callback.before_iteration({'num_iter': iteration,
                                   'data': {'train': {'grad': G, 'hess': np.ones_like(G)}}})
    expected = .25 * (G1 ** 2).sum(axis=0) + .5 * (G2 ** 2).sum(axis=0)
    np.testing.assert_allclose(callback.output_energy_history, expected)
    np.testing.assert_allclose(callback._decision_tensor(G2, G2).sum(axis=0), expected)
    callback.before_train({})
    assert callback.output_energy_history is None


def test_zero_alpha_matches_current_output_energy(monkeypatch):
    monkeypatch.setattr(history_callback, 'cp', np)
    callback = history_callback.EmaGradientEnergySampling(
        smoothing_alpha=0., scheduler_mode='fixed_warmup', min_warmup_rounds=1)
    G = np.array([[1., 2., 3.], [4., 5., 6.]])
    callback.before_iteration({'num_iter': 0, 'data': {'train': {'grad': G, 'hess': np.ones_like(G)}}})
    np.testing.assert_allclose(callback._decision_tensor(G, G).sum(axis=0), (G ** 2).sum(axis=0))
