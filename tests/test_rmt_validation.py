"""Contracts: time/GH conservation, causal windows and actual-index geometry."""
import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.revision_rmt_validation import (basis_overlap, epoch_moments,
    normalize_parts, permutation_orders, rolling_moments, significant_indices)


@given(rows=st.integers(2, 12), outputs=st.integers(1, 6), epochs=st.integers(1, 5),
       seed=st.integers(0, 10000))
@settings(max_examples=30, deadline=None)
def test_permutation_conserves_rows_time_and_paired_hessian(rows, outputs, epochs, seed):
    orders = permutation_orders(np.random.default_rng(seed), 4, rows, outputs)
    G = np.arange(epochs * rows * outputs).reshape(epochs, rows, outputs)
    H = G * 3 + 7
    g, h = (a[:, orders, np.arange(outputs)] for a in (G, H))
    np.testing.assert_array_equal(np.sort(orders, axis=1), np.broadcast_to(np.arange(rows)[None, :, None], orders.shape))
    np.testing.assert_array_equal(h, 3 * g + 7)
    for epoch in range(1, epochs):
        np.testing.assert_array_equal(g[epoch] - g[0], np.full(g[0].shape, epoch * rows * outputs))
    # A batching change must not change a sensitivity prefix.
    prefix = permutation_orders(np.random.default_rng(seed), 2, rows, outputs)
    np.testing.assert_array_equal(orders[:2], prefix)


@given(window=st.integers(1, 8), extra=st.integers(0, 8))
@settings(max_examples=20, deadline=None)
def test_rolling_windows_are_causal_and_match_simple_reference(window, extra):
    moments = np.arange((window + extra) * 4, dtype=float).reshape(window + extra, 2, 2)
    observed = rolling_moments(moments, window)
    reference = np.stack([moments[i - window:i].mean(axis=0) for i in range(window, len(moments) + 1)])
    np.testing.assert_allclose(observed, reference)
    changed = moments.copy()
    changed[-1] += 10000
    np.testing.assert_allclose(rolling_moments(changed, window)[:-1], observed[:-1])


def test_constant_hessian_and_mean_shift_disappear_only_after_centering():
    G = np.arange(3 * 5 * 4, dtype=float).reshape(3, 5, 4)
    H = np.ones_like(G)
    centered = normalize_parts(G, H, True)
    assert np.count_nonzero(centered[1]) == 0
    shifted = normalize_parts(G + np.arange(4)[None, None, :] * 100, H, True)
    np.testing.assert_allclose(epoch_moments(centered), epoch_moments(shifted), atol=1e-6)
    assert np.count_nonzero(normalize_parts(G, H, False)[1]) == G.size


def test_significant_basis_uses_actual_nonprefix_indices():
    protocol = {'null_quantile': .95, 'relative_tolerance': 1e-6, 'absolute_tolerance': 1e-7}
    selected, _ = significant_indices(np.array([10., 5., 3.]), np.tile([11., 4., 4.], (20, 1)), protocol)
    assert selected.tolist() == [1]
    basis = np.eye(3)[:, selected]
    assert basis_overlap(basis, np.eye(3)[:, [1]]) == 1
    assert basis_overlap(basis, np.eye(3)[:, [0]]) == 0
    assert basis_overlap(basis, np.empty((3, 0))) is None


@pytest.mark.skipif(os.getenv('HASBOOST_GPU_ACCEPTANCE') != '1', reason='orx GPU acceptance only')
def test_gpu_null_matches_identity_constant_and_nested_prefix():
    import cupy as cp
    from experiments.revision_rmt_validation import null_spectra, observed_geometry
    parts = normalize_parts(cp.zeros((6, 12, 4)), cp.ones((6, 12, 4)), False, cp)
    observed, _ = observed_geometry(parts, 3)
    short = null_spectra(parts, 3, 42, 4)
    long = null_spectra(parts, 3, 42, 8, ends=[3, 6])
    np.testing.assert_allclose(short[:4, [0, 3]], long[:4], atol=1e-6)
    np.testing.assert_allclose(short, np.broadcast_to(observed, short.shape), atol=1e-6)
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(6, 12, 4)).astype(np.float32)
    parts = normalize_parts(cp.asarray(raw), cp.ones_like(cp.asarray(raw)), True, cp)
    short = null_spectra(parts, 3, 43, 4)
    long = null_spectra(parts, 3, 43, 8, ends=[3, 6])
    np.testing.assert_allclose(short[:4, [0, 3]], long[:4], atol=1e-6)
