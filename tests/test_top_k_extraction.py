"""Exactness properties for alternative GPU top-K extraction strategies."""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

pytestmark = pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)


def _extract(loss, col_indexer, top_k):
    import cupy as cp
    from py_boost.gpu.utils import (
        _fused_merge_top_k_split_candidate_arrays,
        _top_k_split_candidate_arrays,
    )

    gpu_loss = cp.asarray(loss)
    gpu_columns = cp.asarray(col_indexer, dtype=cp.int64)
    baseline = _top_k_split_candidate_arrays(gpu_loss, gpu_columns, top_k)
    fused = _fused_merge_top_k_split_candidate_arrays(
        gpu_loss, gpu_columns, top_k
    )
    return (
        {name: values.get() for name, values in baseline.items()},
        {name: values.get() for name, values in fused.items()},
    )


@given(
    nnodes=st.integers(min_value=1, max_value=4),
    nfeatures=st.integers(min_value=1, max_value=7),
    nbins=st.integers(min_value=2, max_value=12),
    ndirections=st.integers(min_value=1, max_value=2),
    requested_k=st.integers(min_value=1, max_value=24),
)
@settings(max_examples=20, deadline=None)
def test_fused_merge_top_k_preserves_unique_candidate_identities(
    nnodes, nfeatures, nbins, ndirections, requested_k
):
    """The two-stage reduction returns exactly the global top-K candidates."""

    size = nnodes * nfeatures * nbins * ndirections
    rng = np.random.default_rng(
        nnodes * 100_000 + nfeatures * 1_000 + nbins * 10 + ndirections
    )
    values = rng.permutation(size).astype(np.float32).reshape(
        nnodes, nfeatures, nbins, ndirections
    )
    columns = np.arange(nfeatures, dtype=np.int64) * 3 + 7
    k = min(requested_k, nfeatures * nbins * ndirections)
    baseline, fused = _extract(values, columns, k)

    for name in ('features', 'split_bins', 'nan_left', 'diagnostic_gains'):
        np.testing.assert_array_equal(fused[name], baseline[name])


def test_fused_merge_top_k_preserves_gain_multiset_when_candidates_tie():
    """Candidate identities may differ at ties, but their exact gains may not."""

    rng = np.random.default_rng(20260921)
    values = rng.integers(0, 7, size=(4, 9, 16, 2)).astype(np.float32)
    columns = np.arange(values.shape[1], dtype=np.int64)
    baseline, fused = _extract(values, columns, top_k=23)

    np.testing.assert_array_equal(
        fused['diagnostic_gains'], baseline['diagnostic_gains']
    )


@pytest.mark.parametrize('top_k', [8, 64])
def test_fused_merge_top_k_matches_realistic_candidate_shape(top_k):
    """Check the 120 x 256 x 2 shape used by the planned benchmark."""

    rng = np.random.default_rng(20260921 + top_k)
    size = 2 * 120 * 256 * 2
    values = rng.permutation(size).astype(np.float32).reshape(2, 120, 256, 2)
    columns = np.arange(120, dtype=np.int64)
    baseline, fused = _extract(values, columns, top_k)

    for name in ('features', 'split_bins', 'nan_left', 'diagnostic_gains'):
        np.testing.assert_array_equal(fused[name], baseline[name])


def test_fused_merge_top_k_supports_genbase_feature_width():
    """Feature chunking preserves the exact global top-K beyond 1024 columns."""

    rng = np.random.default_rng(20260924)
    size = 2 * 1185 * 8 * 2
    values = rng.permutation(size).astype(np.float32).reshape(2, 1185, 8, 2)
    columns = np.arange(1185, dtype=np.int64)
    baseline, fused = _extract(values, columns, top_k=64)

    for name in ('features', 'split_bins', 'nan_left', 'diagnostic_gains'):
        np.testing.assert_array_equal(fused[name], baseline[name])
