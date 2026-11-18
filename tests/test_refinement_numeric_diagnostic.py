"""Properties for the K=64 float32 accumulation-order diagnosis."""

from __future__ import annotations

from math import fsum
import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.refinement_numeric_diagnostic import (
    _run_case,
    bounded_error_summary,
    summation_error_bound,
)
from experiments.refinement_integration_benchmark import _make_inputs


finite_float32 = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@given(st.lists(finite_float32, min_size=0, max_size=200))
@settings(max_examples=100, deadline=None)
def test_registered_bound_contains_sequential_float32_sum(values):
    """The diagnostic bound contains ordinary float32 accumulation error."""

    array = np.asarray(values, dtype=np.float32)
    observed = float(array.sum(dtype=np.float32))
    reference = float(fsum(float(value) for value in array))
    bound = summation_error_bound(array)
    assert abs(observed - reference) <= bound + np.finfo(np.float64).tiny


@given(
    st.lists(finite_float32, min_size=0, max_size=100),
    st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
@settings(max_examples=80, deadline=None)
def test_registered_bound_scales_with_input(values, scale):
    """Scaling all summands scales the absolute roundoff bound equally."""

    array = np.asarray(values, dtype=np.float32)
    scaled = array.astype(np.float64) * float(scale)
    expected = summation_error_bound(array) * abs(float(scale))
    observed = summation_error_bound(scaled, term_count=array.size)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_bounded_error_summary_rejects_an_error_above_the_bound():
    summary = bounded_error_summary(
        observed=np.asarray([1.0, 2.2]),
        reference=np.asarray([1.0, 2.0]),
        bound=np.asarray([0.0, 0.1]),
    )
    assert summary['within_bound'] is False
    assert summary['exactly_equal'] is False
    assert summary['maximum_bound_fraction'] == pytest.approx(2.0)


@pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)
def test_k64_dispatch_uses_segmented_reduction_and_preserves_contract():
    """Production K64 uses the stable kernel with exact counts and bounded sums."""

    protocol = {
        'n_rows': 512,
        'n_features': 8,
        'n_outputs': 6,
        'max_bin': 256,
        'node_counts': [2],
        'diagnostic_output_budget': 4,
        'shortlist_size': 64,
        'inactive_row_period': 17,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 5,
        'seed': 20260922,
        'roundoff_extra_terms': 256,
        'existing_gain_rtol': 0.00005,
        'existing_gain_atol': 0.005,
    }
    arr, grad, hess = _make_inputs(protocol)
    encoded = arr[:, :protocol['n_features']].get()
    case = _run_case(arr, grad, hess, encoded, 2, protocol)
    assert case['integer_contract_exact'] is True
    assert case['float_statistics_within_roundoff_bound'] is True
    assert case['winner_gains_equivalent_to_float64'] is True
    assert case['production_vs_segmented_gains']['maximum_absolute_error'] == 0.0
