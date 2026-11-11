"""Invariant tests for pure output-subspace policy calculations."""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from py_boost.multioutput.output_subspace import (
    diversity_aware_topk,
    entropy_effective_rank,
    spectral_output_policy,
)


def test_rank_one_spectrum_has_unit_effective_rank():
    assert entropy_effective_rank((4.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_nonprefix_significant_components_are_preserved():
    summary = spectral_output_policy(
        (5.0, 3.0, 1.0), np.eye(3), (6.0, 2.0, 2.0), tau=0.0)
    assert summary.significant_component_indices == (1,)
    assert summary.loading_profiles == ((0.0,), (1.0,), (0.0,))
    assert summary.probabilities[1] > summary.probabilities[0]


def test_spectral_policy_is_normalized_and_uniform_protected():
    summary = spectral_output_policy(
        eigenvalues=(4.0, 1.0),
        eigenvectors=((1.0, 0.0), (0.0, 1.0)),
        null_p95=(2.0, 2.0),
        beta=0.8,
        tau=0.0,
    )

    assert summary.spike_count == 1
    assert summary.spike_energy_fraction == pytest.approx(0.4)
    assert summary.loading_profiles[0] == pytest.approx((2.0 ** 0.5,))
    assert summary.loading_profiles[1] == pytest.approx((0.0,))
    assert sum(summary.probabilities) == pytest.approx(1.0)
    assert min(summary.probabilities) == pytest.approx(0.1)


@given(
    spectrum=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False),
        min_size=1,
        max_size=20,
    ),
    beta=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_diagonal_spectral_policy_is_a_protected_distribution(spectrum, beta):
    dimension = len(spectrum)
    eigenvectors = tuple(
        tuple(float(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    summary = spectral_output_policy(
        eigenvalues=spectrum,
        eigenvectors=eigenvectors,
        null_p95=tuple(0.0 for _ in spectrum),
        beta=beta,
    )

    assert sum(summary.probabilities) == pytest.approx(1.0)
    assert all(value >= (1 - beta) / dimension for value in summary.probabilities)


def test_diversity_aware_topk_filters_a_duplicate_direction():
    selected = diversity_aware_topk(
        probabilities=(0.50, 0.49, 0.40),
        loading_profiles=((1.0, 0.0), (0.99, 0.01), (0.0, 1.0)),
        budget=2,
        diversity_weight=0.25,
    )

    assert selected == (0, 2)


@given(
    probabilities=st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False),
        min_size=1,
        max_size=20,
    ),
    data=st.data(),
)
def test_diversity_selection_is_unique_bounded_and_deterministic(probabilities, data):
    budget = data.draw(st.integers(min_value=1, max_value=len(probabilities)))
    profiles = tuple((value, 1.0 - value) for value in probabilities)

    first = diversity_aware_topk(probabilities, profiles, budget)
    second = diversity_aware_topk(probabilities, profiles, budget)

    assert first == second
    assert len(first) == budget
    assert len(set(first)) == budget
    assert all(0 <= index < len(probabilities) for index in first)


def test_zero_diversity_weight_matches_deterministic_topk():
    probabilities = (0.2, 0.5, 0.5, 0.1)
    selected = diversity_aware_topk(
        probabilities,
        ((1.0,), (1.0,), (1.0,), (1.0,)),
        budget=2,
        diversity_weight=0.0,
    )

    assert selected == (1, 2)
