"""Tests for offline spectral trajectory diagnostics."""

import numpy as np

from experiments.analyze_optimization_history import (
    effective_rank,
    output_covariance,
    principal_angles_degrees,
    rolling_window_end_positions,
    safe_output_probabilities,
)


def test_output_covariance_has_expected_shape_and_rank():
    direction = np.array([1.0, 2.0, 3.0])
    scores = np.arange(1, 9, dtype=float).reshape(2, 4, 1)
    history = scores * direction

    covariance = output_covariance(history)

    assert covariance.shape == (3, 3)
    assert np.linalg.matrix_rank(covariance, tol=1e-10) == 1
    np.testing.assert_allclose(
        effective_rank(np.linalg.eigvalsh(covariance)), 1.0, atol=1e-10
    )


def test_safe_output_policy_is_normalized_and_has_a_uniform_floor():
    eigenvalues = np.array([4.0, 1.0])
    eigenvectors = np.eye(2)
    null_p95 = np.array([2.0, 2.0])

    probabilities, spikes = safe_output_probabilities(
        eigenvalues, eigenvectors, null_p95, beta=0.8, tau=0.0
    )

    np.testing.assert_allclose(probabilities.sum(), 1.0)
    np.testing.assert_allclose(probabilities.min(), 0.1)
    np.testing.assert_array_equal(spikes, [True, False])


def test_principal_angles_detect_orthogonal_one_dimensional_subspaces():
    left = np.array([[1.0], [0.0]])
    right = np.array([[0.0], [1.0]])

    angles = principal_angles_degrees(left, right)

    np.testing.assert_allclose(angles, [90.0])


def test_strided_windows_always_keep_the_final_endpoint():
    assert rolling_window_end_positions(10, window=4, stride=4) == (3, 7, 9)
