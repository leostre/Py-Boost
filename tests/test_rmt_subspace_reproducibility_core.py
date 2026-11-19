import numpy as np

from experiments.rmt_subspace_reproducibility import _window_bases
from experiments.rmt_subspace_reproducibility_core import (
    balanced_partitions,
    bootstrap_median_interval,
    paired_distance_samples,
    spearman_correlation,
    subspace_distance,
    summarize_verdict,
)


def test_subspace_distance_has_projector_endpoints():
    first = np.eye(3)[:, :1]
    same_projector = -first
    orthogonal = np.eye(3)[:, 1:2]

    assert subspace_distance(first, same_projector) == 0.0
    assert subspace_distance(first, orthogonal) == 1.0


def test_balanced_partitions_are_deterministic_and_exhaustive():
    first = balanced_partitions(12, 5, 17)
    second = balanced_partitions(12, 5, 17)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 2, 6)
    for partition in first:
        np.testing.assert_array_equal(np.sort(partition.ravel()), np.arange(12))


def test_paired_samples_separate_temporal_rotation_from_estimator_noise():
    previous = np.tile(np.eye(2)[:, :1], (8, 1, 1))
    current = np.tile(np.eye(2)[:, 1:2], (8, 1, 1))

    rotated = paired_distance_samples(current, current, previous, previous)
    stable = paired_distance_samples(previous, previous, previous, previous)

    np.testing.assert_allclose(rotated["noise"], 0.0)
    np.testing.assert_allclose(rotated["temporal"], 1.0)
    np.testing.assert_allclose(rotated["delta"], 1.0)
    np.testing.assert_allclose(stable["delta"], 0.0)


def test_bootstrap_interval_is_reproducible():
    first = bootstrap_median_interval([0.1, 0.2, 0.3, 0.4], 1000, 11)
    second = bootstrap_median_interval([0.1, 0.2, 0.3, 0.4], 1000, 11)

    assert first == second
    assert first["low"] <= first["median"] <= first["high"]


def test_window_bases_support_broadcast_hessian_and_batches():
    rng = np.random.default_rng(23)
    gradient = rng.normal(size=(5, 8, 3)).astype(np.float32)
    hessian = np.ones((5, 8, 1), dtype=np.float32)
    groups = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

    bases = _window_bases(gradient, hessian, 5, 4, groups, 2, np, 1)

    assert bases.shape == (2, 3, 2)
    for basis in bases:
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)


def _decision_rows(risk_reversed=False, unstable=False):
    rows = []
    for dataset in ("helena", "mediamill"):
        for model_seed in (204, 214, 224):
            for index, end in enumerate((20, 24, 28, 32)):
                risk = float(3 - index if risk_reversed else index)
                rows.append({
                    "dataset": dataset,
                    "model_seed": model_seed,
                    "end": end,
                    "rank": 4,
                    "median_overlap": 0.5 if unstable else 0.9,
                    "delta_ci_low": 0.1,
                    "median_delta": float(index),
                    "current_risk": risk,
                })
    return rows


def _verdict(rows):
    return summarize_verdict(
        rows,
        primary_rank=4,
        stable_overlap=0.75,
        minimum_stable_fraction=0.5,
        minimum_identifiable_fraction=0.2,
        minimum_risk_rho=0.2,
    )


def test_verdict_distinguishes_actionable_geometric_and_rejected_states():
    assert _verdict(_decision_rows())["verdict"] == "confirmed_actionable"
    assert _verdict(_decision_rows(risk_reversed=True))["verdict"] == "geometric_only"
    assert _verdict(_decision_rows(unstable=True))["verdict"] == "rejected"


def test_spearman_handles_ties_and_constant_vectors():
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == 1.0
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) is None
