import numpy as np

from experiments.rmt_scalar_reproducibility import _window_eigenvalues
from experiments.rmt_scalar_reproducibility_core import (
    FEATURE_NAMES,
    relative_pair_errors,
    spectral_summaries,
    summarize_scalar_verdict,
)


def test_spectral_summaries_have_expected_limits():
    concentrated = spectral_summaries([10, 0, 0, 0, 0], top_k=4)
    separated = spectral_summaries([10, 9, 8, 7, 0], top_k=4)
    uniform = spectral_summaries([1, 1, 1, 1, 1], top_k=4)

    assert concentrated["effective_rank_fraction"] == 0.2
    assert concentrated["tail_energy_after_top4"] == 0.0
    assert separated["inverse_relative_gap4"] == 0.0
    assert np.isclose(uniform["effective_rank_fraction"], 1.0)
    assert np.isclose(uniform["tail_energy_after_top4"], 0.2)
    assert uniform["inverse_relative_gap4"] == 1.0


def test_relative_pair_errors_are_symmetric():
    left = np.array([1.0, 2.0, 4.0])
    right = np.array([1.0, 3.0, 2.0])

    first = relative_pair_errors(left, right)
    second = relative_pair_errors(right, left)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, [0.0, 0.4, 2.0 / 3.0])


def test_window_eigenvalues_support_broadcast_hessian_and_batches():
    rng = np.random.default_rng(31)
    gradient = rng.normal(size=(5, 8, 6)).astype(np.float32)
    hessian = np.ones((5, 8, 1), dtype=np.float32)
    groups = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

    spectra = _window_eigenvalues(gradient, hessian, 5, 4, groups, np, 1)

    assert spectra.shape == (2, 6)
    assert np.all(spectra[:, :-1] >= spectra[:, 1:])
    assert np.all(spectra >= 0.0)


def _rows(primary_risk_reversed=False, secondary_actionable=False, unstable=False):
    rows = []
    for dataset in ("helena", "mediamill"):
        for model_seed in (204, 214, 224):
            for index, end in enumerate((20, 24, 28, 32, 36)):
                risk = float(index)
                for feature in FEATURE_NAMES:
                    feature_reversed = (
                        feature == "effective_rank_fraction" and primary_risk_reversed
                    )
                    if feature != "effective_rank_fraction" and not secondary_actionable:
                        feature_reversed = True
                    value = float(4 - index if feature_reversed else index)
                    left = np.asarray([value + repeat * 1e-4 for repeat in range(5)])
                    right = left[::-1] if unstable else left + 1e-5
                    rows.append({
                        "dataset": dataset,
                        "model_seed": model_seed,
                        "end": end,
                        "feature": feature,
                        "partition_a_values": left.tolist(),
                        "partition_b_values": right.tolist(),
                        "median_relative_error": 0.5 if unstable else 0.001,
                        "full_value": value,
                        "current_risk": risk,
                    })
    return rows


def _decision(rows):
    return summarize_scalar_verdict(
        rows,
        primary_feature="effective_rank_fraction",
        secondary_features=("tail_energy_after_top4", "inverse_relative_gap4"),
        maximum_relative_error=0.10,
        minimum_stable_fraction=0.80,
        minimum_split_rank_rho=0.80,
        minimum_risk_rho=0.20,
    )


def test_verdict_confirms_only_the_preregistered_primary_feature():
    assert _decision(_rows())["verdict"] == "confirmed_actionable"


def test_verdict_marks_secondary_without_rescuing_primary():
    result = _decision(_rows(primary_risk_reversed=True, secondary_actionable=True))

    assert result["verdict"] == "secondary_candidate"
    assert result["primary_actionable"] is False
    assert result["secondary_actionable"] == [
        "tail_energy_after_top4", "inverse_relative_gap4"
    ]


def test_verdict_separates_descriptive_and_rejected_states():
    descriptive = _decision(_rows(primary_risk_reversed=True))
    rejected = _decision(_rows(primary_risk_reversed=True, unstable=True))

    assert descriptive["verdict"] == "descriptive_only"
    assert rejected["verdict"] == "rejected"
