import numpy as np

from experiments.rmt_residual_synthetic_control_core import (
    direction_reversed_risk,
    latent_complexity,
    output_spectra,
    summarize_control_gate,
    window_risk,
)
from experiments.rmt_scalar_reproducibility_core import spectral_summaries


def test_latent_complexity_has_common_trend_and_balanced_phases():
    trajectories = np.stack([
        latent_complexity(16, index, 4, 0.2, 0.01, 0.1, 8.0)
        for index in range(4)
    ])

    np.testing.assert_allclose(
        trajectories.mean(axis=0), 0.2 + 0.01 * np.arange(1, 17)
    )
    assert np.all((trajectories > 0.0) & (trajectories < 1.0))


def test_spectrum_effective_rank_increases_with_complexity():
    spectra = output_spectra([0.2, 0.8], outputs=8, concentration=18.0)
    low = spectral_summaries(spectra[0], top_k=4)
    high = spectral_summaries(spectra[1], top_k=4)

    assert high["effective_rank_fraction"] > low["effective_rank_fraction"]
    assert high["tail_energy_after_top4"] > low["tail_energy_after_top4"]


def test_window_and_reversed_risks_are_complementary():
    risk = window_risk([0.1, 0.2, 0.3, 0.4], end=4, window=2)

    assert np.isclose(risk, 0.35)
    assert np.isclose(direction_reversed_risk(risk), 0.65)


def _decision(actionable, reproducible, risk_link, rho):
    return {
        "primary_actionable": actionable,
        "features": [{
            "role": "primary",
            "reproducibility_pass": reproducible,
            "risk_link_pass": risk_link,
            "datasets": [{"median_within_trajectory_risk_rho": rho}],
        }],
    }


def test_control_gate_requires_both_sensitive_and_direction_specific_paths():
    analyses = {
        "coupled": {
            name: {"decision": _decision(True, True, True, 0.8)}
            for name in ("cross_trajectory", "local_linear")
        },
        "direction_reversed": {
            name: {"decision": _decision(False, True, False, -0.8)}
            for name in ("cross_trajectory", "local_linear")
        },
    }

    confirmed = summarize_control_gate(analyses, maximum_reversed_rho=-0.2)
    assert confirmed["scientific_status"] == "confirmed"

    analyses["direction_reversed"]["local_linear"] = {
        "decision": _decision(False, True, False, -0.1)
    }
    rejected = summarize_control_gate(analyses, maximum_reversed_rho=-0.2)
    assert rejected["scientific_status"] == "rejected"
