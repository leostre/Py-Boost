"""Pure generator and acceptance helpers for the preregistered H5-3C control."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def latent_complexity(
    steps: int,
    trajectory_index: int,
    trajectories: int,
    intercept: float,
    slope: float,
    amplitude: float,
    period: float,
) -> np.ndarray:
    """Return the preregistered trend plus trajectory-specific oscillation."""

    if steps < 1 or trajectories < 2:
        raise ValueError("steps must be positive and at least two trajectories are required")
    if not 0 <= trajectory_index < trajectories:
        raise ValueError("trajectory index lies outside the configured ensemble")
    if period <= 0.0:
        raise ValueError("period must be positive")
    iteration = np.arange(1, steps + 1, dtype=float)
    phase = 2.0 * np.pi * trajectory_index / trajectories
    values = (
        intercept
        + slope * iteration
        + amplitude * np.sin(2.0 * np.pi * iteration / period + phase)
    )
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("latent complexity must remain strictly between zero and one")
    return values


def output_spectra(
    complexity: Sequence[float], outputs: int, concentration: float
) -> np.ndarray:
    """Map complexity to one dominant and ``outputs - 1`` unit eigenvalues."""

    complexity = np.asarray(complexity, dtype=float)
    if complexity.ndim != 1 or not len(complexity):
        raise ValueError("complexity must be a non-empty vector")
    if outputs < 2:
        raise ValueError("at least two outputs are required")
    if concentration <= 0.0:
        raise ValueError("concentration must be positive")
    spectra = np.ones((len(complexity), outputs), dtype=float)
    spectra[:, 0] = 1.0 + concentration * (1.0 - complexity)
    return spectra


def window_risk(complexity: Sequence[float], end: int, window: int) -> float:
    """Use the same causal window as the spectral feature."""

    complexity = np.asarray(complexity, dtype=float)
    if not 0 < window <= end <= len(complexity):
        raise ValueError("risk window lies outside the latent trajectory")
    return float(np.mean(complexity[end - window:end]))


def direction_reversed_risk(risk: float) -> float:
    """Return the preregistered direction-reversed control."""

    if not np.isfinite(risk) or not 0.0 < risk < 1.0:
        raise ValueError("risk must be finite and lie strictly between zero and one")
    return float(1.0 - risk)


def _primary_dataset(decision: Mapping[str, object]) -> Mapping[str, object]:
    primary = [
        feature for feature in decision["features"]
        if feature["role"] == "primary"
    ]
    if len(primary) != 1 or len(primary[0]["datasets"]) != 1:
        raise ValueError("synthetic control expects one primary and one dataset")
    return primary[0]["datasets"][0]


def summarize_control_gate(
    analyses: Mapping[str, Mapping[str, Mapping[str, object]]],
    maximum_reversed_rho: float,
) -> dict[str, object]:
    """Require sensitivity to the coupled arm and rejection of reversed direction."""

    checks = []
    for transform in ("cross_trajectory", "local_linear"):
        coupled = analyses["coupled"][transform]["decision"]
        reversed_decision = analyses["direction_reversed"][transform]["decision"]
        reversed_dataset = _primary_dataset(reversed_decision)
        checks.append({
            "transform": transform,
            "coupled_primary_actionable": bool(coupled["primary_actionable"]),
            "reversed_primary_reproducible": bool(
                next(feature for feature in reversed_decision["features"]
                     if feature["role"] == "primary")["reproducibility_pass"]
            ),
            "reversed_primary_risk_link_rejected": not bool(
                next(feature for feature in reversed_decision["features"]
                     if feature["role"] == "primary")["risk_link_pass"]
            ),
            "reversed_median_risk_rho": reversed_dataset[
                "median_within_trajectory_risk_rho"
            ],
        })
    for check in checks:
        rho = check["reversed_median_risk_rho"]
        check["reversed_direction_pass"] = (
            rho is not None and float(rho) <= maximum_reversed_rho
        )
        check["passed"] = bool(
            check["coupled_primary_actionable"]
            and check["reversed_primary_reproducible"]
            and check["reversed_primary_risk_link_rejected"]
            and check["reversed_direction_pass"]
        )
    passed = all(check["passed"] for check in checks)
    return {
        "scientific_status": "confirmed" if passed else "rejected",
        "passed": passed,
        "maximum_reversed_rho": maximum_reversed_rho,
        "checks": checks,
    }
