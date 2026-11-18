"""Pure statistical gates for paired histogram timing repeats."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np


def analyze_paired_runtimes(
    perf1_seconds: Sequence[float],
    hybrid_seconds: Sequence[float],
    *,
    maximum_regression_fraction: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if len(perf1_seconds) != len(hybrid_seconds) or not perf1_seconds:
        raise ValueError("paired timing vectors must have the same positive length")
    if any(value <= 0 for value in (*perf1_seconds, *hybrid_seconds)):
        raise ValueError("timings must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    perf1 = np.asarray(perf1_seconds, dtype=np.float64)
    hybrid = np.asarray(hybrid_seconds, dtype=np.float64)
    paired_regressions = hybrid / perf1 - 1.0
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        paired_regressions.size,
        size=(bootstrap_samples, paired_regressions.size),
    )
    bootstrap_medians = np.median(paired_regressions[indices], axis=1)
    lower, upper = np.quantile(bootstrap_medians, [0.025, 0.975])
    median_regression = float(np.median(paired_regressions))
    return {
        "pair_count": int(paired_regressions.size),
        "perf1_median_seconds": float(statistics.median(perf1_seconds)),
        "hybrid_median_seconds": float(statistics.median(hybrid_seconds)),
        "paired_regressions": paired_regressions.tolist(),
        "median_paired_regression_fraction": median_regression,
        "bootstrap_median_regression_ci95": [float(lower), float(upper)],
        "hybrid_faster_pairs": int(np.count_nonzero(paired_regressions < 0)),
        "hybrid_slower_pairs": int(np.count_nonzero(paired_regressions > 0)),
        "ties": int(np.count_nonzero(paired_regressions == 0)),
        "maximum_regression_fraction": float(maximum_regression_fraction),
        "timing_gate_passed": bool(upper <= maximum_regression_fraction),
    }
