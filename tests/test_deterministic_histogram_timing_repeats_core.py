"""Tests for paired histogram runtime statistics."""

from experiments.deterministic_histogram_timing_repeats_core import (
    analyze_paired_runtimes,
)


def test_paired_bootstrap_is_deterministic_and_passes_clear_speedup():
    kwargs = dict(
        maximum_regression_fraction=0.10,
        bootstrap_samples=2000,
        bootstrap_seed=1729,
    )
    first = analyze_paired_runtimes([10.0] * 12, [9.0] * 12, **kwargs)
    second = analyze_paired_runtimes([10.0] * 12, [9.0] * 12, **kwargs)

    assert first == second
    assert first["timing_gate_passed"] is True
    assert first["hybrid_faster_pairs"] == 12


def test_paired_bootstrap_rejects_material_regression():
    result = analyze_paired_runtimes(
        [10.0] * 12,
        [12.0] * 12,
        maximum_regression_fraction=0.10,
        bootstrap_samples=1000,
        bootstrap_seed=1729,
    )

    assert result["timing_gate_passed"] is False
    assert result["bootstrap_median_regression_ci95"][0] > 0.10
