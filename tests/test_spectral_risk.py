"""Tests for joining spectral state with split-risk diagnostics."""

from experiments.analyze_spectral_risk import (
    join_spectral_risk,
    safe_spearman,
    summarize_joined,
)


def test_safe_spearman_rejects_too_short_or_constant_samples():
    assert safe_spearman((1.0, 2.0), (2.0, 3.0)) is None
    assert safe_spearman((1.0, 1.0, 1.0), (1.0, 2.0, 3.0)) is None
    assert safe_spearman((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)) == -1.0


def test_join_uses_next_tree_risk_and_summary_compares_iteration_baseline():
    spectral = [
        {
            'representation': 'z',
            'window': 4,
            'end_iteration': iteration,
            'mean_principal_angle_deg': angle,
        }
        for iteration, angle in ((1, ''), (2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0))
    ]
    split = [
        {
            'iteration': iteration,
            'regret': risk,
            'full_best_gain': 1.0,
            'gradient_norm_volatility': 1 / iteration,
        }
        for iteration, risk in ((1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5), (6, 0.6))
    ]

    joined = join_spectral_risk(spectral, split)
    summary = summarize_joined(joined)[0]

    assert joined[1]['next_tree_risk'] == 0.3
    assert summary['angle_next_risk_pairs'] == 4
    assert summary['angle_next_risk_spearman'] == 1.0
