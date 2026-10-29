"""Tests for post-warmup online spectral shadow analysis."""

import pytest

from experiments.analyze_spectral_shadow import (
    consecutive_topk_jaccard,
    probability_effective_rank,
    summarize_run,
)


def test_uniform_probability_distribution_has_full_effective_rank():
    assert probability_effective_rank((0.25, 0.25, 0.25, 0.25)) == pytest.approx(4.0)


def test_consecutive_jaccard_tracks_output_set_stability():
    trace = [
        {'top_outputs': [0, 1]},
        {'top_outputs': [1, 2]},
        {'top_outputs': [1, 2]},
    ]

    assert consecutive_topk_jaccard(trace) == pytest.approx((1 / 3 + 1) / 2)


def test_summary_excludes_pre_window_records():
    records = [
        {'iteration': 15, 'same_split': False, 'regret': 1.0, 'full_best_gain': 1.0},
        {'iteration': 16, 'same_split': True, 'regret': 0.0, 'full_best_gain': 1.0},
    ]
    trace = [
        {
            'iteration': 16,
            'spike_count': 1,
            'effective_rank': 2.0,
            'top_outputs': [0],
            'probabilities': [0.75, 0.25],
        }
    ]

    summary = summarize_run(records, trace, min_iteration=16)

    assert summary['nodes'] == 1
    assert summary['split_match_rate'] == 1.0
    assert summary['split_regret_weighted_p95'] == 0.0
