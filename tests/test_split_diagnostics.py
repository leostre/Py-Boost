"""Tests for split-regret definitions used by scheduler experiments."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "multioutput"
    / "split_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("split_diagnostics", MODULE_PATH)
split_diagnostics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = split_diagnostics
SPEC.loader.exec_module(split_diagnostics)

normalized_gain_regret = split_diagnostics.normalized_gain_regret
summarize_split_risks = split_diagnostics.summarize_split_risks
weighted_quantile = split_diagnostics.weighted_quantile
action_gain_regret = split_diagnostics.action_gain_regret


def test_no_split_is_zero_gain_not_the_unused_candidate_gain():
    assert action_gain_regret(10.0, 9.0, False, True) == pytest.approx(1.0, abs=1e-6)
    assert action_gain_regret(10.0, 9.0, True, True) == pytest.approx(0.1, abs=1e-6)
    assert action_gain_regret(-1.0, -2.0, False, False) is None
    assert action_gain_regret(-1.0, -2.0, True, False) == 1.0


def test_spurious_split_count_is_not_hidden_by_gain_weights():
    summary = summarize_split_risks([
        {'regret': 0.0, 'full_best_gain': 100.0},
        {'regret': 1.0, 'full_best_gain': -1.0, 'spurious_split': True}])
    assert summary.weighted_quantile == 0.0
    assert summary.spurious_split_count == 1


def test_normalized_regret_matches_the_research_definition():
    assert normalized_gain_regret(10.0, 7.5, eps=1e-12) == pytest.approx(0.25)
    assert normalized_gain_regret(10.0, 11.0, eps=1e-12) == 0.0


def test_weighted_quantile_follows_high_gain_nodes():
    result = weighted_quantile(
        values=(0.1, 0.9),
        weights=(100.0, 1.0),
        quantile=0.95,
    )

    assert result == 0.1


def test_split_summary_keeps_distribution_extremes_and_weighted_risk():
    summary = summarize_split_risks(
        (
            {'regret': 0.1, 'full_best_gain': 100.0},
            {'regret': 0.9, 'full_best_gain': 1.0},
        )
    )

    assert summary.node_count == 2
    assert summary.maximum == 0.9
    assert summary.mean == 0.5
    assert summary.weighted_quantile == 0.1


def test_split_summary_ignores_nodes_without_a_valid_full_split():
    summary = summarize_split_risks(
        (
            {'regret': None, 'full_best_gain': 0.0},
            {'regret': 1.0, 'full_best_gain': 0.0},
        )
    )

    assert summary.node_count == 1
    assert summary.maximum == 1.0
    assert summary.mean == 1.0
    assert summary.weighted_quantile == 1.0
