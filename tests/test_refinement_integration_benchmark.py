"""Properties for the controlled shortlist-refinement integration gate."""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.refinement_integration_benchmark import (
    _make_inputs,
    _run_once,
    summarize_integration_cases,
)


@st.composite
def _valid_cases(draw):
    node_counts = draw(
        st.lists(
            st.integers(min_value=1, max_value=32),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    cases = []
    for budget, shortlist_size in ((50, 64), (75, 8)):
        for active_nodes in node_counts:
            full = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
            ratio = draw(st.floats(min_value=0.05, max_value=1.5, allow_nan=False))
            cases.append({
                'active_nodes': active_nodes,
                'output_budget': budget,
                'shortlist_size': shortlist_size,
                'two_stage_median_ms': full * ratio,
                'full_search_median_ms': full,
                'correctness': {'passed': True},
            })
    return cases


@given(cases=_valid_cases(), data=st.data())
def test_integration_summary_is_independent_of_case_order(cases, data):
    """Reordering node-count measurements cannot change either policy verdict."""

    permutation = data.draw(st.permutations(cases))
    policies = {50: 64, 75: 8}
    expected = summarize_integration_cases(cases, policies, 1.0)
    observed = summarize_integration_cases(permutation, policies, 1.0)
    assert observed == expected


def test_integration_gate_rejects_one_incorrect_policy():
    """A fast path cannot pass when one policy fails its correctness oracle."""

    cases = [
        {
            'active_nodes': 1,
            'output_budget': budget,
            'shortlist_size': shortlist,
            'two_stage_median_ms': 0.5,
            'full_search_median_ms': 1.0,
            'correctness': {'passed': budget != 75},
        }
        for budget, shortlist in ((50, 64), (75, 8))
    ]
    summary = summarize_integration_cases(cases, {50: 64, 75: 8}, 1.0)
    assert summary['per_policy']['50']['passed'] is True
    assert summary['per_policy']['75']['runtime_passed'] is True
    assert summary['per_policy']['75']['correctness_passed'] is False
    assert summary['passed'] is False


@pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)
def test_controlled_integration_case_matches_full_candidate_oracle():
    """The benchmark shell must exercise the exact fused shadow path end to end."""

    import cupy as cp

    protocol = {
        'n_rows': 512,
        'n_features': 8,
        'n_outputs': 6,
        'max_bin': 256,
        'seed': 20260921,
    }
    arr, grad, hess = _make_inputs(protocol)
    nodes_cpu = np.arange(512, dtype=np.int32) % 2
    nodes_cpu[::17] = -1
    entry, _, _ = _run_once(
        arr,
        grad,
        hess,
        grad[:, :4],
        hess[:, :4],
        cp.asarray(nodes_cpu),
        cp.arange(512, dtype=cp.uint64),
        cp.arange(8, dtype=cp.uint64),
        np.arange(2, dtype=np.int32),
        budget=4,
        shortlist_size=4,
        params={
            'max_bin': 256,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 5,
            'min_gain_to_split': 0.0,
        },
        full_first=False,
    )
    assert entry['shortlist_selector'] == 'exact_fused_feature_list_merge'
    assert entry['candidate_gains_allclose'] is True
    assert entry['all_refined_winners_gain_equivalent'] is True
    assert entry['two_stage_seconds'] > 0
    assert entry['full_search_seconds'] > 0
