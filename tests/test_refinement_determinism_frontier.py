"""Checks for classification of the fully captured determinism frontier."""

import numpy as np

from experiments.refinement_determinism_frontier_core import (
    analyze_determinism_frontier,
)
from py_boost.gpu.utils import _array_fingerprint, _attach_determinism_frontier


def _candidate(identity, gain):
    return {
        'identity': list(identity),
        'diagnostic_gain': gain,
        'refined_gain': gain,
        'oracle_gain': gain,
    }


def _trace(winner, *, state=None, candidates=None):
    state = state or {
        'full_grad': 'g',
        'full_hess': 'h',
        'node_indexes': 'n',
        'full_rows': 'r',
        'col_indexer': 'c',
        'diagnostic_loss': 'd',
        'full_loss': 'f',
        'candidate_identities': 'i',
    }
    candidates = candidates or [
        _candidate((1, 2, False), 2.0),
        _candidate((3, 4, True), 2.0),
    ]
    return [{
        'iteration': 8,
        'group': 0,
        'depth': 4,
        'state_fingerprints': state,
        'nodes': [{
            'node_id': 24,
            'refined_identity': list(winner),
            'candidate_frontier': candidates,
            'top1_top2_margins': {
                'diagnostic': 0.0,
                'refined': 0.0,
                'oracle': 0.0,
            },
        }],
    }]


def _diagnostics(outputs=(1, 2)):
    return [{
        'iteration': 8,
        'group': 0,
        'depth': 4,
        'node_id': 24,
        'selected_output_indices': list(outputs),
        'diagnostic_top_k': [],
        'full_top_k': [],
    }]


def test_classifies_pure_tie_breaking_when_all_captured_inputs_match():
    result = analyze_determinism_frontier(
        _trace((1, 2, False)),
        _trace((3, 4, True)),
        _diagnostics(),
        _diagnostics(),
    )
    assert result['classification'] == 'tie_breaking_drift'
    assert result['first_divergence']['ordered_shortlist_equal'] is True


def test_classifies_diagnostic_histogram_before_shortlist_difference():
    changed = {
        'full_grad': 'g',
        'full_hess': 'h',
        'node_indexes': 'n',
        'full_rows': 'r',
        'col_indexer': 'c',
        'diagnostic_loss': 'different',
        'full_loss': 'f',
        'candidate_identities': 'different',
    }
    result = analyze_determinism_frontier(
        _trace((1, 2, False)),
        _trace((3, 4, True), state=changed),
        _diagnostics(),
        _diagnostics(),
    )
    assert result['classification'] == 'diagnostic_histogram_drift'


def test_classifies_shortlist_membership_when_state_matches():
    first = [_candidate((1, 2, False), 2.0), _candidate((3, 4, True), 1.0)]
    second = [_candidate((1, 2, False), 2.0), _candidate((5, 6, True), 1.0)]
    result = analyze_determinism_frontier(
        _trace((1, 2, False), candidates=first),
        _trace((5, 6, True), candidates=second),
        _diagnostics(),
        _diagnostics(),
    )
    assert result['classification'] == 'shortlist_membership_drift'


def test_reports_registered_horizon_when_no_winner_diverges():
    result = analyze_determinism_frontier(
        _trace((1, 2, False)),
        _trace((1, 2, False)),
        _diagnostics(),
        _diagnostics(),
    )
    assert result['classification'] == 'no_refined_identity_divergence'
    assert result['registered_horizon_trees'] == 30


def test_capture_contains_ordered_candidates_margins_and_stable_fingerprint():
    nodes = [{}]
    cpu = {
        'features': np.asarray([[1, 3]], dtype=np.int64),
        'split_bins': np.asarray([[2, 4]], dtype=np.int64),
        'nan_left': np.asarray([[False, True]], dtype=np.bool_),
        'diagnostic_gains': np.asarray([[2.0, 1.5]], dtype=np.float32),
        'refined_gains': np.asarray([[2.25, 2.0]], dtype=np.float32),
        'oracle_candidate_gains': np.asarray([[2.5, 2.0]], dtype=np.float32),
    }
    _attach_determinism_frontier(nodes, cpu)
    assert [candidate['rank'] for candidate in nodes[0]['candidate_frontier']] == [0, 1]
    assert nodes[0]['top1_top2_margins'] == {
        'diagnostic': 0.5,
        'refined': 0.25,
        'oracle': 0.5,
    }
    assert _array_fingerprint(cpu['features']) == _array_fingerprint(
        cpu['features'].copy()
    )
