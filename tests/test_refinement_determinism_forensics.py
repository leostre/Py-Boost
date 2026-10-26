"""Checks for artifact-only localization of refinement nondeterminism."""

from experiments.refinement_determinism_forensics_core import (
    analyze_refinement_determinism,
)


def _trace(refined, *, diagnostic=(1, 2, False), oracle=None, full=(1, 2, False), gain=2.0):
    return [{
        'iteration': 8,
        'group': 0,
        'depth': 4,
        'nodes': [{
            'node_id': 24,
            'refined_identity': list(refined),
            'diagnostic_identity': list(diagnostic),
            'oracle_refined_identity': list(oracle or refined),
            'full_best_identity': list(full),
            'refined_full_gain': gain,
            'oracle_refined_gain': gain,
            'full_best_gain': gain,
        }],
    }]


def _diagnostics(outputs):
    return [{
        'iteration': 8,
        'group': 0,
        'depth': 4,
        'node_id': 24,
        'selected_output_indices': list(outputs),
    }]


def test_localizes_output_selection_drift_before_refinement():
    report = analyze_refinement_determinism(
        _trace((3, 4, False)),
        _trace((5, 6, True)),
        _diagnostics((1, 2)),
        _diagnostics((1, 3)),
    )
    assert report['classification'] == 'output_selection_drift'
    assert report['selected_output_mismatches'] == 1


def test_localizes_numeric_or_argmax_drift_when_oracle_winner_is_stable():
    report = analyze_refinement_determinism(
        _trace((3, 4, False), oracle=(7, 8, True), gain=2.0),
        _trace((5, 6, True), oracle=(7, 8, True), gain=2.000001),
        _diagnostics((1, 2)),
        _diagnostics((1, 2)),
    )
    assert report['classification'] == 'segmented_gain_or_argmax_drift'
    assert report['first_refined_identity_mismatch']['selected_outputs_equal'] is True


def test_reports_identifiability_limit_when_oracle_shortlist_winner_changes():
    report = analyze_refinement_determinism(
        _trace((3, 4, False), oracle=(3, 4, False)),
        _trace((5, 6, True), oracle=(5, 6, True)),
        _diagnostics((1, 2)),
        _diagnostics((1, 2)),
    )
    assert report['classification'] == 'post_diagnostic_ambiguous_shortlist_or_state_drift'
    assert report['evidence_inventory']['full_k64_candidate_identities_saved'] is False
    assert report['evidence_inventory']['float64_reconstruction_possible'] is False


def test_reports_no_divergence_for_matching_traces():
    report = analyze_refinement_determinism(
        _trace((3, 4, False)),
        _trace((3, 4, False)),
        _diagnostics((1, 2)),
        _diagnostics((1, 2)),
    )
    assert report['classification'] == 'no_refined_identity_divergence'
    assert report['identity_mismatch_counts']['refined_identity'] == 0
