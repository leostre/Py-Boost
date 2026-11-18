"""Pure checks for H-HIST-PROFILE-1 aggregation and routing."""

from experiments.deterministic_histogram_profile_core import (
    analyze_histogram_profile,
)


def _record(kind, nodes, *, canonical=4.0, kernel=2.0, post=1.0):
    return {
        'kind': kind,
        'nodes': nodes,
        'allocation_ms': 0.2,
        'row_filter_ms': 0.1,
        'row_to_host_ms': canonical * 0.5,
        'cpu_sort_ms': canonical * 0.3,
        'row_to_device_ms': canonical * 0.2,
        'kernel_ms': kernel,
        'cumsum_ms': post * 0.8,
        'sibling_subtraction_ms': post * 0.2,
        'total_ms': canonical + kernel + post + 0.5,
    }


def test_profile_routes_a_dominant_canonicalization_stage():
    records = [
        _record(kind, nodes)
        for kind in ('diagnostic', 'structure')
        for nodes in (1, 2, 4, 8, 16)
    ]
    result = analyze_histogram_profile(records, train_seconds=1.0)
    assert result['coverage_gate_passed'] is True
    assert result['dominant_component'] == 'canonicalization'
    assert result['next_action'] == 'optimize_canonicalization'
    assert len(result['by_path_and_nodes']) == 10


def test_profile_requires_both_paths_at_every_depth():
    records = [
        _record(kind, nodes, canonical=2.0, kernel=2.0, post=2.0)
        for kind in ('diagnostic', 'structure')
        for nodes in (1, 2, 4, 8, 16)
        if not (kind == 'structure' and nodes == 16)
    ]
    result = analyze_histogram_profile(records, train_seconds=1.0)
    assert result['coverage_gate_passed'] is False
    assert result['dominant_component'] is None
    assert result['next_action'] == 'controlled_microbenchmark'


def test_profile_rejects_negative_timing_as_invalid_coverage():
    records = [
        _record(kind, nodes)
        for kind in ('diagnostic', 'structure')
        for nodes in (1, 2, 4, 8, 16)
    ]
    records[0]['kernel_ms'] = -1.0
    result = analyze_histogram_profile(records, train_seconds=1.0)
    assert result['coverage_gate_passed'] is False
    assert result['invalid_records'][0]['invalid_values'] == ['kernel_ms']
