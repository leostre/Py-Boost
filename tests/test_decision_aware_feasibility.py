from __future__ import annotations

from hypothesis import given, strategies as st

from experiments.decision_aware_feasibility_core import (
    assess_sufficiency,
    assess_shortlist_recall_gate,
    audit_records,
    candidate_identity,
    evaluate_shortlist_collection,
    minimal_logging_contract,
)


def _record(shortlist_size: int, outputs: int = 5) -> dict[str, object]:
    shortlist = [
        {'feature': rank, 'split_bin': rank + 1, 'nan_left': bool(rank % 2)}
        for rank in range(shortlist_size)
    ]
    return {
        'full_best_gain': 2.0,
        'full_split_available': True,
        'diagnostic_top_k': shortlist,
        'full_top_k': shortlist,
        'candidate_output_contributions': [
            [float(rank + output) for output in range(outputs)]
            for rank in range(shortlist_size)
        ],
        'selected_output_indices': list(range(outputs)),
        'n_outputs': outputs,
    }


def test_empty_legacy_lists_do_not_fabricate_shortlist_metrics():
    record = _record(0)
    record['diagnostic_top_k'] = []
    record['full_top_k'] = []
    record['candidate_output_contributions'] = []
    record['selected_output_indices'] = []
    coverage = audit_records([record], (2, 4, 8), 1e-6)
    verdict = assess_sufficiency(coverage, (2, 4, 8), (50, 75))
    assert coverage.material_nodes == 1
    assert coverage.nonempty_diagnostic_shortlists == 0
    assert coverage.output_profile_records == 0
    assert verdict['sufficient_for_stage_a_metrics'] is False
    assert verdict['metrics_computed'] is False
    assert verdict['gate_passed'] is False


def test_leave_one_output_regret_is_not_a_candidate_profile():
    record = _record(8)
    record.pop('candidate_output_contributions')
    record['leave_one_output_regret'] = [0.1] * 5
    coverage = audit_records([record], (2, 4, 8), 1e-6)
    assert coverage.complete_shortlists[8] == 1
    assert coverage.output_profile_records == 0
    assert assess_sufficiency(coverage, (2, 4, 8), (50, 75))['gate_passed'] is False


def test_complete_contract_allows_metric_evaluation_but_does_not_prejudge_gate():
    coverage = audit_records([_record(8)], (2, 4, 8), 1e-6)
    verdict = assess_sufficiency(coverage, (2, 4, 8), (50, 75))
    assert verdict['sufficient_for_stage_a_metrics'] is True
    assert verdict['metrics_computed'] is True
    assert verdict['gate_passed'] is None


@given(st.lists(st.integers(min_value=0, max_value=8), min_size=1, max_size=30))
def test_coverage_is_invariant_to_record_order(shortlist_sizes):
    records = [_record(size) for size in shortlist_sizes]
    forward = audit_records(records, (2, 4, 8), 1e-6)
    backward = audit_records(list(reversed(records)), (2, 4, 8), 1e-6)
    assert forward == backward


@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=20),
    st.booleans(),
)
def test_candidate_identity_preserves_all_structural_coordinates(feature, split_bin, nan_left):
    candidate = {'feature': feature, 'split_bin': split_bin, 'nan_left': nan_left}
    assert candidate_identity(candidate) == (feature, split_bin, nan_left)


def test_minimal_contract_has_profiles_selections_and_audit_semantics():
    contract = minimal_logging_contract(8)
    assert contract['required_array_fields']['diagnostic_top_k'] == [8]
    assert contract['required_array_fields']['candidate_output_contributions'] == [8, 'n_outputs']
    assert 'selected_output_indices' in contract['required_array_fields']
    assert any('sum(candidate_output_contributions' in item for item in contract['semantic_checks'])
    assert any('one selected sketched winner' in item for item in contract['not_substitutes'])


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=5, max_size=5))
def test_valid_contribution_profiles_sum_to_candidate_full_gain(values):
    candidate = {
        'feature': 2,
        'split_bin': 3,
        'nan_left': False,
        'full_gain': float(sum(values)),
    }
    record = {
        'iteration': 1,
        'group': 0,
        'depth': 0,
        'node_id': 0,
        'diagnostic_output_budget': 3,
        'full_best_feature': 2,
        'full_best_split_bin': 3,
        'full_best_nan_left': False,
        'full_best_gain': float(sum(values)),
        'full_split_available': True,
        'diagnostic_top_k': [candidate],
        'full_top_k': [candidate],
        'candidate_output_contributions': [[float(value) for value in values]],
        'selected_output_indices': [0, 2, 4],
        'n_outputs': 5,
    }
    evaluation = evaluate_shortlist_collection([record], (1,), (3,), 0.0)
    assert evaluation['contract_valid'] is True
    assert evaluation['maximum_profile_sum_error'] == 0.0
    assert evaluation['per_budget']['3']['recall_at_k']['1'] == 1.0


def test_float32_reduction_uses_relative_tolerance_for_large_gains():
    record = {
        'iteration': 1,
        'group': 0,
        'depth': 0,
        'node_id': 0,
        'diagnostic_output_budget': 2,
        'full_best_feature': 1,
        'full_best_split_bin': 2,
        'full_best_nan_left': True,
        'full_best_gain': 10_000.002,
        'full_split_available': True,
        'diagnostic_top_k': [{
            'feature': 1,
            'split_bin': 2,
            'nan_left': True,
            'full_gain': 10_000.002,
        }],
        'full_top_k': [],
        'candidate_output_contributions': [[5_000.0, 5_000.0]],
        'selected_output_indices': [0, 1],
        'n_outputs': 2,
    }
    strict = evaluate_shortlist_collection(
        [record], (1,), (2,), 0.0, contribution_tolerance=1e-4,
        contribution_relative_tolerance=0.0,
    )
    float32_aware = evaluate_shortlist_collection(
        [record], (1,), (2,), 0.0, contribution_tolerance=1e-4,
        contribution_relative_tolerance=1e-6,
    )
    assert strict['contract_valid'] is False
    assert float32_aware['contract_valid'] is True


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ),
    st.permutations((8, 16, 32, 64)),
)
def test_shortlist_gate_is_order_invariant_for_monotone_recall(raw, size_order):
    monotone = []
    current = 0.0
    for value in raw:
        current = max(current, value)
        monotone.append(current)
    recalls = {str(size): value for size, value in zip((8, 16, 32, 64), monotone)}
    evaluation = {
        'contract_valid': True,
        'per_budget': {
            '50': {'recall_at_k': dict(recalls)},
            '75': {'recall_at_k': dict(recalls)},
        },
    }
    gate = assess_shortlist_recall_gate(evaluation, size_order, (75, 50), 0.95)
    expected = next(
        (size for size in (8, 16, 32, 64) if recalls[str(size)] >= 0.95),
        None,
    )
    assert gate['joint_minimum_viable_k'] == expected
    assert gate['passed'] is (expected is not None)
    assert gate['monotonicity_violations'] == []


def test_shortlist_gate_rejects_nonmonotone_recall_even_if_last_size_passes():
    evaluation = {
        'contract_valid': True,
        'per_budget': {
            '50': {'recall_at_k': {'8': 0.90, '16': 0.80, '32': 0.96}},
            '75': {'recall_at_k': {'8': 0.90, '16': 0.95, '32': 0.97}},
        },
    }
    gate = assess_shortlist_recall_gate(evaluation, (8, 16, 32), (50, 75), 0.95)
    assert gate['joint_minimum_viable_k'] == 32
    assert gate['monotonicity_violations']
    assert gate['passed'] is False
