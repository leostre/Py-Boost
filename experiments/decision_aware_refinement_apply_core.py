"""Pure summaries and acceptance checks for applying refined split candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def summarize_application_trace(trace: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize which audited refinement batches affected tree structure."""

    nodes = [node for entry in trace for node in entry['nodes']]
    return {
        'batches': len(trace),
        'nodes': len(nodes),
        'requested_batches': sum(bool(entry['application_requested']) for entry in trace),
        'applied_batches': sum(bool(entry['applied_to_structure']) for entry in trace),
        'refused_batches': sum(
            entry.get('application_refusal_reason') is not None for entry in trace
        ),
        'applied_change_count': sum(
            int(entry.get('applied_change_count', 0)) for entry in trace
        ),
        'budgets': sorted({int(entry['diagnostic_output_budget']) for entry in trace}),
        'shortlist_sizes': sorted({int(entry['shortlist_size']) for entry in trace}),
        'all_candidate_gains_close': all(
            bool(entry['candidate_gains_allclose']) for entry in trace
        ),
        'all_winners_gain_equivalent': all(
            bool(entry['all_refined_winners_gain_equivalent']) for entry in trace
        ),
    }


def assess_application_gate(
    records: Mapping[str, Mapping[str, object]],
    traces: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    expected_trees: int,
    expected_budget: int,
    expected_shortlist: int,
    maximum_log_loss_increase: float,
    maximum_f1_weighted_decrease: float,
    maximum_runtime_ratio: float,
) -> dict[str, object]:
    """Evaluate the preregistered H-APPLY-1 correctness and non-inferiority gate."""

    baseline = records['baseline']
    treatment = records['treatment']
    repeat = records['repeat']
    summaries = {name: summarize_application_trace(trace) for name, trace in traces.items()}
    runtime_ratio = float(treatment['train_seconds']) / float(baseline['train_seconds'])
    log_loss_delta = float(treatment['log_loss']) - float(baseline['log_loss'])
    f1_delta = float(treatment['f1_weighted']) - float(baseline['f1_weighted'])
    checks = {
        'all_arms_completed': all(
            int(record['trees_built']) == expected_trees for record in records.values()
        ),
        'baseline_remained_shadow_only': (
            summaries['baseline']['applied_batches'] == 0
            and summaries['baseline']['requested_batches'] == 0
        ),
        'treatment_applied_every_requested_batch': (
            summaries['treatment']['requested_batches'] > 0
            and summaries['treatment']['requested_batches']
            == summaries['treatment']['applied_batches']
            and summaries['treatment']['refused_batches'] == 0
        ),
        'repeat_applied_every_requested_batch': (
            summaries['repeat']['requested_batches'] > 0
            and summaries['repeat']['requested_batches']
            == summaries['repeat']['applied_batches']
            and summaries['repeat']['refused_batches'] == 0
        ),
        'registered_policy_only': all(
            summary['budgets'] == [expected_budget]
            and summary['shortlist_sizes'] == [expected_shortlist]
            for summary in summaries.values()
        ),
        'numeric_audit_passed': all(
            summary['all_candidate_gains_close']
            and summary['all_winners_gain_equivalent']
            for summary in summaries.values()
        ),
        'at_least_one_structure_action_changed': (
            summaries['treatment']['applied_change_count'] > 0
        ),
        'baseline_and_treatment_structures_differ': (
            baseline['model_structure_fingerprint']
            != treatment['model_structure_fingerprint']
        ),
        'treatment_repeat_structure_is_exact': (
            treatment['model_structure_fingerprint']
            == repeat['model_structure_fingerprint']
        ),
        'treatment_repeat_predictions_are_exact': (
            treatment['prediction_fingerprint'] == repeat['prediction_fingerprint']
        ),
        'treatment_repeat_metrics_are_exact': all(
            float(treatment[metric]) == float(repeat[metric])
            for metric in ('log_loss', 'f1_weighted', 'f1_macro', 'f1_micro')
        ),
        'log_loss_non_inferior': log_loss_delta <= maximum_log_loss_increase,
        'f1_weighted_non_inferior': f1_delta >= -maximum_f1_weighted_decrease,
        'runtime_within_limit': runtime_ratio <= maximum_runtime_ratio,
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'trace_summaries': summaries,
        'log_loss_delta': log_loss_delta,
        'f1_weighted_delta': f1_delta,
        'treatment_to_baseline_runtime_ratio': runtime_ratio,
    }
