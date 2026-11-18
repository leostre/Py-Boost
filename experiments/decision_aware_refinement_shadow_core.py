"""Pure aggregation and gates for GPU shortlist-refinement shadow traces."""

from __future__ import annotations

from math import fsum
from typing import Mapping, Sequence

from py_boost.multioutput.split_diagnostics import weighted_quantile


def _risk_summary(nodes: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    regrets = [float(node[field]) for node in nodes]
    weights = [max(0.0, float(node['full_best_gain'])) for node in nodes]
    if not regrets:
        return {'nodes': 0, 'mean_regret': None, 'weighted_p95_regret': None}
    return {
        'nodes': len(nodes),
        'mean_regret': fsum(regrets) / len(regrets),
        'weighted_p95_regret': weighted_quantile(regrets, weights, 0.95),
    }


def _relative_reduction(baseline: float, refined: float) -> float:
    if baseline == 0:
        return 0.0 if refined == 0 else float('-inf')
    return (baseline - refined) / baseline


def _timing_summary(batches: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Sum synchronized timing components for one comparable batch group."""

    two_stage_seconds = fsum(float(entry['two_stage_seconds']) for entry in batches)
    full_search_seconds = fsum(float(entry['full_search_seconds']) for entry in batches)
    return {
        'initial_statistics': fsum(
            float(entry['initial_statistics_seconds']) for entry in batches
        ),
        'shortlist_selection': fsum(
            float(entry['shortlist_selection_seconds']) for entry in batches
        ),
        'refinement': fsum(float(entry['refinement_seconds']) for entry in batches),
        'two_stage': two_stage_seconds,
        'full_statistics': fsum(
            float(entry['full_statistics_seconds']) for entry in batches
        ),
        'full_selection': fsum(
            float(entry['full_selection_seconds']) for entry in batches
        ),
        'full_search': full_search_seconds,
        'runtime_ratio_to_full_search': two_stage_seconds / full_search_seconds,
    }


def evaluate_refinement_shadow(
    trace: Sequence[Mapping[str, object]],
    budgets: Sequence[int],
    material_gain_epsilon: float,
) -> dict[str, object]:
    """Aggregate correctness, regret, and synchronized GPU timings by budget."""

    if material_gain_epsilon < 0:
        raise ValueError('material_gain_epsilon must be non-negative')
    per_budget = {}
    for budget in sorted({int(value) for value in budgets}):
        batches = [
            entry for entry in trace
            if int(entry['diagnostic_output_budget']) == budget
        ]
        material_nodes = [
            node
            for entry in batches
            for node in entry['nodes']
            if bool(node['full_split_available'])
            and float(node['full_best_gain']) > material_gain_epsilon
        ]
        baseline = _risk_summary(material_nodes, 'baseline_regret')
        refined = _risk_summary(material_nodes, 'refined_regret')
        timings = _timing_summary(batches) if batches else None
        full_search_seconds = 0.0 if timings is None else timings['full_search']
        if not batches or not material_nodes or full_search_seconds <= 0:
            raise ValueError(f'incomplete refinement trace for budget {budget}')
        timings_by_depth = {
            str(depth): _timing_summary([
                entry for entry in batches if int(entry['depth']) == depth
            ])
            for depth in sorted({int(entry['depth']) for entry in batches})
        }
        per_budget[str(budget)] = {
            'batches': len(batches),
            'shortlist_sizes': sorted({int(entry['shortlist_size']) for entry in batches}),
            'baseline': baseline,
            'refined': refined,
            'mean_regret_reduction_fraction': _relative_reduction(
                float(baseline['mean_regret']), float(refined['mean_regret'])
            ),
            'weighted_p95_regret_reduction_fraction': _relative_reduction(
                float(baseline['weighted_p95_regret']),
                float(refined['weighted_p95_regret']),
            ),
            'timings_seconds': timings,
            'timings_by_depth_seconds': timings_by_depth,
            'root_runtime_ratio_to_full_search': (
                timings_by_depth.get('0', {}).get('runtime_ratio_to_full_search')
            ),
            'runtime_ratio_to_full_search': timings['runtime_ratio_to_full_search'],
            'maximum_candidate_gain_absolute_error': max(
                float(entry['maximum_candidate_gain_absolute_error']) for entry in batches
            ),
            'maximum_candidate_gain_relative_error': max(
                float(entry['maximum_candidate_gain_relative_error']) for entry in batches
            ),
            'candidate_gains_allclose': all(
                bool(entry['candidate_gains_allclose']) for entry in batches
            ),
            'all_refined_winners_gain_equivalent': all(
                bool(entry['all_refined_winners_gain_equivalent']) for entry in batches
            ),
        }
    return {'per_budget': per_budget}


def assess_refinement_shadow_gate(
    evaluation: Mapping[str, object],
    budgets: Sequence[int],
    minimum_p95_reduction: float,
) -> dict[str, object]:
    """Require gain equivalence, regret reduction, and measured speedup."""

    if not 0 <= minimum_p95_reduction <= 1:
        raise ValueError('minimum_p95_reduction must lie in [0, 1]')
    verdicts = {}
    for budget in sorted({int(value) for value in budgets}):
        metrics = evaluation['per_budget'][str(budget)]
        correctness = bool(
            metrics['candidate_gains_allclose']
            and metrics['all_refined_winners_gain_equivalent']
        )
        regret = (
            float(metrics['weighted_p95_regret_reduction_fraction'])
            >= minimum_p95_reduction
        )
        runtime = float(metrics['runtime_ratio_to_full_search']) < 1.0
        verdicts[str(budget)] = {
            'correctness_passed': correctness,
            'p95_reduction_passed': regret,
            'runtime_passed': runtime,
            'passed': correctness and regret and runtime,
        }
    return {
        'minimum_weighted_p95_regret_reduction': minimum_p95_reduction,
        'per_budget': verdicts,
        'passed': bool(verdicts) and all(item['passed'] for item in verdicts.values()),
    }
