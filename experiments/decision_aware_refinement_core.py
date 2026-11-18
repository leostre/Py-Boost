"""Pure evaluation of exact full-statistic refinement inside a shortlist."""

from __future__ import annotations

from math import fsum, isfinite
from typing import Mapping, Sequence

from py_boost.multioutput.split_diagnostics import (
    action_gain_regret,
    weighted_quantile,
)


def refinement_work_ratio(
    output_budget: int,
    output_count: int,
    shortlist_size: int,
    candidate_count: int,
) -> float:
    """Estimate two-stage gain work relative to an all-output exhaustive search."""

    if output_count <= 0 or candidate_count <= 0 or shortlist_size <= 0:
        raise ValueError('output, candidate, and shortlist counts must be positive')
    if not 0 < output_budget <= output_count:
        raise ValueError('output budget must lie in [1, output_count]')
    if shortlist_size > candidate_count:
        raise ValueError('shortlist size cannot exceed the candidate count')
    initial = output_budget * candidate_count
    completion = (output_count - output_budget) * shortlist_size
    return (initial + completion) / (output_count * candidate_count)


def exact_shortlist_action(
    record: Mapping[str, object],
    shortlist_size: int,
) -> dict[str, object]:
    """Rerank the shortlist by the sum of all per-output gain contributions."""

    if shortlist_size <= 0:
        raise ValueError('shortlist_size must be positive')
    candidates = record.get('diagnostic_top_k')
    profiles = record.get('candidate_output_contributions')
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError('diagnostic shortlist is missing')
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise ValueError('candidate output profiles are missing')
    if len(candidates) < shortlist_size or len(profiles) < shortlist_size:
        raise ValueError('requested shortlist is not fully recorded')

    output_count = int(record['n_outputs'])
    scored: list[tuple[float, int]] = []
    for index, profile in enumerate(profiles[:shortlist_size]):
        if not isinstance(profile, Sequence) or isinstance(profile, (str, bytes)):
            raise ValueError('candidate output profile must be a sequence')
        if len(profile) != output_count:
            raise ValueError('candidate output profile has the wrong output count')
        gain = fsum(float(value) for value in profile)
        if not isfinite(gain):
            raise ValueError('candidate full gain must be finite')
        scored.append((gain, index))

    best_gain, best_index = max(scored, key=lambda pair: (pair[0], -pair[1]))
    min_gain = float(record.get('min_gain_to_split') or 0.0)
    split_available = best_gain > min_gain
    regret = action_gain_regret(
        float(record['full_best_gain']),
        best_gain,
        split_available,
        bool(record['full_split_available']),
    )
    candidate = candidates[best_index]
    recorded_gain = float(candidate['full_gain'])
    return {
        'candidate_index': best_index,
        'candidate_identity': [
            int(candidate['feature']),
            int(candidate['split_bin']),
            bool(candidate['nan_left']),
        ],
        'full_gain_from_profile': best_gain,
        'recorded_full_gain': recorded_gain,
        'profile_sum_error': abs(best_gain - recorded_gain),
        'split_available': split_available,
        'regret': regret,
    }


def _risk_summary(records: Sequence[Mapping[str, float]]) -> dict[str, object]:
    if not records:
        return {'nodes': 0, 'mean_regret': None, 'weighted_p95_regret': None}
    regrets = [float(record['regret']) for record in records]
    weights = [max(0.0, float(record['full_best_gain'])) for record in records]
    return {
        'nodes': len(records),
        'mean_regret': fsum(regrets) / len(regrets),
        'weighted_p95_regret': weighted_quantile(regrets, weights, 0.95),
    }


def _relative_reduction(baseline: float, refined: float) -> float:
    if baseline < 0 or refined < 0:
        raise ValueError('regret must be non-negative')
    if baseline == 0:
        return 0.0 if refined == 0 else float('-inf')
    return (baseline - refined) / baseline


def evaluate_exact_refinement(
    records: Sequence[Mapping[str, object]],
    policies: Mapping[str, Mapping[int, int]],
    budgets: Sequence[int],
    candidate_count: int,
    material_gain_epsilon: float,
) -> dict[str, object]:
    """Compare exact shortlist refinement with the recorded diagnostic action."""

    if material_gain_epsilon < 0:
        raise ValueError('material_gain_epsilon must be non-negative')
    requested_budgets = tuple(sorted({int(value) for value in budgets}))
    if not requested_budgets or requested_budgets[0] <= 0:
        raise ValueError('budgets must be positive')

    result: dict[str, object] = {'per_policy': {}, 'maximum_profile_sum_error': 0.0}
    maximum_error = 0.0
    for policy_name, sizes in policies.items():
        budget_results: dict[str, object] = {}
        for budget in requested_budgets:
            shortlist_size = int(sizes[budget])
            material = [
                record for record in records
                if int(record.get('diagnostic_output_budget') or 0) == budget
                and bool(record.get('full_split_available'))
                and float(record.get('full_best_gain') or 0.0) > material_gain_epsilon
            ]
            baseline_rows = [
                {
                    'regret': float(record['regret']),
                    'full_best_gain': float(record['full_best_gain']),
                }
                for record in material
            ]
            refined_rows = []
            output_counts = {int(record['n_outputs']) for record in material}
            if len(output_counts) != 1:
                raise ValueError('every budget must have one output count')
            for record in material:
                action = exact_shortlist_action(record, shortlist_size)
                if action['regret'] is None:
                    raise ValueError('material node produced no refinement regret')
                maximum_error = max(maximum_error, float(action['profile_sum_error']))
                refined_rows.append({
                    'regret': float(action['regret']),
                    'full_best_gain': float(record['full_best_gain']),
                })

            baseline = _risk_summary(baseline_rows)
            refined = _risk_summary(refined_rows)
            baseline_p95 = float(baseline['weighted_p95_regret'])
            refined_p95 = float(refined['weighted_p95_regret'])
            output_count = next(iter(output_counts))
            cost_ratio = refinement_work_ratio(
                budget, output_count, shortlist_size, candidate_count,
            )
            budget_results[str(budget)] = {
                'shortlist_size': shortlist_size,
                'baseline': baseline,
                'refined': refined,
                'mean_regret_reduction_fraction': _relative_reduction(
                    float(baseline['mean_regret']), float(refined['mean_regret'])
                ),
                'weighted_p95_regret_reduction_fraction': _relative_reduction(
                    baseline_p95, refined_p95
                ),
                'estimated_work_ratio_to_full_search': cost_ratio,
                'candidate_count': candidate_count,
                'output_count': output_count,
            }
        result['per_policy'][policy_name] = budget_results
    result['maximum_profile_sum_error'] = maximum_error
    return result


def assess_refinement_gate(
    evaluation: Mapping[str, object],
    primary_policy: str,
    budgets: Sequence[int],
    minimum_p95_reduction: float,
) -> dict[str, object]:
    """Apply the pre-registered regret and analytical-cost gate."""

    if not 0 <= minimum_p95_reduction <= 1:
        raise ValueError('minimum_p95_reduction must lie in [0, 1]')
    policy = evaluation['per_policy'][primary_policy]
    per_budget = {}
    for budget in sorted({int(value) for value in budgets}):
        metrics = policy[str(budget)]
        reduction = float(metrics['weighted_p95_regret_reduction_fraction'])
        work_ratio = float(metrics['estimated_work_ratio_to_full_search'])
        per_budget[str(budget)] = {
            'p95_reduction_passed': reduction >= minimum_p95_reduction,
            'work_ratio_passed': work_ratio < 1.0,
            'passed': reduction >= minimum_p95_reduction and work_ratio < 1.0,
        }
    return {
        'primary_policy': primary_policy,
        'minimum_weighted_p95_regret_reduction': minimum_p95_reduction,
        'per_budget': per_budget,
        'passed': bool(per_budget) and all(item['passed'] for item in per_budget.values()),
    }
