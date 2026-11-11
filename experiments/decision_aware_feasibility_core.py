"""Pure data-sufficiency checks for decision-aware shortlist analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ArtifactCoverage:
    records: int
    material_nodes: int
    nonempty_diagnostic_shortlists: int
    nonempty_full_shortlists: int
    complete_shortlists: Mapping[int, int]
    output_profile_records: int
    selected_output_records: int


def candidate_identity(candidate: Mapping[str, object]) -> tuple[int, int, bool]:
    """Return the stable identity of a split candidate."""

    return (
        int(candidate['feature']),
        int(candidate['split_bin']),
        bool(candidate['nan_left']),
    )


def _has_output_profiles(record: Mapping[str, object]) -> bool:
    profiles = record.get('candidate_output_contributions')
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        return False
    shortlist = record.get('diagnostic_top_k')
    if not isinstance(shortlist, Sequence) or not shortlist:
        return False
    if len(profiles) < len(shortlist):
        return False
    output_count = record.get('n_outputs')
    if not isinstance(output_count, int) or output_count <= 0:
        return False
    return all(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) == output_count
        for values in profiles[:len(shortlist)]
    )


def audit_records(
    records: Iterable[Mapping[str, object]],
    shortlist_sizes: Sequence[int],
    material_gain_epsilon: float,
) -> ArtifactCoverage:
    """Count evidence needed for shortlist recall and contribution profiles."""

    if material_gain_epsilon < 0:
        raise ValueError('material_gain_epsilon must be non-negative')
    sizes = tuple(sorted({int(value) for value in shortlist_sizes}))
    if not sizes or sizes[0] <= 0:
        raise ValueError('shortlist sizes must be positive')

    total = material = diagnostic = full = profiles = selections = 0
    complete = {size: 0 for size in sizes}
    for record in records:
        total += 1
        gain = float(record.get('full_best_gain') or 0.0)
        if bool(record.get('full_split_available')) and gain > material_gain_epsilon:
            material += 1
        diagnostic_list = record.get('diagnostic_top_k')
        full_list = record.get('full_top_k')
        diagnostic_list = diagnostic_list if isinstance(diagnostic_list, list) else []
        full_list = full_list if isinstance(full_list, list) else []
        diagnostic += int(bool(diagnostic_list))
        full += int(bool(full_list))
        for size in sizes:
            complete[size] += int(len(diagnostic_list) >= size)
        profiles += int(_has_output_profiles(record))
        selected = record.get('selected_output_indices')
        selections += int(isinstance(selected, list) and bool(selected))
    return ArtifactCoverage(
        records=total,
        material_nodes=material,
        nonempty_diagnostic_shortlists=diagnostic,
        nonempty_full_shortlists=full,
        complete_shortlists=complete,
        output_profile_records=profiles,
        selected_output_records=selections,
    )


def assess_sufficiency(
    coverage: ArtifactCoverage,
    shortlist_sizes: Sequence[int],
    budgets: Sequence[int],
) -> dict[str, object]:
    """Return an explicit gate verdict without inventing unavailable metrics."""

    sizes = tuple(sorted({int(value) for value in shortlist_sizes}))
    requested_budgets = tuple(sorted({int(value) for value in budgets}))
    reasons: list[str] = []
    if coverage.material_nodes == 0:
        reasons.append('no material-gain nodes are present')
    for size in sizes:
        if coverage.complete_shortlists.get(size, 0) < coverage.material_nodes:
            reasons.append(
                f'diagnostic shortlist of size {size} is not recorded for every material node'
            )
    if coverage.output_profile_records < coverage.material_nodes:
        reasons.append('per-output candidate contribution profiles are missing')
    if coverage.selected_output_records < coverage.material_nodes:
        reasons.append('selected output identities are missing from node records')
    if not requested_budgets:
        reasons.append('no budgets were requested')
    sufficient = not reasons
    return {
        'sufficient_for_stage_a_metrics': sufficient,
        'requested_shortlist_sizes': list(sizes),
        'requested_budgets': list(requested_budgets),
        'missing_evidence': reasons,
        'metrics_computed': sufficient,
        'gate_passed': False if not sufficient else None,
        'coverage': asdict(coverage),
    }


def minimal_logging_contract(shortlist_size: int = 8) -> dict[str, object]:
    """Machine-readable minimum for a future offline shortlist evaluation."""

    if shortlist_size <= 0:
        raise ValueError('shortlist_size must be positive')
    return {
        'schema_version': 1,
        'scope': 'one record per diagnostic budget and active tree node',
        'required_scalar_fields': [
            'dataset', 'split_seed', 'model_seed', 'iteration', 'group', 'depth',
            'node_id', 'diagnostic_kind', 'diagnostic_output_budget', 'n_outputs',
            'full_best_feature', 'full_best_split_bin', 'full_best_nan_left',
            'full_best_gain', 'full_split_available', 'lambda_l2',
            'min_gain_to_split', 'initial_search_seconds', 'refinement_seconds',
        ],
        'required_array_fields': {
            'selected_output_indices': ['diagnostic_output_budget'],
            'diagnostic_top_k': [shortlist_size],
            'candidate_output_contributions': [shortlist_size, 'n_outputs'],
        },
        'candidate_fields': [
            'feature', 'split_bin', 'nan_left', 'diagnostic_gain', 'full_gain',
        ],
        'semantic_checks': [
            'candidate identities are unique within each shortlist',
            'sum(candidate_output_contributions[k]) equals diagnostic_top_k[k].full_gain',
            'full winner identity is included separately even when absent from the shortlist',
            'selected output identities are unique and their count equals the recorded budget',
            'records sharing a node and different budgets have the same full winner and full gain',
        ],
        'storage_guidance': (
            'Store dense per-output contributions as compressed float32 arrays keyed by compact '
            'integer node/candidate tables; JSON is metadata only. Raw G/H candidate sums are '
            'optional audit enrichment, not part of the minimum Stage-A contract.'
        ),
        'not_substitutes': [
            'one selected sketched winner',
            'leave-one-output regret for the full winner',
            'global gradient or spectral output weights',
            'trajectory G/H snapshots without node assignments and candidate identities',
        ],
    }


def evaluate_shortlist_collection(
    records: Sequence[Mapping[str, object]],
    shortlist_sizes: Sequence[int],
    budgets: Sequence[int],
    material_gain_epsilon: float,
    contribution_tolerance: float = 1e-4,
    contribution_relative_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Evaluate shortlist recall and the integrity of newly collected profiles."""

    sizes = tuple(sorted({int(value) for value in shortlist_sizes}))
    requested_budgets = tuple(sorted({int(value) for value in budgets}))
    if not sizes or sizes[0] <= 0:
        raise ValueError('shortlist sizes must be positive')
    if not requested_budgets or requested_budgets[0] <= 0:
        raise ValueError('budgets must be positive')

    per_budget: dict[str, object] = {}
    maximum_profile_error = 0.0
    invalid_profile_records = 0
    invalid_selection_records = 0
    winner_by_node: dict[tuple[int, int, int, int], tuple[tuple[int, int, bool], float]] = {}
    inconsistent_full_winners = 0
    for budget in requested_budgets:
        subset = [
            record for record in records
            if int(record.get('diagnostic_output_budget') or 0) == budget
        ]
        material = [
            record for record in subset
            if bool(record.get('full_split_available'))
            and float(record.get('full_best_gain') or 0.0) > material_gain_epsilon
        ]
        recall = {}
        for size in sizes:
            hits = 0
            for record in material:
                full_identity = (
                    int(record['full_best_feature']),
                    int(record['full_best_split_bin']),
                    bool(record['full_best_nan_left']),
                )
                shortlist = record.get('diagnostic_top_k') or []
                hits += int(
                    full_identity in {
                        candidate_identity(candidate)
                        for candidate in shortlist[:size]
                    }
                )
            recall[str(size)] = (hits / len(material)) if material else None
        per_budget[str(budget)] = {
            'records': len(subset),
            'material_nodes': len(material),
            'recall_at_k': recall,
        }

    for record in records:
        profiles = record.get('candidate_output_contributions') or []
        shortlist = record.get('diagnostic_top_k') or []
        profile_valid = len(profiles) == len(shortlist) and bool(shortlist)
        if profile_valid:
            for candidate, contributions in zip(shortlist, profiles):
                expected = float(candidate['full_gain'])
                error = abs(sum(float(value) for value in contributions) - expected)
                maximum_profile_error = max(maximum_profile_error, error)
                permitted_error = (
                    contribution_tolerance
                    + contribution_relative_tolerance * abs(expected)
                )
                profile_valid = profile_valid and error <= permitted_error
        invalid_profile_records += int(not profile_valid)

        selected = record.get('selected_output_indices')
        budget = int(record.get('diagnostic_output_budget') or 0)
        selection_valid = (
            isinstance(selected, list)
            and len(selected) == budget
            and len(set(int(index) for index in selected)) == budget
        )
        invalid_selection_records += int(not selection_valid)

        key = (
            int(record.get('iteration') or 0),
            int(record.get('group') or 0),
            int(record.get('depth') or 0),
            int(record.get('node_id') or 0),
        )
        winner = (
            (
                int(record['full_best_feature']),
                int(record['full_best_split_bin']),
                bool(record['full_best_nan_left']),
            ),
            float(record['full_best_gain']),
        )
        previous = winner_by_node.setdefault(key, winner)
        inconsistent_full_winners += int(previous != winner)

    coverage = audit_records(records, sizes, material_gain_epsilon)
    return {
        'coverage': asdict(coverage),
        'per_budget': per_budget,
        'maximum_profile_sum_error': maximum_profile_error,
        'contribution_tolerance': contribution_tolerance,
        'contribution_relative_tolerance': contribution_relative_tolerance,
        'invalid_profile_records': invalid_profile_records,
        'invalid_selection_records': invalid_selection_records,
        'inconsistent_full_winners': inconsistent_full_winners,
        'contract_valid': (
            bool(records)
            and invalid_profile_records == 0
            and invalid_selection_records == 0
            and inconsistent_full_winners == 0
        ),
    }


def assess_shortlist_recall_gate(
    evaluation: Mapping[str, object],
    shortlist_sizes: Sequence[int],
    budgets: Sequence[int],
    minimum_recall: float,
) -> dict[str, object]:
    """Find the smallest nested shortlist that passes for every budget."""

    if not 0 <= minimum_recall <= 1:
        raise ValueError('minimum_recall must be within [0, 1]')
    sizes = tuple(sorted({int(value) for value in shortlist_sizes}))
    requested_budgets = tuple(sorted({int(value) for value in budgets}))
    if not sizes or sizes[0] <= 0:
        raise ValueError('shortlist sizes must be positive')
    if not requested_budgets or requested_budgets[0] <= 0:
        raise ValueError('budgets must be positive')

    per_budget = evaluation['per_budget']
    minimum_by_budget: dict[str, int | None] = {}
    monotonicity_violations: list[dict[str, object]] = []
    for budget in requested_budgets:
        recalls = per_budget[str(budget)]['recall_at_k']
        previous = -1.0
        minimum = None
        for size in sizes:
            value = recalls[str(size)]
            if value is not None:
                numeric = float(value)
                if numeric < previous:
                    monotonicity_violations.append({
                        'budget': budget,
                        'shortlist_size': size,
                        'previous_recall': previous,
                        'recall': numeric,
                    })
                previous = numeric
                if minimum is None and numeric >= minimum_recall:
                    minimum = size
        minimum_by_budget[str(budget)] = minimum

    joint_sizes = [
        size for size in sizes
        if all(
            per_budget[str(budget)]['recall_at_k'][str(size)] is not None
            and float(per_budget[str(budget)]['recall_at_k'][str(size)]) >= minimum_recall
            for budget in requested_budgets
        )
    ]
    joint_minimum = min(joint_sizes) if joint_sizes else None
    contract_valid = bool(evaluation.get('contract_valid'))
    return {
        'threshold': minimum_recall,
        'minimum_viable_k_by_budget': minimum_by_budget,
        'joint_minimum_viable_k': joint_minimum,
        'monotonicity_violations': monotonicity_violations,
        'passed': bool(
            contract_valid
            and not monotonicity_violations
            and joint_minimum is not None
        ),
    }
