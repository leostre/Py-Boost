"""Validation-only minimum-byte selection with explicit infeasibility."""
from dataclasses import dataclass

from experiments.shared_leaf_basis import quality_guard
from experiments.sparse_leaf_core import Budget, Cost
from experiments.uniform_threshold_core import family_allows


@dataclass(frozen=True)
class QualifiedMinimum:
    candidate: dict
    feasible_count: int
    qualified_count: int
    frontier: tuple


@dataclass(frozen=True)
class NoQualifiedMinimum:
    feasible_count: int


def memory_key(row):
    return (row['payload_bytes'], row['parameters'], -row['validation']['f1_macro'],
            row['validation']['loss'], row['candidate_id'])


def quality_frontier(qualified_rows):
    """Non-dominated (bytes, parameters, -macro-F1); canonicalize exact ties.

    Rows already pass all quality guards. Sorted byte order means a later row
    cannot dominate an earlier one, except exact byte ties handled by P/F1 order.
    """
    frontier = []
    for row in sorted(qualified_rows, key=memory_key):
        if not any(p['parameters'] <= row['parameters']
                   and p['validation']['f1_macro'] >= row['validation']['f1_macro']
                   for p in frontier):
            frontier.append(row)
    return tuple(frontier)


def choose_minimum(rows, reference_quality, reference_parameters, family, budget):
    """No test data, scores or fallback outside the qualified set exist here."""
    if not isinstance(budget, Budget):
        raise TypeError('Budget required')
    feasible = [r for r in rows if family_allows(r, family, reference_parameters)
                and budget.allows(Cost(r['parameters'], r['payload_bytes']))]
    qualified = [r for r in feasible if quality_guard(r['validation'], reference_quality)]
    if not qualified:
        return NoQualifiedMinimum(len(feasible))
    return QualifiedMinimum(min(qualified, key=memory_key), len(feasible), len(qualified),
                            quality_frontier(qualified))
