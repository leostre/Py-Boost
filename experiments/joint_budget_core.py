"""Development-only decisions for the shared global leaf/rank/residual budget."""
from dataclasses import asdict, dataclass
from enum import Enum

from experiments.sparse_leaf_core import Cost, select_candidate


class SelectionState(str, Enum):
    NO_MEMORY_FEASIBLE = 'no_memory_feasible'
    NO_QUALITY_FEASIBLE = 'no_quality_feasible'
    ACCEPTED = 'accepted'


@dataclass(frozen=True)
class JointChoice:
    state: SelectionState
    candidate: dict | None
    feasible_count: int


def choose_joint(rows, reference_quality, budget, family, residual_percent, reference_parameters):
    if family not in ('leaf_only', 'rank_only', 'rank_residual', 'joint_static'):
        raise ValueError('unknown joint-budget family')
    if residual_percent not in (0, 1, 2, 5):
        raise ValueError('unregistered residual cap')
    def allowed(row):
        if row['nnz']*100 > residual_percent*reference_parameters:
            return False
        if family == 'leaf_only':
            return row['kind'] == 'full'
        if family == 'rank_only':
            return row['depth'] == 5 and row['kind'] == 'rank_only'
        if family == 'rank_residual':
            return row['depth'] == 5 and row['kind'] in ('rank_only', 'sparse')
        return True
    feasible = [r for r in rows if allowed(r) and budget.allows(Cost(r['parameters'], r['payload_bytes']))]
    if not feasible:
        return JointChoice(SelectionState.NO_MEMORY_FEASIBLE, None, 0)
    chosen, passed = select_candidate(feasible, reference_quality, budget)
    return JointChoice(SelectionState.ACCEPTED if passed else SelectionState.NO_QUALITY_FEASIBLE,
                       chosen, len(feasible))


def choice_record(choice):
    return {**asdict(choice), 'state': choice.state.value}
