"""Independent budget channels and fixed uniform references; no scores or I/O."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MechanismBudget:
    topk: int
    depth_ceiling: int


def channel_budget(outputs, proposed_topk, proposed_depth, channels):
    if channels not in ('joint', 'outputs', 'depth'):
        raise ValueError('unknown budget channels')
    if not 1 <= proposed_topk <= outputs or not 3 <= proposed_depth <= 7:
        raise ValueError('invalid proposed budget')
    return MechanismBudget(proposed_topk if channels != 'depth' else outputs,
                           proposed_depth if channels != 'outputs' else 7)


def static_budget(outputs, step, fraction, ceiling, activation_step=60):
    if outputs < 1 or step < 0 or not 0 < fraction <= 1 or ceiling not in (3, 5, 7) or activation_step < 0:
        raise ValueError('invalid static budget')
    if step < activation_step:
        return MechanismBudget(outputs, 7)
    return MechanismBudget(max(1, math.ceil(outputs * fraction)), ceiling)


def uniform_reference_settings(records, dataset, seed):
    """Explicit d5/lambda1/t200 full control, selected by metadata only."""
    rows = []
    for r in records:
        if r['dataset'] != dataset or r['seed'] != seed or r['boosting_steps'] != 200 or r['lambda_l2'] != 1.:
            continue
        if (r['method'] == 'full' and r['max_depth'] == 5) or (
                r['method'] == 'full_reference_l21' and r.get('is_full_reference') is True):
            rows.append(r)
    if len(rows) != 1 or not isinstance(rows[0]['parameters'], int) or rows[0]['parameters'] < 1:
        raise ValueError('exactly one positive uniform full reference is required')
    r = rows[0]
    return dict(max_depth=5, lambda_l2=1., parameters=r['parameters'],
                candidate_id=r['candidate_id'], data_sha256=r['data_sha256'])
