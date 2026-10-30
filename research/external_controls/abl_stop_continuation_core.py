"""Pure SR2 planning, restored controller state and paid-tree reservations."""
from dataclasses import dataclass
from abl_stop_rule_core import Anchor, Candidate, Phase, Quality, State

EXPECTED_STARTS = {
    'mediamill_1104': {'current': 2, 'history': 2},
    'mediamill_1114': {'current': 1, 'mean': 2, 'history': 2},
    'mediamill_1124': {'mean': 1, 'history': 1},
    'mnist_3104': {'functional': 3, 'current': 2, 'history': 2},
    'mnist_3114': {'functional': 2, 'mean': 2, 'history': 3},
    'mnist_3124': {'functional': 2, 'current': 2, 'history': 2},
}


@dataclass(frozen=True)
class Job:
    case: str
    kind: str
    start_block: int

    @property
    def blocks(self):
        return tuple(range(self.start_block+1, 5))

    @property
    def maximum_new_trees(self):
        return 40*len(self.blocks)


def plan(cases):
    jobs = []
    for case in cases:
        if case['case'] not in EXPECTED_STARTS:
            continue
        for path in case['paths']:
            if not path['censored']:
                continue
            if path['terminal'] != 'censored_old_rejected_block_now_accepted':
                raise ValueError('only approved old-rejected continuations')
            last = path['attempts'][-1]
            if last['action'] != 'accepted' or last['snapshot'] != path['last_observed']:
                raise ValueError('continuation must start from the last actual accepted state')
            jobs.append(Job(case['case'], path['path'], last['block']))
    expected = {(c, k, b) for c, v in EXPECTED_STARTS.items() for k, b in v.items()}
    actual = {(j.case, j.kind, j.start_block) for j in jobs}
    if actual != expected or len(jobs) != 16 or sum(j.maximum_new_trees for j in jobs) != 1320:
        raise ValueError('approved 16-path /1320-tree continuation plan changed')
    return tuple(sorted(jobs, key=lambda j: (j.case, j.kind)))


def candidate_from_json(row):
    return Candidate(**dict(row, quality=Quality(**row['quality'])))


def state_from_json(row):
    return State(Anchor(row['anchor']['token'], Quality(**row['anchor']['quality'])),
        row['budget_bytes'], candidate_from_json(row['current']), Phase(row['phase']),
        None if row['certificate'] is None else candidate_from_json(row['certificate']))


def reserve(ledger, job, block):
    """Reserve BEFORE fitting, including any block later rejected by the rule."""
    if block not in job.blocks:
        raise ValueError('outside approved continuation horizon')
    key = (job.case, job.kind, block)
    seen = {(a['case'], a['kind'], a['block']) for a in ledger['attempts']}
    if key in seen:
        raise ValueError('duplicate paid fit')
    if ledger['reserved_rounds'] != 40*len(ledger['attempts']):
        raise ValueError('inconsistent paid ledger')
    if ledger['reserved_rounds']+40 > 1320 or ledger['phase_limit'] != 1320:
        raise ValueError('approved new-tree cap exhausted')
    return dict(ledger, reserved_rounds=ledger['reserved_rounds']+40,
        attempts=[*ledger['attempts'], dict(case=job.case, kind=job.kind,
            block=block, rounds=40, state='reserved')])


def initial_arrays(model, train_x, calibration_x):
    """Training sees raw compact scores, independent of external T/thresholds."""
    import numpy as np
    return dict(train=model.raw(train_x).astype(np.float32),
        validation=model.raw(calibration_x))
