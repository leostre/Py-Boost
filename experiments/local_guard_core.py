"""Pure, bounded local-veto protocols. Never handles models, labels, or I/O."""
from dataclasses import dataclass, asdict
from enum import Enum


class GuardPolicy(str, Enum):
    STRICT = 'strict'
    WARMUP2 = 'warmup2'
    CONFIRM2 = 'confirm2'


class Outcome(str, Enum):
    PASS = 'local_pass'
    FAIL = 'local_fail'
    NO_CANDIDATE = 'no_candidate'


class StopCause(str, Enum):
    QUALITY = 'quality'
    RESOURCE = 'resource'


@dataclass(frozen=True)
class ActiveGuard:
    next_block: int = 0
    consecutive_failures: int = 0

    def __post_init__(self):
        for value in (self.next_block, self.consecutive_failures):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('invalid guard counter')
        if self.consecutive_failures > self.next_block:
            raise ValueError('failure streak exceeds processed checkpoints')

    @property
    def last_confirmed_block(self):
        return self.next_block-self.consecutive_failures-1


@dataclass(frozen=True)
class StoppedGuard:
    retained_through: int
    evaluated_block: int
    reason: StopCause


@dataclass(frozen=True)
class CommitCheckpoint:
    state: ActiveGuard
    locally_confirmed: bool


@dataclass(frozen=True)
class StopAndRollback:
    state: StoppedGuard


def advance_guard(policy, state, block, outcome):
    """All policies prefer a feasible local candidate before considering FAIL.

    The shell is responsible for that common candidate choice and for applying
    the hard cost cap BEFORE emitting an event. A resource failure never warms up.
    """
    policy, outcome = GuardPolicy(policy), Outcome(outcome)
    if not isinstance(state, ActiveGuard):
        raise ValueError('guard already stopped')
    if isinstance(block, bool) or not isinstance(block, int) or block != state.next_block:
        raise ValueError('guard event duplicated or out of order')
    if outcome is Outcome.PASS:
        return CommitCheckpoint(ActiveGuard(block+1, 0), True)
    tolerated = (outcome is Outcome.FAIL and
        ((policy is GuardPolicy.WARMUP2 and block < 2) or
         (policy is GuardPolicy.CONFIRM2 and state.consecutive_failures == 0)))
    if tolerated:
        return CommitCheckpoint(ActiveGuard(block+1, state.consecutive_failures+1), False)
    return StopAndRollback(StoppedGuard(state.last_confirmed_block, block,
        StopCause.RESOURCE if outcome is Outcome.NO_CANDIDATE else StopCause.QUALITY))


def state_record(state):
    record = asdict(state)
    if isinstance(state, ActiveGuard):
        return dict(kind='active', last_confirmed_block=state.last_confirmed_block, **record)
    return dict(kind='stopped', **record)
