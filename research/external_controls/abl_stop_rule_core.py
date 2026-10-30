"""Pure fixed-anchor controller. Selection/test labels cannot enter this API."""
from dataclasses import dataclass
from enum import Enum
import math

NLL_EPSILON = 1e-6  # Existing absolute NLL indifference band, not a confidence bound.


@dataclass(frozen=True)
class Quality:
    loss: float
    f1_macro: float
    f1_weighted: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.loss, self.f1_macro, self.f1_weighted)):
            raise ValueError('finite quality required')
        if self.loss < 0 or not 0 <= self.f1_macro <= 1 or not 0 <= self.f1_weighted <= 1:
            raise ValueError('quality outside domain')


@dataclass(frozen=True)
class Anchor:
    token: str
    quality: Quality


@dataclass(frozen=True)
class Candidate:
    uid: str
    representation: str
    raw_nll: float
    quality: Quality
    cost_bytes: int
    reconstruction_mse: float
    ordinal: int

    def __post_init__(self):
        if not self.uid or not self.representation or isinstance(self.cost_bytes, bool) or self.cost_bytes < 1:
            raise ValueError('explicit candidate identity and positive bytes required')
        if not math.isfinite(self.raw_nll) or self.raw_nll < 0:
            raise ValueError('finite raw NLL required')
        if not math.isfinite(self.reconstruction_mse) or self.reconstruction_mse < 0:
            raise ValueError('finite reconstruction error required')


class Phase(str, Enum):
    SEEKING = 'seeking_quality'
    QUALIFIED = 'qualified'


class StopReason(str, Enum):
    MEMORY = 'no_candidate_within_memory'
    PROGRESS = 'no_useful_candidate'
    QUALITY = 'no_useful_candidate_within_fixed_quality_guard'


def quality_failures(q, r):
    failed = []
    if q.loss > 1.05*r.loss:
        failed.append('loss')
    if q.f1_macro < .95*r.f1_macro:
        failed.append('f1_macro')
    if q.f1_weighted < .95*r.f1_weighted:
        failed.append('f1_weighted')
    return tuple(failed)


@dataclass(frozen=True)
class State:
    anchor: Anchor
    budget_bytes: int
    current: Candidate
    phase: Phase
    certificate: Candidate | None

    def __post_init__(self):
        if self.budget_bytes < 1 or self.current.cost_bytes > self.budget_bytes:
            raise ValueError('current model exceeds fixed memory cap')
        if (self.phase is Phase.QUALIFIED) != (self.certificate is not None):
            raise ValueError('qualification requires an explicit certificate')
        if self.certificate is not None and (self.certificate.cost_bytes > self.budget_bytes or
                quality_failures(self.certificate.quality, self.anchor.quality)):
            raise ValueError('invalid fixed-anchor certificate')


@dataclass(frozen=True)
class Gates:
    uid: str
    memory: bool
    useful: bool
    fixed_quality: bool
    raw_nll_gain: float
    quality_failures: tuple[str, ...]


@dataclass(frozen=True)
class Accepted:
    state: State
    gates: tuple[Gates, ...]
    entered_qualification: bool


@dataclass(frozen=True)
class Stopped:
    state: State
    reason: StopReason
    gates: tuple[Gates, ...]


def original_order(c):
    return (c.reconstruction_mse, -c.quality.f1_macro, c.quality.loss,
        c.cost_bytes, c.ordinal, c.uid)


def initialize(candidates, anchor, budget_bytes):
    feasible = [c for c in candidates if c.cost_bytes <= budget_bytes]
    if not feasible:
        raise ValueError('saved initial model does not fit the unchanged budget')
    certified = [c for c in feasible if not quality_failures(c.quality, anchor.quality)]
    current = min(certified or feasible, key=original_order)
    return State(anchor, budget_bytes, current,
        Phase.QUALIFIED if certified else Phase.SEEKING, current if certified else None)


def decide(state, candidates):
    if len({c.uid for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate identities')
    gates = tuple(Gates(c.uid, c.cost_bytes <= state.budget_bytes,
        state.current.raw_nll-c.raw_nll > NLL_EPSILON,
        not quality_failures(c.quality, state.anchor.quality),
        state.current.raw_nll-c.raw_nll, quality_failures(c.quality, state.anchor.quality))
        for c in candidates)
    by = {c.uid: c for c in candidates}
    feasible = [g for g in gates if g.memory]
    useful = [g for g in feasible if g.useful]
    certified = [g for g in useful if g.fixed_quality]
    if not feasible:
        return Stopped(state, StopReason.MEMORY, gates)
    if not useful:
        return Stopped(state, StopReason.PROGRESS, gates)
    if state.phase is Phase.QUALIFIED and not certified:
        return Stopped(state, StopReason.QUALITY, gates)
    # Enter qualification as soon as a useful, fitting candidate can satisfy it.
    chosen = min((by[g.uid] for g in (certified or useful)), key=original_order)
    qualifies = not quality_failures(chosen.quality, state.anchor.quality)
    next_state = State(state.anchor, state.budget_bytes, chosen,
        Phase.QUALIFIED if qualifies else Phase.SEEKING,
        chosen if qualifies else state.certificate)
    return Accepted(next_state, gates, state.phase is Phase.SEEKING and qualifies)
