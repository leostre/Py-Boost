"""Pure, validation-only quality-efficiency decisions (no fitted risk probabilities)."""
from dataclasses import dataclass
from math import isfinite, log, log1p, expm1


@dataclass(frozen=True)
class Quality:
    loss: float
    f1_macro: float
    f1_weighted: float

    def __post_init__(self):
        if not isfinite(self.loss) or self.loss <= 0:
            raise ValueError('loss must be finite and positive')
        if any(not isfinite(x) or not 0 <= x <= 1 for x in (self.f1_macro, self.f1_weighted)):
            raise ValueError('F1 must be finite in [0, 1]')


def deficit(current: Quality, reference: Quality) -> float:
    """Maximum relative gap to CE <= 1.05 ref and both F1 >= .95 ref."""
    gaps = [0., current.loss / reference.loss - 1.05]
    for a, b in ((current.f1_macro, reference.f1_macro),
                 (current.f1_weighted, reference.f1_weighted)):
        if b > 0:  # A zero reference imposes no F1 constraint.
            gaps.append(.95 - a / b)
    return max(gaps)


@dataclass(frozen=True)
class WindowGain:
    depth: int
    start: int
    end: int
    parameter_delta: int
    deficit_start: float
    deficit_end: float

    def __post_init__(self):
        if self.depth not in (3, 5) or self.start < 60 or self.end - self.start != 20:
            raise ValueError('expected a completed post-warmup 20-tree depth 3/5 window')
        if self.parameter_delta <= 0:
            raise ValueError('positive actual coefficient cost required')
        if any(not isfinite(x) or x < 0 for x in (self.deficit_start, self.deficit_end)):
            raise ValueError('invalid deficit')


@dataclass(frozen=True)
class Forecast:
    depth: int
    windows: int
    parameter_rate: float
    tree_rate: float
    required_parameters: float | None
    required_trees: float | None
    feasible: bool
    score: float | None
    block_decay: float = 1.
    forecast_model: str = 'constant'


def projected_gain(rate, block_decay, trees):
    """Continuous geometric sum; integer trees match a direct discrete oracle."""
    if (not isfinite(rate) or rate < 0 or not isfinite(block_decay)
            or not 0 <= block_decay <= 1 or not isfinite(trees) or trees < 0):
        raise ValueError('invalid gain projection')
    if block_decay == 1:
        return rate * trees
    if block_decay == 0 or rate == 0:
        return 0.
    log_a = log(block_decay) / 20.
    # Stable near rho=1: avoid subtracting two nearly equal numbers.
    a_over_one_minus_a = 1. / expm1(-log_a)
    return rate * a_over_one_minus_a * (-expm1(trees * log_a))


def required_trees(rate, block_decay, gap):
    """None denotes no finite solution, not a guarantee about true convergence."""
    if (not isfinite(gap) or gap < 0 or not isfinite(rate) or rate < 0
            or not isfinite(block_decay) or not 0 <= block_decay <= 1):
        raise ValueError('invalid inverse projection')
    if gap == 0:
        return 0.
    if rate == 0 or block_decay == 0:
        return None
    if block_decay == 1:
        return gap / rate
    log_a = log(block_decay) / 20.
    limit = rate / expm1(-log_a)
    if gap >= limit:
        return None
    fraction = gap * expm1(-log_a) / rate
    if fraction >= 1:
        return None
    return log1p(-fraction) / log_a


@dataclass(frozen=True)
class Decision:
    depth: int
    reason: str
    forecasts: tuple[Forecast, ...] = ()


def plan_depth(*, step, remaining_parameters, remaining_trees, current_deficit,
               windows, history, incumbent, deceleration=False):
    if (step < 60 or remaining_parameters <= 0 or remaining_trees <= 0
            or not isfinite(current_deficit) or current_deficit < 0
            or history not in (1, 2) or incumbent not in (3, 5)):
        raise ValueError('invalid planning state')
    if deceleration and history != 2:
        raise ValueError('deceleration requires two-window history')
    if any(w.end > step for w in windows):
        raise ValueError('future windows cannot control a causal decision')
    if step == 60:
        return Decision(3, 'initial_probe_d3')
    if step == 80:
        return Decision(5, 'initial_probe_d5')
    if current_deficit == 0:
        return Decision(incumbent, 'quality_confirmation_pending')
    projections = []
    latest = {}
    for depth in (3, 5):
        selected = sorted((w for w in windows if w.depth == depth), key=lambda w: w.end)[-history:]
        latest[depth] = max((w.end for w in selected), default=60)
        # Clip AFTER summing signed gains, not each gain independently.
        gain = max(0., sum(w.deficit_start - w.deficit_end for w in selected))
        cost = sum(w.parameter_delta for w in selected)
        trees = sum(w.end - w.start for w in selected)
        pr, tr = (gain / cost if cost else 0.), (gain / trees if trees else 0.)
        rho = 1.
        if deceleration and len(selected) == 2:
            old, new = (w.deficit_start - w.deficit_end for w in selected)
            if old > 0:
                rho = min(1., max(0., new) / old)
        if deceleration:
            t = required_trees(tr, rho, current_deficit)
            p = t * cost / trees if t is not None and trees else None
        else:
            p, t = (current_deficit / pr if pr else None), (current_deficit / tr if tr else None)
        feasible = p is not None and p <= remaining_parameters and t <= remaining_trees
        score = None if p is None else (p if feasible else max(p / remaining_parameters, t / remaining_trees))
        projections.append(Forecast(depth, len(selected), pr, tr, p, t, feasible, score,
                                    rho, 'geometric' if deceleration else 'constant'))
    stale = [d for d in (3, 5) if step - latest[d] >= 60]
    if stale:
        return Decision(min(stale, key=lambda d: (latest[d], d)), 'scheduled_refresh', tuple(projections))
    viable = [p for p in projections if p.score is not None]
    if not viable:
        return Decision(incumbent, 'no_positive_gain_hold', tuple(projections))
    best = min(viable, key=lambda p: (not p.feasible, p.score, p.depth))
    previous = next(p for p in projections if p.depth == incumbent)
    if (previous.score is not None and previous.feasible == best.feasible
            and best.score >= .95 * previous.score):
        return Decision(incumbent, 'hysteresis_hold', tuple(projections))
    return Decision(best.depth, 'efficiency_forecast', tuple(projections))
