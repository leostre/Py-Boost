"""Pure, causal budget rules; no test labels, GPU state, or I/O."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GrowthDecision:
    depth: int
    stop: bool
    reason: str


def affordable_depth(preferred, remaining, outputs):
    """Conservative bound: a depth-d SingleSplitter tree has <= 2**d leaves."""
    if preferred < 1 or remaining < 0 or outputs < 1:
        raise ValueError('invalid structural budget')
    capacity = remaining // outputs
    if capacity < 2:
        return 0
    return min(preferred, int(capacity).bit_length() - 1)


def growth_decision(*, mode, step, depth, losses, remaining, outputs,
                    check_every=20, warmup=40, improvement_floor=.002, max_depth=7):
    if mode not in ('fixed', 'schedule', 'adaptive'):
        raise ValueError('unknown growth policy')
    if step < 0 or check_every < 1 or warmup < check_every or not 1 <= depth <= max_depth:
        raise ValueError('invalid iteration policy')
    values = np.asarray(losses, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('validation losses must be finite and positive')
    preferred, reason = depth, 'continue'
    if mode == 'schedule':
        preferred = 3 if step < 40 else 5 if step < 120 else 7
        reason = 'fixed_schedule'
    elif mode == 'adaptive' and step >= warmup and step % check_every == 0:
        if len(values) <= check_every:
            raise ValueError('insufficient causal validation history')
        gain = (values[-check_every - 1] - values[-1]) / values[-check_every - 1]
        if gain < improvement_floor:
            if depth >= max_depth:
                return GrowthDecision(depth, True, 'validation_plateau_at_allowed_depth')
            preferred, reason = min(depth + 2, max_depth), 'validation_plateau_increase_capacity'
    feasible = affordable_depth(preferred, remaining, outputs)
    if feasible == 0:
        return GrowthDecision(1, True, 'parameter_cap')
    return GrowthDecision(feasible, False,
                          'remaining_budget_limits_depth' if feasible < preferred else reason)


def spectral_profile(gradient, hessian):
    """Normalized separate G/H covariance profiles, not a parameter-space Hessian."""
    g, h = np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)
    if g.ndim != 2 or h.shape != g.shape or min(g.shape) < 2:
        raise ValueError('aligned nontrivial G/H matrices required')
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
        raise ValueError('nonfinite G/H')
    profile = np.zeros((g.shape[1], g.shape[1]))
    for a in (g, h):
        centered = a - a.mean(axis=0)
        gram = centered.T @ centered
        trace = np.trace(gram)
        if trace > 0:
            profile += gram / trace / 2
    return profile


def spectral_state(profile, effective_rows):
    p = np.asarray(profile, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or effective_rows < 2:
        raise ValueError('square profile and sample size required')
    eigen = np.maximum(np.linalg.eigvalsh((p + p.T) / 2), 0)[::-1]
    total = eigen.sum()
    if total <= 1e-15:
        return {'effective_rank': 0., 'energy_rank': 1, 'above_noise_edge': 0,
                'noise_edge': 0.}
    rank = total * total / np.square(eigen).sum()
    energy_rank = int(np.searchsorted(np.cumsum(eigen), .9 * total) + 1)
    # An MP-inspired diagnostic only: dependence violates the IID null assumptions.
    edge = total / len(eigen) * (1 + math.sqrt(len(eigen) / effective_rows)) ** 2
    return {'effective_rank': float(rank), 'energy_rank': energy_rank,
            'above_noise_edge': int(np.count_nonzero(eigen > edge)), 'noise_edge': float(edge)}


def output_budget(outputs, state, stable, fraction_floor=.25):
    if outputs < 1 or not 0 < fraction_floor <= 1:
        raise ValueError('invalid output budget')
    if not stable:
        return outputs
    signal = max(1, int(state['energy_rank']), int(state['above_noise_edge']))
    return min(outputs, max(int(math.ceil(outputs * fraction_floor)), signal))


def stable_profile_window(shifts, noise, step, frames):
    """Require both typical reproducibility and consistency of the latest state."""
    values = np.asarray(shifts, dtype=float)
    if not np.isfinite(noise) or noise < 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError('finite nonnegative shifts and noise required')
    if step < 40 or frames != 4 or len(values) != 3:
        return False
    limit = 1.5 * max(noise, 1e-12)
    return bool(np.median(values) <= limit and values[-1] <= limit)
