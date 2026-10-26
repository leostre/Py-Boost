"""Validated partitions, support objectives and frozen multilabel decisions."""
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from experiments.shared_leaf_basis import probabilities
from experiments.stable_budget_core import QualitySeries


class SupportStatus(str, Enum):
    NOOP = 'support_control_noop'
    IMPROVED = 'support_control_improved'
    UNRESOLVED = 'support_search_unresolved'


@dataclass(frozen=True)
class Partition:
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        a, b = (np.asarray(x) for x in (self.a, self.b))
        if any(x.ndim != 1 or x.dtype.kind not in 'iu' or not len(x) for x in (a, b)):
            raise ValueError('nonempty integer partition required')
        if not np.array_equal(np.sort(np.r_[a, b]), np.arange(len(a)+len(b))):
            raise ValueError('partition must be disjoint, exhaustive and unique')
        for key, x in [('a', a), ('b', b)]:
            owned = x.astype(np.int32, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, key, owned)


@dataclass(frozen=True)
class SupportResult:
    partition: Partition
    status: SupportStatus
    before: tuple
    after: tuple
    optimal: bool
    solver_statuses: tuple


def validated_target(y):
    y = np.asarray(y)
    if y.ndim != 2 or not y.size or not np.isin(y, [0, 1]).all():
        raise ValueError('nonempty binary indicator target required')
    return y.astype(bool)


def support_objective(y, partition, original, quota=5):
    y = validated_target(y)
    if quota < 1 or len(y) != len(partition.a)+len(partition.b):
        raise ValueError('invalid support objective')
    uncovered = deficit = 0
    for sign in (y, ~y):
        total = sign.sum(axis=0)
        a, b = sign[partition.a].sum(axis=0), sign[partition.b].sum(axis=0)
        q = np.minimum(quota, total//2)
        uncovered += int(((a == 0) & (total >= 2)).sum()+((b == 0) & (total >= 2)).sum())
        deficit += int(np.maximum(q-a, 0).sum()+np.maximum(q-b, 0).sum())
    deviation = int(np.abs(2*y[partition.a].sum(axis=0)-y.sum(axis=0)).sum())
    membership, base = np.zeros(len(y), bool), np.zeros(len(y), bool)
    membership[partition.a], base[original.a] = True, True
    return uncovered, deficit, deviation, int((membership != base).sum())


def repair_support(y, original, quota=5, seconds=60., solver=milp, clock=time.monotonic):
    """Bounded solver shell; pure objective independently validates each incumbent."""
    y = validated_target(y)
    before = support_objective(y, original, original, quota)
    if before[:2] == (0, 0):
        return SupportResult(original, SupportStatus.NOOP, before, before, False, ())
    n, c = y.shape
    # z[n], missing[4c], deficits[4c], abs-frequency-error[c].
    size = n+9*c
    lower, upper = np.zeros(size), np.full(size, np.inf)
    upper[:n+4*c] = 1
    integrality = np.zeros(size, int)
    integrality[:n+4*c] = 1
    rows, lo, hi = [], [], []

    def constraint(coeff, minimum=-np.inf, maximum=np.inf):
        rows.append(coeff)
        lo.append(minimum)
        hi.append(maximum)

    v = np.zeros(size)
    v[:n] = 1
    constraint(v, len(original.a), len(original.a))
    objectives = [np.zeros(size) for _ in range(4)]
    for sign_no, sign in enumerate((y, ~y)):
        for j in range(c):
            w = sign[:, j].astype(float)
            total = int(w.sum())
            q = min(quota, total//2)
            for side in range(2):
                k = (2*sign_no+side)*c+j
                base = np.zeros(size)
                base[:n] = w if side == 0 else -w
                offset = 0 if side == 0 else total
                if total >= 2:
                    v = base.copy()
                    v[n+k] = 1
                    constraint(v, 1-offset)
                    objectives[0][n+k] = 1
                v = base.copy()
                v[n+4*c+k] = 1
                constraint(v, q-offset)
                objectives[1][n+4*c+k] = 1
    for j in range(c):
        for direction in (1, -1):
            v = np.zeros(size)
            v[:n] = -direction*2*y[:, j].astype(float)
            v[n+8*c+j] = 1
            constraint(v, -direction*int(y[:, j].sum()))
        objectives[2][n+8*c+j] = 1
    objectives[3][:n] = 1
    objectives[3][original.a] = -1  # omitted constant is len(original.a)
    start, statuses, best = clock(), [], original
    all_optimal = True
    for objective in objectives:
        remaining = seconds-(clock()-start)
        if remaining <= 0:
            all_optimal = False
            break
        result = solver(objective, integrality=integrality, bounds=Bounds(lower, upper),
            constraints=LinearConstraint(sparse.csc_matrix(np.vstack(rows)), np.asarray(lo), np.asarray(hi)),
            options={'time_limit': remaining, 'mip_rel_gap': 0.})
        statuses.append(int(result.status))
        if result.x is not None:
            z = np.asarray(result.x[:n])
            if np.max(np.abs(z-np.rint(z))) < 1e-6:
                candidate = Partition(np.flatnonzero(z > .5), np.flatnonzero(z <= .5))
                if len(candidate.a) == len(original.a) and support_objective(y, candidate, original, quota) < support_objective(y, best, original, quota):
                    best = candidate
        if result.status != 0:
            all_optimal = False
            break
        value = float(np.rint(objective@result.x))
        constraint(objective.copy(), value, value)
    after = support_objective(y, best, original, quota)
    return SupportResult(best, SupportStatus.IMPROVED if after < before else SupportStatus.UNRESOLVED,
                         before, after, all_optimal, tuple(statuses))


def fit_thresholds(raw, y, grid):
    """Vectorized exact replica of the inherited multilabel grid/tie convention."""
    y = validated_target(y)
    raw = np.asarray(raw, float)
    if raw.shape != y.shape or not np.isfinite(raw).all():
        raise ValueError('invalid calibration scores')
    grid = np.array(sorted(set(map(float, grid)), key=lambda t: (abs(t), -t)))
    if not len(grid) or not np.isfinite(grid).all() or 0. not in grid:
        raise ValueError('finite decision grid including zero required')
    p = raw[None, :, :] >= grid[:, None, None]
    tp = (p & y).sum(axis=1)
    den = p.sum(axis=1)+y.sum(axis=0)
    f1 = np.divide(2.*tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
    result = grid[np.argmax(f1, axis=0)].astype(np.float32)
    result[y.sum(axis=0) == 0] = 0.
    return result


def decisions(raw, partition, tau_a, tau_b, alpha, cross):
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or len(raw) != len(partition.a)+len(partition.b) or not 0 <= alpha <= 1:
        raise ValueError('invalid decision request')
    c = raw.shape[1]
    if np.shape(tau_a) != (c,) or np.shape(tau_b) != (c,) or not np.isfinite(raw).all():
        raise ValueError('invalid threshold dimensions')
    # Match persisted float32 threshold multiplication before comparison.
    ta = (alpha*np.asarray(tau_a, float)).astype(np.float32)
    tb = (alpha*np.asarray(tau_b, float)).astype(np.float32)
    if cross:
        pred = np.empty(raw.shape, bool)
        pred[partition.a] = raw[partition.a] >= tb
        pred[partition.b] = raw[partition.b] >= ta
        return pred
    return raw[partition.b] >= ta


def decision_series(design, raw, pred):
    """Original probability loss with explicit (possibly OOF) class decisions."""
    y = validated_target(design.target)
    raw, pred = np.asarray(raw, float), np.asarray(pred, bool)
    if design.task != 'multilabel' or raw.shape != y.shape or pred.shape != y.shape:
        raise ValueError('multilabel decision/design mismatch')
    counts = (sparse.csr_matrix(pred, dtype=float).T@design.weights).T
    tp = (sparse.csr_matrix(pred & y, dtype=float).T@design.weights).T
    den = design.truth_counts+counts
    f1 = np.divide(2*tp, den, out=np.zeros_like(den), where=den > 0)
    mass = design.truth_counts.sum(axis=1)
    weighted = np.divide((f1*design.truth_counts).sum(axis=1), mass, out=np.zeros_like(mass), where=mass > 0)
    p = np.clip(probabilities(raw, 'multilabel'), 1e-7, 1-1e-7)
    row_loss = -(y*np.log(p)+(1-y)*np.log1p(-p)).mean(axis=1)
    return QualitySeries(design.weights.T@row_loss/len(y), f1.mean(axis=1), weighted)


def quality_from_series(series, index=0):
    return {k: float(getattr(series, k)[index]) for k in ('loss', 'f1_macro', 'f1_weighted')}


def point_quality(y, raw, pred):
    y = validated_target(y)
    f1 = label_counts(y, pred)[3]
    support = y.sum(axis=0)
    p = np.clip(probabilities(raw, 'multilabel'), 1e-7, 1-1e-7)
    return {'loss': float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
            'f1_macro': float(f1.mean()),
            'f1_weighted': float(f1@support/support.sum()) if support.sum() else 0.}


def label_counts(y, pred):
    y, pred = validated_target(y), np.asarray(pred, bool)
    if pred.shape != y.shape:
        raise ValueError('label-count dimensions differ')
    tp, fp, fn = (y & pred).sum(axis=0), (~y & pred).sum(axis=0), (y & ~pred).sum(axis=0)
    den = 2*tp+fp+fn
    f1 = np.divide(2.*tp, den, out=np.zeros_like(den, float), where=den > 0)
    return tp, fp, fn, f1
