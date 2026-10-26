"""Deterministic output-only calibration plans. No models, files or test labels."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize_scalar

from budget_quality_core import probabilities, metrics_from_sufficient


class Scheme(str, Enum):
    SEPARATE = 'separate'
    POOLED = 'pooled'
    CROSSFIT = 'crossfit'


@dataclass(frozen=True)
class Fold:
    temperature: tuple[int, ...]
    shifts: tuple[int, ...]
    validation: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationPlan:
    scheme: Scheme
    repeat: int
    fraction: float
    folds: tuple[Fold, ...]
    refit_temperature: tuple[int, ...]
    refit_shifts: tuple[int, ...]


@dataclass(frozen=True)
class Policy:
    name: str
    temperature: bool
    shift_fraction: float


@dataclass(frozen=True)
class Control:
    temperature: float
    shifts: tuple[float, ...]


POLICIES = (Policy('raw', False, 0.), Policy('temperature', True, 0.),
            *(Policy(f'raw_shift_{s:g}', False, s) for s in (.25, .5, 1.)),
            *(Policy(f'temperature_shift_{s:g}', True, s) for s in (.25, .5, 1.)))
POLICY_BY_NAME = {p.name: p for p in POLICIES}


def class_labels(target, classes=None):
    source = np.asarray(target)
    if source.ndim != 1 or not len(source) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError('nonempty integer labels required')
    y = source.astype(np.int64, copy=False)
    if np.any(y < 0) or (classes is not None and np.any(y >= classes)):
        raise ValueError('labels outside explicit class domain')
    return y


def stratified_order(target, seed):
    """Exact permutation with approximately proportional class support in prefixes.

    A singleton has one random position: no impossible all-fold support promise.
    All randomness is local and fully determined by the explicit seed.
    """
    y = class_labels(target)
    rng = np.random.default_rng(seed)
    rows, positions = [], []
    for label in np.unique(y):
        group = rng.permutation(np.flatnonzero(y == label))
        rows.extend(group.tolist())
        positions.extend(((np.arange(len(group)) + rng.random(len(group))) / len(group)).tolist())
    return np.asarray(rows, np.int64)[np.argsort(positions, kind='stable')]


def partition(target, sizes, seed):
    if not sizes or any(int(n) != n or n < 1 for n in sizes) or sum(sizes) != len(target):
        raise ValueError('positive exact partition sizes required')
    order = stratified_order(target, seed)
    return tuple(tuple(a.tolist()) for a in np.split(order, np.cumsum(sizes)[:-1]))


def subset(rows, y, fraction, seed):
    if not 0 < fraction <= 1 or not len(rows):
        raise ValueError('nonempty fitting subset and fraction in (0,1] required')
    # At 100% preserve original row order for exact historical replay.
    if fraction == 1.:
        return tuple(map(int, rows))
    a = np.asarray(rows, np.int64)
    count = max(1, int(np.floor(len(a)*fraction)))
    return tuple(a[stratified_order(y[a], seed)[:count]].tolist())


def validate_plan(plan, n):
    validation = []
    for fold in plan.folds:
        for group in (fold.temperature, fold.shifts, fold.validation):
            if not group or len(set(group)) != len(group) or min(group) < 0 or max(group) >= n:
                raise ValueError('invalid or duplicate fold indices')
        if set(fold.validation) & (set(fold.temperature) | set(fold.shifts)):
            raise ValueError('calibration-validation leakage')
        validation.extend(fold.validation)
    if len(set(validation)) != len(validation):
        raise ValueError('validation folds overlap')
    if plan.scheme == Scheme.CROSSFIT and set(validation) != set(range(n)):
        raise ValueError('cross-fitting must cover every development row once')
    for group in (plan.refit_temperature, plan.refit_shifts):
        if not group or len(group) != len(set(group)) or min(group) < 0 or max(group) >= n:
            raise ValueError('invalid refit indices')
    return plan


def make_plans(target, sizes, repeat, fractions=(.25, .5, 1.), seed=20261023):
    y = class_labels(target)
    if len(sizes) != 3 or sum(sizes) != len(y):
        raise ValueError('three development roles required')
    local_seed = seed + 1009*repeat
    # Repeat zero reproduces historical roles; nine further repeats repartition
    # development rows only. The outer train/test split is never changed.
    roles = (tuple(tuple(a.tolist()) for a in np.split(np.arange(len(y)), np.cumsum(sizes)[:-1]))
             if repeat == 0 else partition(y, sizes, local_seed))
    fold_sizes = [len(y)//3 + int(i < len(y)%3) for i in range(3)]
    heldouts = partition(y, fold_sizes, local_seed+101)
    result = []
    for fraction in fractions:
        cal = subset(roles[0], y, fraction, local_seed+201)
        ctrl = subset(roles[1], y, fraction, local_seed+301)
        pooled = cal+ctrl  # Same fitting objects as separate, not a new sample.
        result.append(CalibrationPlan(Scheme.SEPARATE, repeat, fraction,
            (Fold(cal, ctrl, roles[2]),), cal, ctrl))
        result.append(CalibrationPlan(Scheme.POOLED, repeat, fraction,
            (Fold(pooled, pooled, roles[2]),), pooled, pooled))
        folds = []
        for i, validation in enumerate(heldouts):
            training = tuple(sorted(set(range(len(y)))-set(validation)))
            fitting = subset(training, y, fraction, local_seed+401+i)
            folds.append(Fold(fitting, fitting, validation))
        refit = subset(tuple(range(len(y))), y, fraction, local_seed+501)
        result.append(CalibrationPlan(Scheme.CROSSFIT, repeat, fraction, tuple(folds), refit, refit))
    return tuple(validate_plan(p, len(y)) for p in result)


def fast_temperature(raw, target):
    """Same bounded/clipped objective as the frozen fitter, without unused F1."""
    z = np.asarray(raw, np.float64)
    y = class_labels(target, z.shape[1])
    if z.shape[0] != len(y) or not np.isfinite(z).all():
        raise ValueError('invalid finite fitting logits')
    def loss(t):
        p = probabilities(z, t)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1-1e-7)).mean())
    fitted = minimize_scalar(lambda a: loss(float(np.exp(a))),
        bounds=(np.log(.1), np.log(10.)), method='bounded',
        options={'xatol': 1e-7, 'maxiter': 150})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError('temperature fitting failed')
    values = [float(np.float32(np.clip(t, .1, 10.))) for t in (1., .1, 10., np.exp(fitted.x))]
    return min(values, key=lambda t: (loss(t), abs(np.log(t))))


def decision_shifts(raw, target, grid):
    """Frozen one-vs-rest heuristic; disable an unsupported binary contrast."""
    z = np.asarray(raw, np.float64)
    y = class_labels(target, z.shape[1])
    if z.shape[0] != len(y) or not np.isfinite(z).all():
        raise ValueError('invalid decision fitting logits')
    z = z-z.mean(axis=1, keepdims=True)
    c = z.shape[1]
    best = z.argmax(axis=1)
    competitor = np.broadcast_to(z.max(axis=1)[:, None], z.shape).copy()
    competitor[np.arange(len(z)), best] = np.partition(z, -2, axis=1)[:, -2]
    margin = z-competitor
    thresholds = sorted(set(map(float, grid)), key=lambda t: (abs(t), -t))
    if 0. not in thresholds:
        raise ValueError('decision grid must contain zero')
    # Vectorize over classes, not training rows: same tie-breaking as old code.
    truth = y[:, None] == np.arange(c)
    support = truth.sum(axis=0)
    scores = []
    for threshold in thresholds:
        pred = margin >= threshold
        tp = (pred & truth).sum(axis=0)
        denominator = pred.sum(axis=0)+support
        scores.append(np.divide(2*tp, denominator, out=np.zeros(c), where=denominator > 0))
    shifts = np.asarray(thresholds, np.float32)[np.argmax(scores, axis=0)]
    shifts[(support == 0) | (support == len(y))] = 0
    return shifts


def fit_controls(raw, y, temperature_rows, shift_rows, grid):
    ti, si = np.asarray(temperature_rows), np.asarray(shift_rows)
    t = fast_temperature(raw[ti], y[ti])
    return (t, decision_shifts(raw[si], y[si], grid),
            decision_shifts(raw[si]/t, y[si], grid))


def control_for(policy, fitted):
    t, raw_shifts, calibrated_shifts = fitted
    shifts = calibrated_shifts if policy.temperature else raw_shifts
    return Control(t if policy.temperature else 1.,
                   tuple((shifts*policy.shift_fraction).astype(np.float32).tolist()))


def evaluate_plan(raw, target, plan, grid):
    """Fit controls on each training fold and concatenate honest OOF statistics."""
    z = np.asarray(raw, np.float64)
    y = class_labels(target, z.shape[1])
    validate_plan(plan, len(y))
    collected = {p.name: [] for p in POLICIES}
    diagnostics = []
    for fold in plan.folds:
        fitted = fit_controls(z, y, fold.temperature, fold.shifts, grid)
        ids = np.asarray(fold.validation)
        for p in POLICIES:
            control = control_for(p, fitted)
            prob = probabilities(z[ids], control.temperature)
            yy = y[ids]
            pred = (z[ids]/control.temperature-control.shifts).argmax(axis=1)
            loss = -np.log(np.clip(prob[np.arange(len(yy)), yy], 1e-7, 1-1e-7))
            brier = np.sum(prob*prob, axis=1)-2*prob[np.arange(len(yy)), yy]+1
            collected[p.name].append((yy, pred, loss, brier))
        support = np.bincount(y[list(fold.shifts)], minlength=z.shape[1])
        diagnostics.append(dict(temperature=fitted[0], temperature_rows=len(fold.temperature),
            shifts_rows=len(fold.shifts), validation_rows=len(ids), support=support.tolist(),
            unsupported_classes=np.flatnonzero((support == 0) | (support == len(fold.shifts))).tolist()))
    scores = {}
    for p in POLICIES:
        parts = collected[p.name]
        # Macro-F1 is non-additive: never average fold-wise macro-F1.
        args = [np.concatenate([part[i] for part in parts]) for i in range(4)]
        scores[p.name] = metrics_from_sufficient(*args, z.shape[1])
    return scores, diagnostics


def policy_key(name, score, objective):
    p = POLICY_BY_NAME[name]
    tail = (int(p.temperature)+int(p.shift_fraction > 0), p.shift_fraction, name)
    if objective == 'loss':
        return (score['loss'], *tail)
    if objective == 'f1_macro':
        return (-score['f1_macro'], score['loss'], *tail)
    raise ValueError('unknown objective')


def choose_policy(scores, objective):
    allowed = ('raw', 'temperature') if objective == 'loss' else tuple(scores)
    return min(allowed, key=lambda name: policy_key(name, scores[name], objective))


def refit_control(raw, target, plan, policy_name, grid):
    # Selection sees only fold scores. Refit is performed after choosing policy.
    p = POLICY_BY_NAME[policy_name]
    if p.name == 'raw':
        return Control(1., (0.,)*raw.shape[1])
    fitted = fit_controls(raw, target, plan.refit_temperature, plan.refit_shifts, grid)
    return control_for(p, fitted)


def choose_model(rows, objective, budget):
    feasible = [r for r in rows if r['cost_bytes'] <= budget]
    def rank(r):
        score = r['validation']
        if objective == 'loss':
            return (score['loss'], r['cost_bytes'], r['model_id'])
        if objective == 'f1_macro':
            return (-score['f1_macro'], score['loss'], r['cost_bytes'], r['model_id'])
        raise ValueError('unknown objective')
    return min(feasible, key=rank) if feasible else None
