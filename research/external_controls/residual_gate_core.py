"""Frozen scalar residual gate. Pure CPU functions; no learner or test access."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.special import expit, softmax
from calibration_pilot_core import POLICIES, fit_controls, control_for


ALPHAS = (0., .125, .25, .5, 1.)


class CalibrationMode(str, Enum):
    FROZEN_BASE = 'frozen_base'
    RECALIBRATED = 'recalibrated'


@dataclass(frozen=True)
class GateDecision:
    alpha: float
    index: int
    reason: str
    best_index: int
    candidates: tuple


def blend_raw(base, corrected, alpha):
    base, corrected = np.asarray(base, float), np.asarray(corrected, float)
    if base.ndim != 2 or base.shape != corrected.shape or not np.isfinite(base).all() or not np.isfinite(corrected).all():
        raise ValueError('matching finite two-dimensional scores required')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    if alpha == 0: return base.copy()
    if alpha == 1: return corrected.copy()
    return base + alpha * (corrected - base)


def label_matrix(y, task, outputs):
    y = np.asarray(y)
    if task == 'multilabel':
        if y.ndim != 2 or y.shape[1] != outputs or not np.isin(y, [0, 1]).all():
            raise ValueError('binary multilabel matrix required')
        return y.astype(bool)
    if task != 'multiclass' or y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= outputs)):
        raise ValueError('integer multiclass domain required')
    return y[:, None] == np.arange(outputs)


def predicted_matrix(prediction, task, outputs):
    return label_matrix(prediction, task, outputs)


def sufficient(y, raw, task, policy):
    raw = np.asarray(raw, float); y = np.asarray(y)
    t = float(policy['temperature']); shifts = np.asarray(policy['shifts'], float)
    if raw.ndim != 2 or len(raw) != len(y) or shifts.shape != (raw.shape[1],) or t <= 0 or not np.isfinite(t) or not np.isfinite(shifts).all() or not np.isfinite(raw).all():
        raise ValueError('invalid score/control domain')
    label_matrix(y, task, raw.shape[1])
    if task == 'multilabel':
        if t != 1.: raise ValueError('matched multilabel protocol keeps T=1')
        p = np.clip(expit(raw), 1e-7, 1-1e-7)
        target = y.astype(float)
        return raw-shifts >= 0, np.mean(-target*np.log(p)-(1-target)*np.log1p(-p), axis=1), np.mean((p-target)**2, axis=1)
    p = np.clip(softmax(raw/t, axis=1), 1e-7, 1-1e-7)
    return (raw/t-shifts).argmax(axis=1), -np.log(p[np.arange(len(y)), y]), np.sum(p*p, axis=1)-2*p[np.arange(len(y)), y]+1


def quality(y, prediction, loss, brier, task, outputs):
    truth = label_matrix(y, task, outputs); pred = predicted_matrix(prediction, task, outputs)
    tp = np.sum(truth & pred, axis=0); support = truth.sum(axis=0)
    den = support + pred.sum(axis=0)
    f1 = np.divide(2.*tp, den, out=np.zeros(outputs), where=den > 0)
    return dict(f1_macro=float(f1.mean()), f1_weighted=float(np.sum(f1*support)/max(support.sum(), 1)),
                loss=float(np.mean(loss)), brier=float(np.mean(brier)))


def fit_policies(raw, y, task, grid, shrinkages):
    """Only fitting labels enter; same old grids and unsupported-label handling."""
    raw = np.asarray(raw, float); y = np.asarray(y); c = raw.shape[1]
    truth = label_matrix(y, task, c)
    if task == 'multiclass':
        ids = tuple(range(len(y))); fitted = fit_controls(raw, y, ids, ids, grid)
        return {p.name: dict(temperature=float(control_for(p, fitted).temperature),
                            shifts=list(control_for(p, fitted).shifts)) for p in POLICIES}
    thresholds = sorted(set(map(float, grid)), key=lambda a: (abs(a), -a))
    if 0. not in thresholds: raise ValueError('decision grid must contain zero')
    scores = []
    for t in thresholds:
        pred = raw >= t; den = pred.sum(axis=0)+truth.sum(axis=0)
        scores.append(np.divide(2.*(pred & truth).sum(axis=0), den, out=np.zeros(c), where=den > 0))
    shifts = np.asarray(thresholds, np.float32)[np.argmax(scores, axis=0)]
    shifts[truth.sum(axis=0) == 0] = 0
    fractions = sorted(set((0., *map(float, shrinkages))))
    if any(not 0 <= f <= 1 for f in fractions): raise ValueError('invalid shift fraction')
    return {('raw' if f == 0 else f'raw_shift_{f:g}'): dict(temperature=1., shifts=(shifts*f).astype(np.float32).tolist()) for f in fractions}


def choose_policy(scores, task):
    # Keep the existing calibration objective, including NLL for multiclass.
    return min(scores, key=lambda name: ((scores[name]['loss'],) if task == 'multiclass' else
        (-scores[name]['f1_macro'], scores[name]['loss'])) + (int(name != 'raw'), name))


def bootstrap_scores(y, predictions, losses, task, outputs, repeats, seed):
    """Joint row resampling preserves all labels, candidates and paired losses."""
    predictions = np.asarray(predictions); losses = np.asarray(losses, float)
    n = len(y); k = len(predictions)
    if n < 2 or repeats < 2 or losses.shape != (k, n): raise ValueError('invalid paired bootstrap domain')
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1./n), size=repeats).astype(float)
    truth = label_matrix(y, task, outputs)
    support = counts @ truth.astype(float)
    f1 = np.empty((repeats, k))
    for i, prediction in enumerate(predictions):
        pred = predicted_matrix(prediction, task, outputs)
        tp = counts @ (truth & pred).astype(float); den = support + counts @ pred.astype(float)
        f1[:, i] = np.divide(2*tp, den, out=np.zeros_like(tp), where=den > 0).mean(axis=1)
    return f1, counts @ losses.T / n


def decide(alphas, scores, boot_f1, boot_loss):
    """One-SE benefit + no-clear-NLL-harm; least competitive nonzero amplitude."""
    alphas = tuple(map(float, alphas)); f1 = np.asarray([q['f1_macro'] for q in scores]); loss = np.asarray([q['loss'] for q in scores])
    bf, bl = np.asarray(boot_f1, float), np.asarray(boot_loss, float)
    if tuple(sorted(set(alphas))) != alphas or alphas[0] != 0 or alphas[-1] > 1 or len(scores) != len(alphas) or bf.ndim != 2 or bf.shape != bl.shape or bf.shape[1] != len(alphas) or len(bf) < 2:
        raise ValueError('invalid gate domain')
    if not all(np.isfinite(a).all() for a in (f1, loss, bf, bl)): raise ValueError('finite gate evidence required')
    gf = f1-f1[0]; gn = loss[0]-loss
    sf = np.std(bf-bf[:, :1], axis=0, ddof=1); sn = np.std(bl[:, :1]-bl, axis=0, ddof=1)
    eligible = [i for i in range(1, len(alphas)) if gf[i] > sf[i] and gn[i] >= -sn[i]]
    best = min(eligible, key=lambda i: (-f1[i], loss[i], alphas[i])) if eligible else 0
    gap_se = np.std(bf[:, best:best+1]-bf, axis=0, ddof=1)
    competitive = [i for i in [0, *eligible] if f1[best]-f1[i] <= gap_se[i]+1e-14]
    chosen = min(competitive, key=lambda i: alphas[i]) if eligible else 0
    rows = tuple(dict(alpha=a, f1_macro=float(f1[i]), loss=float(loss[i]), f1_gain=float(gf[i]), nll_gain=float(gn[i]),
        f1_gain_se=float(sf[i]), nll_gain_se=float(sn[i]), admitted=bool(i == 0 or i in eligible),
        competitive=bool(i in competitive), gap_to_best_se=float(gap_se[i]),
        benefit_passed=bool(gf[i] > sf[i]), nll_guard_passed=bool(gn[i] >= -sn[i])) for i, a in enumerate(alphas))
    return GateDecision(alphas[chosen], chosen, 'no_admissible_benefit' if not eligible else 'least_amplitude_within_paired_one_se', best, rows)


def assess(raws, y, task, folds, mode, grid, shrinkages, seed, repeats=500):
    """Inner OOF assessment. No external raw scores or labels are accepted."""
    mode = CalibrationMode(mode); raws = np.asarray(raws, float); n = len(y); c = raws.shape[-1]
    if raws.shape[:2] != (len(ALPHAS), n): raise ValueError('frozen alpha grid/row mismatch')
    seen = []; states = []; arrays = None; names = None
    for fit, held in folds:
        a, b = np.asarray(fit, int), np.asarray(held, int)
        if not len(a) or not len(b) or set(a) & set(b) or len(set(a)) != len(a) or len(set(b)) != len(b) or min(*a, *b) < 0 or max(*a, *b) >= n:
            raise ValueError('invalid/leaking inner fold')
        seen.extend(b.tolist()); controls = [fit_policies(raws[0, a], y[a], task, grid, shrinkages)]
        for i in range(1, len(ALPHAS)):
            controls.append(controls[0] if mode is CalibrationMode.FROZEN_BASE else fit_policies(raws[i, a], y[a], task, grid, shrinkages))
        if arrays is None:
            names = tuple(controls[0]); prediction_shape = (n, c) if task == 'multilabel' else (n,)
            arrays = {name: [np.empty((len(ALPHAS), *prediction_shape), bool if task == 'multilabel' else int),
                            np.empty((len(ALPHAS), n)), np.empty((len(ALPHAS), n))] for name in names}
        for i in range(len(ALPHAS)):
            for name in names:
                values = sufficient(y[b], raws[i, b], task, controls[i][name])
                for dest, value in zip(arrays[name], values): dest[i, b] = value
        states.append(dict(fitting=a.tolist(), validation=b.tolist(), controls=controls))
    if sorted(seen) != list(range(n)): raise ValueError('inner validation must cover each row once')
    all_scores = [{name: quality(y, *(part[i] for part in arrays[name]), task, c) for name in names} for i in range(len(ALPHAS))]
    policies = [choose_policy(all_scores[0 if mode is CalibrationMode.FROZEN_BASE else i], task) for i in range(len(ALPHAS))]
    prediction, loss, brier = [np.stack([arrays[policies[i]][j][i] for i in range(len(ALPHAS))]) for j in range(3)]
    scores = [all_scores[i][policies[i]] for i in range(len(ALPHAS))]
    bf, bl = bootstrap_scores(y, prediction, loss, task, c, repeats, seed)
    decision = decide(ALPHAS, scores, bf, bl)
    return dict(decision=decision, policies=policies, scores=scores, all_scores=all_scores, states=states,
                prediction=prediction, loss=loss, brier=brier, boot_f1=bf, boot_loss=bl)
