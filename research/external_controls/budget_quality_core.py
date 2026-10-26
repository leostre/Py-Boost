"""Pure stage-1 calibration, metrics and selection; no test-aware fitting."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def probabilities(raw, temperature=1.):
    z = np.asarray(raw, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('finite NxC logits required')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    z = z / temperature
    e = np.exp(z-z.max(axis=1, keepdims=True))
    # Same inherited two-stage clipping/normalization as LeafEnsemble + quality.
    p = np.clip(e/e.sum(axis=1, keepdims=True), 1e-6, 1-1e-6)
    return p/p.sum(axis=1, keepdims=True)


def sufficient(target, raw, temperature=1., shifts=None):
    y = np.asarray(target, dtype=np.int64)
    p = probabilities(raw, temperature)
    c = p.shape[1]
    if y.shape != (len(p),) or not len(y) or np.any((y < 0) | (y >= c)):
        raise ValueError('target outside explicit class domain')
    shift = np.zeros(c) if shifts is None else np.asarray(shifts)
    if shift.shape != (c,) or not np.isfinite(shift).all():
        raise ValueError('invalid decision shifts')
    pred = (np.asarray(raw, np.float64)/temperature-shift).argmax(axis=1)
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1-1e-7))
    brier = np.sum(p*p, axis=1)-2*p[np.arange(len(y)), y]+1
    return y, pred, loss, brier, c


def metrics_from_sufficient(y, pred, loss, brier, c, weights=None):
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    support = np.bincount(y, weights=w, minlength=c)
    predicted = np.bincount(pred, weights=w, minlength=c)
    tp = np.bincount(y, weights=w*(y == pred), minlength=c)
    denominator = support+predicted
    f1 = np.divide(2*tp, denominator, out=np.zeros(c), where=denominator > 0)
    return dict(loss=float(np.dot(w, loss)/w.sum()),
                f1_macro=float(f1.mean()), f1_weighted=float(np.dot(support, f1)/w.sum()),
                brier=float(np.dot(w, brier)/w.sum()))


def quality(target, raw, temperature=1., shifts=None):
    return metrics_from_sufficient(*sufficient(target, raw, temperature, shifts))


def fit_temperature(raw, target):
    objective = lambda log_t: quality(target, raw, float(np.exp(log_t)))['loss']
    fit = minimize_scalar(objective, bounds=(np.log(.1), np.log(10.)),
                          method='bounded', options=dict(xatol=1e-7, maxiter=150))
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError('bounded scalar temperature fit failed')
    candidates = [1., .1, 10., float(np.exp(fit.x))]
    # Storage and applied value agree: exactly one persistent float32 scalar.
    candidates = [float(np.float32(np.clip(t, .1, 10.))) for t in candidates]
    t = min(candidates, key=lambda t: (quality(target, raw, t)['loss'], abs(np.log(t))))
    return dict(temperature=t, calibration_loss=quality(target, raw, t)['loss'],
                identity_loss=quality(target, raw)['loss'], converged=True, evaluations=int(fit.nfev))


def variants(logits, targets, fit_shifts, grid, shrinkages):
    fitted = fit_temperature(logits['calibration'], targets['calibration'])
    c = logits['selection'].shape[1]
    result = []
    for calibrated, t in ((False, 1.), (True, fitted['temperature'])):
        fitted_shifts = fit_shifts(logits['control']/t, targets['control'], 'multiclass', grid)
        for shifted in (False, True):
            for fraction in (sorted(set([0., *shrinkages])) if shifted else [0.]):
                shifts = (fraction*fitted_shifts).astype(np.float32)
                result.append(dict(policy=('temperature' if calibrated else 'raw')+('_shifts' if shifted else ''),
                    temperature=t, temperature_fit=fitted if calibrated else None,
                    shift_fraction=float(fraction), decision_shifts=shifts.tolist(),
                    calibration_bytes=4+4*c,
                    selection=quality(targets['selection'], logits['selection'], t, shifts)))
    raw = next(r for r in result if r['policy'] == 'raw')
    temp = next(r for r in result if r['policy'] == 'temperature')
    for metric in ('f1_macro', 'f1_weighted'):
        if raw['selection'][metric] != temp['selection'][metric]:
            raise RuntimeError('positive scalar temperature changed argmax')
    return result


def ranking(row, objective):
    q = row['selection']
    if objective == 'loss':
        return (q['loss'], -q['f1_macro'], row['cost_bytes'], row['uid'])
    if objective == 'f1_macro':
        return (-q['f1_macro'], q['loss'], row['cost_bytes'], row['uid'])
    raise ValueError('unknown objective')


def select(rows, objective, budget=None):
    candidates = [r for r in rows if budget is None or r['cost_bytes'] <= budget]
    return min(candidates, key=lambda r: ranking(r, objective)) if candidates else None


def paired_bootstrap(target, left, right, repeats=500, seed=20261022):
    a = sufficient(target, *left)
    b = sufficient(target, *right)
    if a[-1] != b[-1] or not np.array_equal(a[0], b[0]):
        raise ValueError('paired targets or domains differ')
    rng = np.random.default_rng(seed)
    deltas = []
    # Same resampled rows in both arms; fixed full class domain in every repeat.
    for _ in range(repeats):
        counts = rng.multinomial(len(target), np.full(len(target), 1/len(target)))
        qa = metrics_from_sufficient(*a, counts)
        qb = metrics_from_sufficient(*b, counts)
        deltas.append([qa['loss']-qb['loss'], 100*(qa['f1_macro']-qb['f1_macro'])])
    intervals = np.quantile(deltas, [.025, .975], axis=0).T
    qa, qb = metrics_from_sufficient(*a), metrics_from_sufficient(*b)
    return dict(repeats=repeats, seed=seed, class_domain=list(range(a[-1])),
        support=np.bincount(a[0], minlength=a[-1]).tolist(),
        delta_loss=qa['loss']-qb['loss'], delta_f1_pp=100*(qa['f1_macro']-qb['f1_macro']),
        delta_loss_ci95=intervals[0].tolist(), delta_f1_pp_ci95=intervals[1].tolist(),
        scope='conditional fixed-model single-split paired exploratory interval; not seed uncertainty')
