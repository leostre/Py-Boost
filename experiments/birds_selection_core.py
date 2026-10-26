"""Pure, paired diagnostics for selection among immutable multilabel models."""
from dataclasses import dataclass

import numpy as np

from experiments.shared_leaf_basis import probabilities


METRICS = ('loss', 'f1_macro', 'f1_weighted')
POOLS = ('all_all', 'all_fixed', 'recent_all', 'recent_fixed')


@dataclass(frozen=True)
class Contributions:
    loss: np.ndarray                 # models x observations
    tp: np.ndarray                   # models x observations x labels
    fp: np.ndarray
    fn: np.ndarray


def contributions(target, raw, shifts):
    """Freeze per-observation sufficient statistics, never fit thresholds."""
    y, z, thresholds = np.asarray(target), np.asarray(raw), np.asarray(shifts)
    if (y.ndim != 2 or z.ndim != 3 or z.shape[1:] != y.shape
            or thresholds.shape != (z.shape[0], z.shape[2]) or not len(y)
            or not z.shape[0] or y.shape[1] < 2 or not np.isin(y, (0, 1)).all()
            or not np.isfinite(z).all() or not np.isfinite(thresholds).all()):
        raise ValueError('invalid frozen multilabel contributions')
    truth = y.astype(bool)[None, :, :]
    labels = z >= thresholds[:, None, :]
    p = probabilities(z, 'multilabel')
    loss = -(y[None, :, :]*np.log(p)+(1-y[None, :, :])*np.log1p(-p)).mean(axis=2)
    return Contributions(loss, (truth & labels).astype(np.int64),
                         (~truth & labels).astype(np.int64), (truth & ~labels).astype(np.int64))


def weighted_metrics(evidence, weights):
    """Return draws x models x (loss, macro-F1, weighted-F1).

    Integer observation multiplicities preserve ordinary bootstrap semantics.
    Macro-F1 retains the entire fixed multilabel output domain, including zeros.
    """
    w = np.asarray(weights)
    if (w.ndim != 2 or w.shape[1] != evidence.loss.shape[1]
            or not np.isfinite(w).all() or (w < 0).any()
            or not np.equal(w, np.floor(w)).all() or (w.sum(axis=1) <= 0).any()):
        raise ValueError('invalid observation multiplicities')
    w = w.astype(np.float64)
    loss = w @ evidence.loss.T / w.sum(axis=1, keepdims=True)
    counts = [np.einsum('dn,knc->dkc', w, a, optimize=True) for a in
              (evidence.tp, evidence.fp, evidence.fn)]
    tp, fp, fn = counts
    denominator = 2*tp+fp+fn
    f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    support = tp+fn
    weighted = np.divide((f1*support).sum(axis=2), support.sum(axis=2),
                         out=np.zeros_like(loss), where=support.sum(axis=2) > 0)
    return np.stack((loss, f1.mean(axis=2), weighted), axis=2)


def as_quality(values):
    return dict(zip(METRICS, map(float, values)))


def quality_vector(quality):
    return np.array([quality[k] for k in METRICS], dtype=float)


def passes(metrics, reference):
    q, full = np.asarray(metrics), np.asarray(reference)
    return ((q[..., 0] <= full[..., 0]*1.05)
            & (q[..., 1] >= full[..., 1]*.95) & (q[..., 2] >= full[..., 2]*.95))


def pool_indices(rows, pool):
    if pool not in POOLS or not rows or len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('invalid frozen shortlist')
    latest = set(sorted({r['block'] for r in rows})[-2:])
    result = [i for i, row in enumerate(rows)
              if (not pool.startswith('recent') or row['block'] in latest)
              and (not pool.endswith('fixed') or row['threshold_fraction'] == 1.)]
    if not result:
        raise ValueError('frozen shortlist is empty')
    return result


def select(rows, metrics, reference, indices, rates=None):
    """Rates, when supplied, add the preregistered >=90% paired gate."""
    q = np.asarray(metrics)
    if (q.shape != (len(rows), 3) or np.asarray(reference).shape != (3,)
            or not np.isfinite(q).all() or not np.isfinite(reference).all()
            or not indices or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(rows) for i in indices)):
        raise ValueError('invalid frozen selection input')
    if rates is not None and (np.asarray(rates).shape != (len(rows),)
            or not np.isfinite(rates).all() or (np.asarray(rates) < 0).any()
            or (np.asarray(rates) > 1).any()):
        raise ValueError('invalid paired acceptance rates')
    allowed = passes(q, reference)
    feasible = [i for i in indices if allowed[i] and (rates is None or rates[i] >= .9)]
    chosen = min(feasible or indices, key=lambda i:(-q[i, 1], q[i, 0], rows[i]['payload_bytes'],
                 rows[i]['parameters'], rows[i]['candidate_id']))
    return dict(accepted=bool(feasible), candidate_id=rows[chosen]['candidate_id'], index=chosen,
                point_pass=bool(allowed[chosen]), eligible_count=len(feasible),
                bootstrap_pass_rate=None if rates is None else float(rates[chosen]))


def bootstrap_weights(count, repeats, seed):
    if count < 2 or repeats < 1:
        raise ValueError('bootstrap requires at least two observations')
    rng = np.random.default_rng(seed)
    observations = rng.integers(count, size=(repeats, count))
    weights = np.zeros((repeats, count), np.int64)
    np.add.at(weights, (np.arange(repeats)[:, None], observations), 1)
    return weights


def split_half_weights(count, repeats, seed):
    if count < 4 or repeats < 1:
        raise ValueError('internal assessment requires at least four observations')
    rng = np.random.default_rng(seed)
    a, b = np.zeros((repeats, count), np.int64), np.zeros((repeats, count), np.int64)
    for i in range(repeats):
        order = rng.permutation(count)
        a[i, order[:count//2]] = 1
        b[i, order[count//2:]] = 1
    # Each direction selects on its first side and assesses on the disjoint side.
    return np.concatenate((a, b)), np.concatenate((b, a))


def leave_one_out_weights(count):
    if count < 2:
        raise ValueError('leave-one-out requires at least two observations')
    return np.ones((count, count), np.int64)-np.eye(count, dtype=np.int64)
