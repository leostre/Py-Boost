"""Train-only structural compression; split thresholds never change here."""
from dataclasses import replace
import numpy as np
from abl_budget_revision_core import DenseView, decode_model, pack, blank


def keep_trees(model, indices):
    view, _ = decode_model(model)
    ids = np.asarray(indices, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= view.trees:
        raise ValueError('nonempty distinct in-range tree indices required')
    nodes, values, no, lo = [], [], [0], [0]
    for t in ids:
        a, b = view.node_offsets[t:t+2]; u, v = view.leaf_offsets[t:t+2]
        nodes.append(view.nodes[a:b]); values.append(view.values[u:v])
        no.append(no[-1]+b-a); lo.append(lo[-1]+v-u)
    result = DenseView(np.concatenate(nodes), np.asarray(no, np.int64), np.asarray(lo, np.int64),
                       np.concatenate(values), view.bias, view.features, view.task)
    return pack([blank(result.base())], view.bias)


def newton_values(values, gradient, diagonal, alpha, penalty=1.):
    values, gradient, diagonal = (np.asarray(a, dtype=float) for a in (values, gradient, diagonal))
    if values.shape != gradient.shape or values.shape != diagonal.shape or not 0 <= alpha <= 1 or penalty <= 0:
        raise ValueError('invalid leaf update')
    return (values-alpha*np.clip(gradient/(diagonal+penalty), -2., 2.)).astype(np.float32)


def importance_gpu(model, x, y):
    import cupy as cp
    from six_common import gpu_raw
    view, _ = decode_model(model); ids = view.base().leaf_indices(x)
    raw = cp.asarray(gpu_raw(model, x)); yy = cp.asarray(y, dtype=cp.float64)
    def loss(a): return cp.mean(cp.logaddexp(0., a)-yy*a)
    baseline = float(loss(raw).item()); scores = []; coefficients = cp.asarray(view.values)
    for t in range(view.trees):
        z = coefficients[cp.asarray(ids[t])]
        scores.append(float(loss(raw-z).item())-baseline)
    return np.asarray(scores), ids, baseline


def refit_gpu(model, x, y, teacher, passes=2):
    """Block-coordinate BCE Newton refit with train-only teacher targets.

    Every accepted update is rounded to the deployed FP32 leaf values before
    updating the ensemble. Backtracking checks the actual deployed objective.
    """
    import cupy as cp
    view, _ = decode_model(model); base = view.base(); ids = base.leaf_indices(x)
    weights = cp.asarray(base.coefficients); raw = cp.asarray(model.raw(x))
    target = .5*(cp.asarray(y, cp.float64)+1/(1+cp.exp(cp.clip(-cp.asarray(teacher), -80, 80))))
    def objective(a): return float(cp.mean(cp.logaddexp(0., a)-target*a).item())
    current = objective(raw); history = [current]; accepted = 0
    for epoch in range(passes):
        for t in range(view.trees):
            u, v = map(int, view.leaf_offsets[t:t+2]); leaf = cp.asarray(ids[t]-u)
            g = cp.zeros((v-u, len(view.bias)), cp.float64); h = cp.zeros_like(g)
            for start in range(0, len(x), 4096):
                stop = min(start+4096, len(x)); p = 1/(1+cp.exp(cp.clip(-raw[start:stop], -80, 80)))
                membership = (cp.arange(v-u)[:, None] == leaf[start:stop][None, :]).astype(cp.float64)
                g += membership@(p-target[start:stop]); h += membership@(p*(1-p))
            old = weights[u:v].copy(); direction = cp.clip(g/(h+1.), -2., 2.)
            for alpha in (1., .5, .25):
                new = (old.astype(cp.float64)-alpha*direction).astype(cp.float32)
                delta = (new.astype(cp.float64)-old.astype(cp.float64))[leaf]
                trial = objective(raw+delta)
                if trial <= current+1e-12:
                    weights[u:v] = new; raw += delta; current = trial; accepted += 1; break
        history.append(current)
    base = replace(base, coefficients=weights.get()); result = pack([blank(base)], view.bias)
    np.testing.assert_allclose(result.raw(x[:19]), cp.asnumpy(raw[:19]), atol=2e-9, rtol=1e-10)
    assert all(b <= a+1e-12 for a, b in zip(history, history[1:]))
    return result, dict(objective=history, accepted=accepted, passes=passes,
                        targets='0.5 true labels + 0.5 full teacher probabilities; train only')
