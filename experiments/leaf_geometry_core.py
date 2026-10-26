"""Gauge-invariant leaf geometries: separate per-tree means from variation."""
import numpy as np


def tree_centered_leaf_basis(values, offsets, frequency, task, *, visitation_weighted=False):
    v = np.asarray(values, float)
    offsets = np.asarray(offsets)
    f = np.asarray(frequency, float)
    if (v.ndim != 2 or min(v.shape) < 2 or f.shape != (len(v),) or not np.isfinite(v).all()
            or not np.isfinite(f).all() or np.any(f < 0) or offsets.ndim != 1
            or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(v) or np.any(np.diff(offsets) <= 0)
            or not np.issubdtype(offsets.dtype, np.integer)):
        raise ValueError('invalid leaf geometry')
    centered = v.copy()
    means = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if not np.isclose(f[lo:hi].sum(), 1., atol=1e-10, rtol=0):
            raise ValueError('leaf frequencies must sum to one per tree')
        mean = f[lo:hi]@v[lo:hi]
        centered[lo:hi] -= mean
        means.append(mean)
    centered[f == 0] = 0.
    if visitation_weighted:
        centered *= np.sqrt(f)[:, None]
    c = v.shape[1]
    if task == 'multiclass':
        contrast = np.linalg.qr(np.vstack((np.eye(c-1), -np.ones(c-1))))[0]
    elif task == 'multilabel':
        contrast = np.eye(c)
    else:
        raise ValueError('invalid leaf geometry task')
    _, singular, vt = np.linalg.svd(centered@contrast, full_matrices=False)
    return np.sum(means, axis=0), contrast@vt.T, singular
