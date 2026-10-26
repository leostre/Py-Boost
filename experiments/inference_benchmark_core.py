"""Frozen-artifact planning and paired descriptive timing, without model fitting."""
import numpy as np

ROLES = ('full', 'rank', 'hybrid')


def frozen_choices(record):
    result = {'full': record['full']}
    for role, family in (('rank', 'rank_only'), ('hybrid', 'sparse_05')):
        matches = [r for r in record['selections'] if r['family'] == family
                   and r['budget']['kind'] == 'joint' and r['budget']['fraction'] == .75]
        if len(matches) != 1:
            raise ValueError('expected exactly one frozen choice per family')
        result[role] = matches[0]
    return result


def batch_sizes(n):
    if n < 1:
        raise ValueError('empty test inputs')
    return sorted({1, min(32, n), min(256, n), n})


def timing_summary(samples, rows):
    a = np.asarray(samples, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or np.any(a <= 0) or rows < 1:
        raise ValueError('invalid timing samples')
    return {'samples_ms': a.tolist(), 'median_ms': float(np.median(a)),
            'q25_ms': float(np.quantile(a, .25)), 'q75_ms': float(np.quantile(a, .75)),
            'min_ms': float(a.min()), 'max_ms': float(a.max()),
            'rows_per_second': float(rows*1000/np.median(a))}


def paired_ratios(full, compact):
    a, b = np.asarray(full), np.asarray(compact)
    if (a.shape != b.shape or a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0)):
        raise ValueError('invalid paired timing arrays')
    return float(np.median(b/a))
