"""Pure contracts for exact quotas, empirical multiple checks and split utility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuotaActions:
    budgets: np.ndarray
    forced: np.ndarray


def quota_actions(probabilities, train_prior, low_budget, full_budget, low_count):
    probabilities = np.asarray(probabilities, dtype=float)
    train_prior = np.asarray(train_prior, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or train_prior.shape != (3,):
        raise ValueError('three measured risk probabilities required')
    if (not np.isfinite(probabilities).all() or not np.isfinite(train_prior).all() or
            np.any((probabilities < 0) | (probabilities > 1)) or
            np.any((train_prior < 0) | (train_prior > 1))):
        raise ValueError('finite probabilities in [0,1] required')
    if low_budget not in (25, 50, 75) or full_budget <= low_budget:
        raise ValueError('invalid measured budget')
    if isinstance(low_count, bool) or not isinstance(low_count, (int, np.integer)) or not 0 <= low_count <= len(probabilities):
        raise ValueError('quota must be an integer within the planned horizon')
    column = (25, 50, 75).index(low_budget)
    low_left, full_left = low_count, len(probabilities) - low_count
    budgets, forced = [], []
    for current in probabilities:
        force = low_left == 0 or full_left == 0
        # Fixed training-only centering is an opportunity-cost proxy. No test
        # suffix score, future label, monotone-risk assumption or sorting.
        low = low_left > 0 and (full_left == 0 or current[column] < train_prior[column])
        budgets.append(low_budget if low else full_budget)
        forced.append(force)
        low_left -= int(low)
        full_left -= int(not low)
    return QuotaActions(np.asarray(budgets, dtype=int), np.asarray(forced, dtype=bool))


def empirical_checks(observed, nulls, alpha=.05, rtol=1e-6, atol=1e-7):
    observed, nulls = np.asarray(observed, dtype=float), np.asarray(nulls, dtype=float)
    if observed.ndim != 1 or nulls.ndim != 2 or nulls.shape[1] != len(observed) or not len(nulls):
        raise ValueError('matched spectra required')
    if not np.isfinite(observed).all() or not np.isfinite(nulls).all() or not 0 < alpha < 1:
        raise ValueError('finite spectra and alpha in (0,1) required')
    threshold = np.quantile(nulls, 1 - alpha, axis=0)
    tolerance = atol + rtol * np.maximum(abs(observed), abs(threshold))
    rankwise = np.flatnonzero(observed > threshold + tolerance)
    # Finite permutation p-values; ties conservatively count as exceedances.
    p = (1 + (nulls >= observed[None, :] - tolerance[None, :]).sum(axis=0)) / (len(nulls) + 1)
    order = np.argsort(p, kind='stable')
    holm = []
    for i, index in enumerate(order):
        if p[index] > alpha / (len(p) - i):
            break
        holm.append(int(index))
    # Symmetric pooled studentization (includes observed), not an asymmetric
    # post-hoc null fit. Exploratory single-window maxT, assumptions unproven
    # for real optimization trajectories; no across-window guarantee.
    pool = np.vstack((observed, nulls))
    location = pool.mean(axis=0)
    floor = atol + rtol * np.max(abs(pool), axis=0)
    scale = np.maximum(pool.std(axis=0), floor)
    z = (pool - location) / scale
    null_max = z[1:].max(axis=1)
    adjusted = (1 + (null_max[:, None] >= z[0][None, :] - 1e-12).sum(axis=0)) / (len(nulls) + 1)
    return {'rankwise': rankwise.tolist(), 'holm': sorted(holm),
            'maxT': np.flatnonzero(adjusted <= alpha).tolist(),
            'p_rankwise': p.tolist(), 'p_maxT': adjusted.tolist(),
            'observed_z': z[0].tolist(), 'null_maxT': null_max.tolist(),
            'pooled_location': location.tolist(), 'pooled_scale': scale.tolist()}


def split_sums(G, H, masks):
    G, H = np.asarray(G, float), np.broadcast_to(H, np.shape(G)).astype(float)
    return {'left_g': np.stack([G[mask].sum(axis=0) for mask in masks]),
            'left_h': np.stack([H[mask].sum(axis=0) for mask in masks]),
            'right_g': np.stack([G[~mask].sum(axis=0) for mask in masks]),
            'right_h': np.stack([H[~mask].sum(axis=0) for mask in masks])}


def gains_from_sums(sums, l2):
    gl, hl, gr, hr = (np.asarray(sums[key], float) for key in ('left_g', 'left_h', 'right_g', 'right_h'))
    l2 = np.asarray(l2, float)
    return .5 * (gl ** 2 / (hl + l2) + gr ** 2 / (hr + l2) - (gl + gr) ** 2 / (hl + hr + l2))


def aggregate_duplicate_sums(sums, groups, l2, weighted):
    result = {}
    for key, value in sums.items():
        result[key] = np.stack([value[:, group].sum(axis=1) if weighted else value[:, group[0]]
                                for group in groups], axis=1)
    penalties = np.array([len(group) * l2 if weighted else l2 for group in groups])
    return result, penalties


def partition_residual(value, masks):
    """Remove all four joint-cell means, so nuisance cannot prefer either split."""
    value = np.asarray(value, float).copy()
    rare, common = masks
    for r in (False, True):
        for c in (False, True):
            selected = (rare == r) & (common == c)
            if not selected.any():
                raise ValueError('joint partition cells must be nonempty')
            value[selected] -= value[selected].mean(axis=0)
    return value


def rare_split_control(protocol):
    rng = np.random.default_rng(protocol['seed'])
    n, m, epochs = (protocol[key] for key in ('functional_rows', 'functional_outputs', 'functional_epochs'))
    rare = np.arange(n) < max(2, 2 * int(.01 * n))
    common = np.arange(n) % 2 == 0
    masks = (rare, common)
    G = np.empty((epochs, n, m), dtype=np.float32)
    for t in range(epochs):
        raw = partition_residual(rng.normal(0, 10, size=(n, m)), masks)
        raw[:, 0] = 12 * (rare - rare.mean())
        raw[:, 1] = 2 * raw[:, 1] + (2 * common - 1)
        raw[:, 2:32] = raw[:, 2:3]  # correlated nuisance, zero split contributions
        G[t] = raw
    return G, np.ones_like(G), masks


def duplicate_split_control(protocol):
    rng = np.random.default_rng(protocol['seed'] + 1)
    n, m = protocol['functional_rows'], protocol['functional_outputs']
    rare = np.arange(n) < max(2, 2 * int(.01 * n))
    common = np.arange(n) % 2 == 0
    masks = (rare, common)
    G = partition_residual(rng.normal(0, 1, size=(n, m)), masks)
    G[:, :16] = (2 * common - 1)[:, None]
    G[:, 16] = 12 * (rare - rare.mean())
    groups = [list(range(16))] + [[j] for j in range(16, m)]
    return G, np.ones_like(G), masks, groups
