"""Paired object bootstrap metrics and validation-only stable minimum selection."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse

from experiments.compactness_core import normalize_multiclass_probabilities
from experiments.minimum_budget_core import memory_key
from experiments.shared_leaf_basis import probabilities, quality_guard
from experiments.sparse_leaf_core import Budget, Cost
from experiments.uniform_threshold_core import family_allows


@dataclass(frozen=True)
class QualitySeries:
    loss: np.ndarray
    f1_macro: np.ndarray
    f1_weighted: np.ndarray

    def __post_init__(self):
        if any(a.ndim != 1 or not np.isfinite(a).all() for a in (self.loss, self.f1_macro, self.f1_weighted)):
            raise ValueError('finite one-dimensional quality series required')
        if not len(self.loss) or len({len(self.loss), len(self.f1_macro), len(self.f1_weighted)}) != 1:
            raise ValueError('quality series dimensions differ')


@dataclass(frozen=True)
class BootstrapDesign:
    task: str
    target: np.ndarray
    weights: np.ndarray  # object x repeat; each column has exactly n objects
    truth_counts: np.ndarray  # repeat x output
    n_classes: int


@dataclass(frozen=True)
class StableMinimum:
    candidate: dict
    successes: int
    checked_count: int
    point_qualified_count: int


@dataclass(frozen=True)
class NoStableMinimum:
    checked_count: int
    point_qualified_count: int


def paired_indices(n_objects, repeats, seed):
    if n_objects < 1 or repeats < 1:
        raise ValueError('positive object and repeat counts required')
    result = np.random.default_rng(seed).integers(0, n_objects, size=(repeats, n_objects), dtype=np.int32)
    result.setflags(write=False)
    return result


def bootstrap_design(target, task, n_classes, indices):
    y, indices = np.asarray(target), np.asarray(indices)
    n = len(y)
    if (n < 1 or indices.ndim != 2 or indices.shape[1] != n or not len(indices)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= n):
        raise ValueError('invalid paired object indices')
    if task == 'multiclass':
        if y.shape != (n,) or not np.equal(y, y.astype(int)).all() or y.min() < 0 or y.max() >= n_classes:
            raise ValueError('invalid multiclass target')
        truth = sparse.csr_matrix((np.ones(n), (np.arange(n), y.astype(int))), shape=(n, n_classes))
    elif task == 'multilabel':
        if y.shape != (n, n_classes) or not np.isin(y, [0, 1]).all():
            raise ValueError('invalid multilabel target')
        truth = sparse.csr_matrix(y, dtype=np.float64)
    else:
        raise ValueError('unsupported task')
    repeats = len(indices)
    flat = indices.astype(np.int64)+(np.arange(repeats)*n)[:, None]
    weights = np.bincount(flat.ravel(), minlength=repeats*n).reshape(repeats, n).T.copy().astype(float)
    np.testing.assert_array_equal(weights.sum(axis=0), np.full(repeats, n))
    weights.setflags(write=False)
    owned_target = y.copy()
    owned_target.setflags(write=False)
    counts = (truth.T@weights).T.copy()
    counts.setflags(write=False)
    return BootstrapDesign(task, owned_target, weights, counts, n_classes)


def bootstrap_quality(design, raw, thresholds):
    """Exact weighted confusion counts; same metric conventions as old studies."""
    raw, thresholds = np.asarray(raw, float), np.asarray(thresholds, float)
    n, c = len(design.target), design.n_classes
    if raw.shape != (n, c) or thresholds.shape != (c,) or not np.isfinite(raw).all() or not np.isfinite(thresholds).all():
        raise ValueError('invalid frozen prediction dimensions')
    p = probabilities(raw, design.task)
    y = design.target
    if design.task == 'multiclass':
        predicted = (raw-thresholds).argmax(axis=1)
        pred = sparse.csr_matrix((np.ones(n), (np.arange(n), predicted)), shape=(n, c))
        matching = predicted == y
        tp = sparse.csr_matrix((matching.astype(float), (np.arange(n), predicted)), shape=(n, c))
        p = np.clip(normalize_multiclass_probabilities(p), 1e-7, 1-1e-7)
        row_loss = -np.log(p[np.arange(n), y.astype(int)])
    else:
        predicted = raw-thresholds >= 0
        pred = sparse.csr_matrix(predicted, dtype=np.float64)
        tp = sparse.csr_matrix(predicted & y.astype(bool), dtype=np.float64)
        p = np.clip(p, 1e-7, 1-1e-7)
        row_loss = -(y*np.log(p)+(1-y)*np.log1p(-p)).mean(axis=1)
    predictions = (pred.T@design.weights).T
    true_positive = (tp.T@design.weights).T
    denominator = design.truth_counts+predictions
    f1 = np.divide(2*true_positive, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    if design.task == 'multiclass':
        active = denominator > 0  # sklearn's unchanged union of true/predicted classes
        macro = f1.sum(axis=1)/active.sum(axis=1)
    else:
        macro = f1.mean(axis=1)
    mass = design.truth_counts.sum(axis=1)
    weighted = np.divide((f1*design.truth_counts).sum(axis=1), mass, out=np.zeros_like(mass), where=mass > 0)
    return QualitySeries(design.weights.T@row_loss/n, macro, weighted)


def paired_pass_mask(candidate, reference):
    if len(candidate.loss) != len(reference.loss):
        raise ValueError('paired repeat counts differ')
    return ((candidate.loss <= 1.05*reference.loss)
            & (candidate.f1_macro >= .95*reference.f1_macro)
            & (candidate.f1_weighted >= .95*reference.f1_weighted))


def eligible_rows(rows, reference, reference_parameters, family, budget):
    return sorted((r for r in rows if family_allows(r, family, reference_parameters)
        and budget.allows(Cost(r['parameters'], r['payload_bytes']))
        and quality_guard(r['validation'], reference)), key=memory_key)


def stable_minimum(rows, reference, reference_parameters, family, budget, successes,
                   repeats=500, minimum_successes=450):
    """Missing cheaper evidence is an error, never an implicit failed check."""
    if not isinstance(budget, Budget) or not 1 <= minimum_successes <= repeats:
        raise ValueError('invalid stable selection request')
    candidates = eligible_rows(rows, reference, reference_parameters, family, budget)
    checked = 0
    for row in candidates:
        count = successes.get(row['candidate_id'])
        if count is None:
            raise ValueError('unscored cheaper candidate: '+row['candidate_id'])
        if not isinstance(count, (int, np.integer)) or not 0 <= count <= repeats:
            raise ValueError('invalid bootstrap success count')
        checked += 1
        if count >= minimum_successes:
            return StableMinimum(row, int(count), checked, len(candidates))
    return NoStableMinimum(checked, len(candidates))


def stability_requests(rows, reference, reference_parameters, scopes):
    """Ordered schedule shared by all caps/families; no Pareto pruning."""
    membership = {}
    by_id = {r['candidate_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('candidate IDs must be unique')
    for i, scope in enumerate(scopes):
        for row in eligible_rows(rows, reference, reference_parameters, scope['family'], Budget(**scope['budget'])):
            membership.setdefault(row['candidate_id'], []).append(i)
    return [(by_id[cid], tuple(membership[cid])) for cid in sorted(membership, key=lambda cid: memory_key(by_id[cid]))]
