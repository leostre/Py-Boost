"""Pure CPU contracts for test-label-free residual selection and storage.

Training labels are allowed for balanced scores. No selector receives validation
or test labels; the separate candidate chooser sees validation metrics only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import numpy as np
from scipy import sparse

from experiments.shared_leaf_basis import LeafEnsemble, probabilities, quality_guard

SELECTORS = ('energy', 'balanced_mse', 'balanced_margin')


def contrast(values, task):
    a = np.asarray(values, dtype=np.float64)
    return a - a.mean(axis=1, keepdims=True) if task == 'multiclass' else a


def leaf_design(leaves, n_leaves):
    leaves = np.asarray(leaves)
    if leaves.ndim != 2 or leaves.size == 0 or leaves.min() < 0 or leaves.max() >= n_leaves:
        raise ValueError('invalid leaf membership')
    n = leaves.shape[1]
    return sparse.coo_matrix((np.ones(leaves.size),
        (np.tile(np.arange(n), len(leaves)), leaves.ravel())), shape=(n, n_leaves)).tocsr()


@dataclass(frozen=True)
class Cost:
    parameters: int
    payload_bytes: int


@dataclass(frozen=True)
class Budget:
    kind: str
    fraction: float
    parameters: int | None
    payload_bytes: int | None

    def allows(self, cost):
        return ((self.parameters is None or cost.parameters <= self.parameters)
                and (self.payload_bytes is None or cost.payload_bytes <= self.payload_bytes))


def budget_for(reference, kind, fraction):
    if kind not in ('parameters', 'memory', 'joint') or not 0 < fraction < 1:
        raise ValueError('invalid budget')
    return Budget(kind, fraction,
        math.floor(reference.parameters*fraction) if kind != 'memory' else None,
        math.floor(reference.payload_bytes*fraction) if kind != 'parameters' else None)


def residual_capacity(base_cost, budget):
    if not budget.allows(base_cost):
        return 0
    capacities = []
    if budget.parameters is not None:
        capacities.append(budget.parameters-base_cost.parameters)
    if budget.payload_bytes is not None:
        capacities.append((budget.payload_bytes-base_cost.payload_bytes)//12)
    return int(min(capacities))  # float32 value, int32 row and output.


@dataclass(frozen=True)
class SparseLeafModel:
    base: LeafEnsemble
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if self.base.representation != 'low_rank':
            raise ValueError('sparse model requires a low-rank base')
        if (self.rows.dtype != np.int32 or self.columns.dtype != np.int32
                or self.values.dtype != np.float32):
            raise ValueError('COO requires int32/int32/float32')
        if any(a.ndim != 1 for a in (self.rows, self.columns, self.values)) or not (
                len(self.rows) == len(self.columns) == len(self.values)):
            raise ValueError('COO dimensions')
        if not np.isfinite(self.values).all() or np.any(self.values == 0):
            raise ValueError('COO values must be finite nonzero values')
        if len(self.rows):
            m, c = len(self.base.coefficients), len(self.base.bias)
            keys = self.rows.astype(np.int64)*c+self.columns
            if (self.rows.min() < 0 or self.rows.max() >= m or self.columns.min() < 0
                    or self.columns.max() >= c or len(np.unique(keys)) != len(keys)):
                raise ValueError('invalid or duplicate COO coordinate')

    @property
    def parameter_count(self):
        return self.base.parameter_count+len(self.values)

    @property
    def payload_bytes(self):
        return self.base.payload_bytes+self.rows.nbytes+self.columns.nbytes+self.values.nbytes

    @property
    def cost(self):
        return Cost(self.parameter_count, self.payload_bytes)

    def residual_matrix(self):
        # Derived inference workspace, not a hidden persistent dense fallback.
        return sparse.coo_matrix((self.values, (self.rows, self.columns)),
            shape=(len(self.base.coefficients), len(self.base.bias))).tocsr()

    def raw_from_design(self, design):
        raw = (design @ self.base.coefficients) @ self.base.basis.T + self.base.bias
        if len(self.values):
            raw += (design @ self.residual_matrix()).toarray()
        return raw

    def predict(self, features):
        design = leaf_design(self.base.leaf_indices(features), len(self.base.coefficients))
        return probabilities(self.raw_from_design(design), self.base.task)

    def save(self, path):
        names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
        np.savez(path, version=np.array(2), **{n:getattr(self.base,n) for n in names},
            n_features=np.array(self.base.n_features), task=np.array(self.base.task),
            rows=self.rows, columns=self.columns, values=self.values)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
            expected = {*names, 'version', 'n_features', 'task', 'rows', 'columns', 'values'}
            if set(data.files) != expected or int(data['version']) != 2:
                raise ValueError('unknown sparse model schema')
            base = LeafEnsemble(**{n:data[n].copy() for n in names},
                n_features=int(data['n_features']), task=str(data['task']), representation='low_rank')
            return cls(base, data['rows'].copy(), data['columns'].copy(), data['values'].copy())


def make_sparse_model(base, residual, order, count, train_frequency):
    if count < 0 or count > len(order):
        raise ValueError('invalid sparse size')
    m, c = residual.shape
    flat = np.asarray(order[:count], dtype=np.int64)
    rows, columns = (flat//c).astype(np.int32), (flat%c).astype(np.int32)
    values = residual.ravel()[flat].astype(np.float32)
    keep = values != 0
    rows, columns, values = rows[keep], columns[keep], values[keep]
    shift = np.bincount(columns, weights=values.astype(float)*train_frequency[rows], minlength=c)
    adjusted = replace(base, bias=(base.bias.astype(float)-shift).astype(np.float32))
    return SparseLeafModel(adjusted, rows, columns, values)


def training_weights(target, teacher_raw, task, selector, margin_scale=.5):
    if selector not in ('balanced_mse', 'balanced_margin'):
        raise ValueError('invalid weighted selector')
    n, c = teacher_raw.shape
    y = np.eye(c)[np.asarray(target, dtype=int)] if task == 'multiclass' else np.asarray(target)
    if y.shape != (n,c) or not np.isin(y, [0,1]).all():
        raise ValueError('invalid training target')
    positives = y.sum(axis=0)
    w = y/np.maximum(positives,1)+(1-y)/np.maximum(n-positives,1)
    if selector == 'balanced_margin':
        if margin_scale <= 0:
            raise ValueError('invalid margin scale')
        if task == 'multiclass':
            best = np.argmax(teacher_raw, axis=1)
            largest = teacher_raw[np.arange(n), best]
            second = np.partition(teacher_raw, -2, axis=1)[:,-2]
            competitor = np.broadcast_to(largest[:,None], (n,c)).copy()
            competitor[np.arange(n), best] = second
            margin = np.abs(teacher_raw-competitor)
        else:
            margin = np.abs(teacher_raw)
        w *= (margin_scale/(margin_scale+margin))**2
    return w/w.sum(axis=0,keepdims=True)


def coordinate_scores(design, residual, error, selector, weights=None):
    """Static single-coordinate decrease in weighted logit MSE.

    It is not an additive guarantee for selecting multiple correlated leaves.
    Center each indicator using its *unweighted* training visit frequency, which
    is exactly the bias-preserving inference update.
    """
    frequency = np.asarray(design.mean(axis=0)).ravel()
    if selector == 'energy':
        return residual.astype(float)**2 * frequency[:,None]
    if selector not in SELECTORS or weights is None or weights.shape != error.shape:
        raise ValueError('invalid coordinate scoring input')
    weighted_error = weights*error
    cross = design.T @ weighted_error-frequency[:,None]*weighted_error.sum(axis=0)
    mass = design.T @ weights
    squared = mass*(1-2*frequency[:,None])+frequency[:,None]**2*weights.sum(axis=0)
    score = 2*residual*cross-residual**2*squared
    return np.maximum(0, score)


def score_order(scores, residual):
    if scores.shape != residual.shape or not np.isfinite(scores).all():
        raise ValueError('invalid scores')
    flat = scores.ravel()
    order = np.argsort(-flat, kind='stable')
    return order[(flat[order]>0) & (residual.ravel()[order]!=0)]


def select_candidate(rows, reference_quality, budget):
    """Validation-only choice; test data are deliberately absent from the API."""
    candidates = [r for r in rows if budget.allows(Cost(r['parameters'],r['payload_bytes']))]
    if not candidates:
        raise ValueError('empty feasible candidate set')
    eligible = [r for r in candidates if quality_guard(r['validation'], reference_quality)]
    # Strong rank-only control: best validation macro-F1 over ALL allowed ranks.
    # Same criterion and fallback for every sparse family.
    chosen = min(eligible or candidates, key=lambda r:(-r['validation']['f1_macro'],
        r['validation']['loss'], r['payload_bytes'],r['parameters'],r['candidate_id']))
    return chosen, bool(eligible)
