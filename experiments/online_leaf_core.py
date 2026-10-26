"""Pure weighted leaf projection and standalone compact tree assembly."""
from dataclasses import dataclass
from decimal import Decimal,ROUND_CEILING
import numpy as np

from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel


def residual_rank_allowance(fraction,outputs):
    """ceil(3*f*C) in the decimal configuration's meaning, not binary noise.

    A COO entry stores value + two indices. This is only a nominal byte
    allowance; actual leaf counts, bases and topology still need measurement.
    In particular f=.05,C=100 means15, not ceil(15.000000000000002)=16.
    """
    f = Decimal(str(fraction))
    if (not f.is_finite() or not 0 <= f <= 1 or isinstance(outputs,bool)
            or not isinstance(outputs,(int,np.integer)) or outputs < 1):
        raise ValueError('invalid nominal residual rank budget')
    return int((Decimal(3)*f*int(outputs)).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class LeafProjection:
    coefficients: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    weighted_error: float

    def decoded(self, basis):
        out = self.coefficients.astype(float) @ np.asarray(basis, dtype=float).T
        out[self.rows, self.columns] += self.values
        return out


def weighted_projection(target, weights, basis, residual_fraction=0.):
    """Minimize the local diagonal Newton surrogate in span(B), then add L.

    D = leaf Hessian sum + lambda. Target already includes learning rate.
    Choose floor(fraction*M*C) coordinates by D*(target-Ba)^2, then refit a
    with these coordinates removed from the quadratic. Sparse entries restore
    their target values. This is one deterministic support/refit step, not a
    globally optimal sparse-plus-low-rank decomposition.
    """
    v, d, b = (np.asarray(a, dtype=float) for a in (target, weights, basis))
    if (v.ndim != 2 or d.shape != v.shape or b.ndim != 2 or b.shape[0] != v.shape[1]
            or not 1 <= b.shape[1] <= b.shape[0] or min(v.shape) < 1
            or not all(np.isfinite(a).all() for a in (v,d,b)) or np.any(d <= 0)
            or not 0 <= residual_fraction <= 1
            or not np.allclose(b.T@b, np.eye(b.shape[1]), atol=2e-5)):
        raise ValueError('invalid weighted leaf projection')
    def fit(diagonal):
        lhs = np.einsum('cr,lc,cs->lrs', b, diagonal, b, optimize=True)
        rhs = (diagonal*v)@b
        # Removing sparse coordinates can make this PSD matrix singular even
        # when the number of remaining rows exceeds r. Use a checked symmetric
        # pseudoinverse instead of accepting an unstable near-singular solve.
        eigenvalues, vectors = np.linalg.eigh(lhs)
        cutoff = 1e-12*np.maximum(eigenvalues[:,-1:],np.finfo(float).tiny)
        inverse = np.divide(1.,eigenvalues,out=np.zeros_like(eigenvalues),where=eigenvalues > cutoff)
        coordinates = np.einsum('lri,lr->li',vectors,rhs)*inverse
        return np.einsum('lri,li->lr',vectors,coordinates)
    a = fit(d)
    error = v-a@b.T
    count = int(np.floor(residual_fraction*v.size))
    keys = np.empty(0, np.int64)
    if count:
        scores = (d*error**2).ravel()
        # Stable row-major tie break and no meaningless zero corrections.
        order = np.argsort(-scores, kind='stable')
        keys = np.sort(order[:count][scores[order[:count]] > 1e-24])
        if len(keys):
            diagonal = d.copy()
            diagonal.ravel()[keys] = 0.
            a = fit(diagonal)
    a = a.astype(np.float32)
    rows, cols = np.unravel_index(keys, v.shape)
    values = (v-a.astype(float)@b.T)[rows,cols].astype(np.float32)
    nonzero = values != 0
    rows, cols, values = rows[nonzero].astype(np.int32), cols[nonzero].astype(np.int32), values[nonzero]
    reconstructed = a.astype(float)@b.T
    reconstructed[rows,cols] += values
    return LeafProjection(a, rows, cols, values, float(np.sum(d*(v-reconstructed)**2)))


@dataclass(frozen=True)
class TreePart:
    nodes: np.ndarray
    coefficients: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    @property
    def payload_bytes(self):
        return sum(a.nbytes for a in (self.nodes,self.coefficients,self.rows,self.columns,self.values))

    def decoded(self, basis=None):
        out = self.coefficients.astype(float) if basis is None else self.coefficients.astype(float)@basis.astype(float).T
        out[self.rows,self.columns] += self.values
        return out


def dense_part(nodes, values):
    return TreePart(nodes.copy(),values.copy(),np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32))


def project_part(part, weights, basis, residual_fraction):
    p = weighted_projection(part.coefficients, weights, basis, residual_fraction)
    return TreePart(part.nodes,p.coefficients,p.rows,p.columns,p.values)


def compact_topology(nodes, values):
    """Remove native reserved/unreachable leaf slots and remap child pointers."""
    block = np.asarray(nodes,dtype=np.float32).reshape(-1,4).copy()
    ids = np.unique((-block[:,2:][block[:,2:] < 0]-1).astype(int))
    lookup = np.full(len(values),-1,np.int32)
    lookup[ids] = np.arange(len(ids),dtype=np.int32)
    for side in (2,3):
        mask = block[:,side] < 0
        block[mask,side] = -lookup[-block[mask,side].astype(int)-1]-1
    return dense_part(block,np.asarray(values,dtype=np.float32)[ids]), lookup, ids


def stitch(parts, basis, bias, n_features, task, shifts=None):
    if not parts:
        raise ValueError('empty compact ensemble')
    node_offsets = np.r_[0,np.cumsum([len(p.nodes) for p in parts])].astype(np.int64)
    leaf_offsets = np.r_[0,np.cumsum([len(p.coefficients) for p in parts])].astype(np.int64)
    b = np.empty((0,0),np.float32) if basis is None else np.asarray(basis,dtype=np.float32)
    base = LeafEnsemble(np.concatenate([p.nodes for p in parts]),node_offsets,leaf_offsets,
        np.concatenate([p.coefficients for p in parts]),b,np.asarray(bias,dtype=np.float32),
        n_features,task,'dense' if basis is None else 'low_rank')
    rows = np.concatenate([p.rows+int(offset) for p,offset in zip(parts,leaf_offsets[:-1])]).astype(np.int32)
    columns = np.concatenate([p.columns for p in parts])
    values = np.concatenate([p.values for p in parts])
    return ThresholdModel(base,rows,columns,values,np.zeros(len(bias),np.float32) if shifts is None else shifts)
