"""CPU-only, standalone fixed-topology leaf ensemble and functional PCA.

No training model, dense fallback, or GPU import is retained by a compact model.
All inference coefficients are float32; fitting the basis uses float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def probabilities(raw, task):
    raw = np.asarray(raw, dtype=np.float64)
    if task == 'multiclass':
        p = np.exp(raw - raw.max(axis=1, keepdims=True))
        return np.clip(p / p.sum(axis=1, keepdims=True), 1e-6, 1 - 1e-6)
    if task != 'multilabel':
        raise ValueError('unsupported task')
    z = np.exp(-np.abs(raw))
    return np.clip(np.where(raw >= 0, 1 / (1 + z), z / (1 + z)), 1e-7, 1 - 1e-7)


@dataclass(frozen=True)
class LeafEnsemble:
    nodes: np.ndarray
    node_offsets: np.ndarray
    leaf_offsets: np.ndarray
    coefficients: np.ndarray
    basis: np.ndarray
    bias: np.ndarray
    n_features: int
    task: str
    representation: str = 'dense'

    def __post_init__(self):
        if self.task not in ('multiclass', 'multilabel') or self.n_features < 1:
            raise ValueError('invalid task or feature count')
        if self.representation not in ('dense', 'low_rank', 'reference_class'):
            raise ValueError('invalid representation')
        for name in ('nodes', 'coefficients', 'basis', 'bias'):
            a = getattr(self, name)
            if a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite float32')
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 4:
            raise ValueError('node shape')
        if self.coefficients.ndim != 2 or self.bias.ndim != 1 or self.basis.ndim != 2:
            raise ValueError('coefficient/bias/basis shape')
        for offsets, end in ((self.node_offsets, len(self.nodes)),
                             (self.leaf_offsets, len(self.coefficients))):
            if (offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) < 2
                    or offsets[0] != 0 or offsets[-1] != end or np.any(np.diff(offsets) <= 0)):
                raise ValueError('invalid offsets')
        if len(self.node_offsets) != len(self.leaf_offsets):
            raise ValueError('tree offset mismatch')
        c, r = len(self.bias), self.coefficients.shape[1]
        if self.representation == 'low_rank':
            if self.basis.shape != (c, r) or not 1 <= r <= c:
                raise ValueError('basis dimensions')
            if not np.allclose(self.basis.T @ self.basis, np.eye(r), atol=2e-5):
                raise ValueError('basis is not orthonormal')
        elif self.basis.shape != (0, 0):
            raise ValueError('dense/reference model cannot retain a decoder')
        elif self.representation == 'dense' and r != c:
            raise ValueError('dense width mismatch')
        elif self.representation == 'reference_class' and (r != c - 1 or self.task != 'multiclass'):
            raise ValueError('reference-class width/task mismatch')
        for t in range(self.n_trees):
            nodes = self.nodes[self.node_offsets[t]:self.node_offsets[t + 1]]
            leaves = int(self.leaf_offsets[t + 1] - self.leaf_offsets[t])
            seen, used = set(), set()
            pending = [0]
            while pending:
                index = pending.pop()
                if index < 0:
                    used.add(-index - 1)
                    continue
                if index in seen or index >= len(nodes):
                    raise ValueError('cycle, shared child, or invalid node')
                seen.add(index)
                feature, _, left, right = nodes[index]
                if feature != int(feature) or not 1 <= abs(feature) <= self.n_features:
                    raise ValueError('invalid feature code')
                if left != int(left) or right != int(right):
                    raise ValueError('noninteger pointer')
                pending.extend((int(left), int(right)))
            if len(seen) != len(nodes) or used != set(range(leaves)):
                raise ValueError('unreachable node or noncompact leaves')

    @property
    def n_trees(self):
        return len(self.node_offsets) - 1

    @property
    def parameter_count(self):
        # Reference-class bias has an implicit zero final component.
        return int(self.coefficients.size + self.basis.size + self.bias.size
                   - (self.representation == 'reference_class'))

    @property
    def payload_bytes(self):
        # Include the physically stored zero bias too: no hidden byte savings.
        return int(sum(a.nbytes for a in (self.nodes, self.node_offsets, self.leaf_offsets,
                                         self.coefficients, self.basis, self.bias)))

    def leaf_indices(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features or np.isinf(x).any():
            raise ValueError('invalid features')
        result = np.empty((self.n_trees, len(x)), dtype=np.int64)
        for t in range(self.n_trees):
            nodes = self.nodes[self.node_offsets[t]:self.node_offsets[t + 1]]
            index = np.zeros(len(x), dtype=np.int64)
            while np.any(index >= 0):
                rows = np.flatnonzero(index >= 0)
                current = nodes[index[rows]]
                feature = current[:, 0].astype(int)
                value = x[rows, np.abs(feature) - 1]
                right = np.where(np.isnan(value), feature > 0, value > current[:, 1])
                index[rows] = current[np.arange(len(rows)), 2 + right.astype(int)].astype(int)
            result[t] = -index - 1 + self.leaf_offsets[t]
        return result

    def contributions(self, leaves):
        # Sum before decoding. FP64 canonical accumulation bounds FP32-order noise.
        out = np.zeros((leaves.shape[1], self.coefficients.shape[1]), dtype=np.float64)
        for indices in leaves:
            out += self.coefficients[indices]
        return out

    def raw_from_leaves(self, leaves):
        out = self.contributions(leaves)
        if self.representation == 'low_rank':
            out = out @ self.basis.T
        elif self.representation == 'reference_class':
            out = np.column_stack((out, np.zeros(len(out))))
        return out + self.bias

    def predict(self, features):
        return probabilities(self.raw_from_leaves(self.leaf_indices(features)), self.task)

    def save(self, path):
        # Uncompressed NPZ: actual file size and raw payload are both reported.
        np.savez(path, version=np.array(1), nodes=self.nodes, node_offsets=self.node_offsets,
                 leaf_offsets=self.leaf_offsets, coefficients=self.coefficients,
                 basis=self.basis, bias=self.bias, n_features=np.array(self.n_features),
                 task=np.array(self.task), representation=np.array(self.representation))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {'version', 'nodes', 'node_offsets', 'leaf_offsets', 'coefficients',
                        'basis', 'bias', 'n_features', 'task', 'representation'}
            if set(data.files) != expected or int(data['version']) != 1:
                raise ValueError('unknown artifact schema')
            return cls(**{k: data[k].copy() for k in
                          ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')},
                       n_features=int(data['n_features']), task=str(data['task']),
                       representation=str(data['representation']))


def export_teacher(model, n_features, task):
    def cpu(a):
        return a.get() if hasattr(a, 'get') else np.asarray(a)
    nodes, values, node_offsets, leaf_offsets = [], [], [0], [0]
    for tree in model.models:
        if int(tree.ngroups) != 1 or np.any(cpu(tree.group_index) != 0):
            raise ValueError('H-LR1 requires one shared output tree per iteration')
        block = cpu(tree.test_format).reshape(-1, 4).astype(np.float32, copy=True)
        reachable = np.unique((-block[:, 2:][block[:, 2:] < 0] - 1).astype(int))
        lookup = {old: new for new, old in enumerate(reachable)}
        for row in block:
            for side in (2, 3):
                if row[side] < 0:
                    row[side] = -lookup[-int(row[side]) - 1] - 1
        nodes.append(block)
        values.append(cpu(tree.values)[reachable].astype(np.float32))
        node_offsets.append(node_offsets[-1] + len(block))
        leaf_offsets.append(leaf_offsets[-1] + len(reachable))
    return LeafEnsemble(np.concatenate(nodes), np.array(node_offsets, dtype=np.int64),
                        np.array(leaf_offsets, dtype=np.int64), np.concatenate(values),
                        np.empty((0, 0), dtype=np.float32),
                        cpu(model.base_score).ravel().astype(np.float32), n_features, task)


def reference_class_model(dense):
    if dense.task != 'multiclass' or dense.representation != 'dense':
        raise ValueError('requires dense multiclass ensemble')
    bias = dense.bias - dense.bias[-1]
    return LeafEnsemble(dense.nodes, dense.node_offsets, dense.leaf_offsets,
                        dense.coefficients[:, :-1] - dense.coefficients[:, -1:],
                        np.empty((0, 0), dtype=np.float32), bias, dense.n_features,
                        dense.task, 'reference_class')


def functional_basis(contributions, task):
    y = np.asarray(contributions, dtype=np.float64)
    if y.ndim != 2 or min(y.shape) < 2 or not np.isfinite(y).all():
        raise ValueError('invalid calibration contributions')
    c = y.shape[1]
    # Explicit contrast space prevents a zero singular value selecting the
    # softmax common-shift direction in a degenerate spectrum.
    if task == 'multiclass':
        contrast, _ = np.linalg.qr(np.vstack((np.eye(c - 1), -np.ones(c - 1))))
    elif task == 'multilabel':
        contrast = np.eye(c)
    else:
        raise ValueError('invalid task')
    mean = y.mean(axis=0)
    centered = (y - mean) @ contrast
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, contrast @ vt.T, singular


def compress(dense, mean, basis):
    b = np.asarray(basis, dtype=np.float64)
    if dense.representation != 'dense' or b.ndim != 2 or b.shape[0] != len(dense.bias):
        raise ValueError('invalid source or basis')
    return LeafEnsemble(dense.nodes, dense.node_offsets, dense.leaf_offsets,
                        (dense.coefficients.astype(float) @ b).astype(np.float32),
                        b.astype(np.float32),
                        (dense.bias + mean - (mean @ b) @ b.T).astype(np.float32),
                        dense.n_features, dense.task, 'low_rank')


def quality_guard(candidate, reference, tolerance=.05):
    return bool(candidate['loss'] <= (1 + tolerance) * reference['loss']
                and candidate['f1_macro'] >= (1 - tolerance) * reference['f1_macro']
                and candidate['f1_weighted'] >= (1 - tolerance) * reference['f1_weighted'])


def select_rank(validation_curve, reference_quality, reference_parameters, fraction=.75):
    """Only validation data enter this decision; no test metric argument exists.

    Fallback: smallest quality-eligible rank even when it misses the budget;
    if no quality-eligible rank exists, the largest rank (diagnostic failure).
    """
    eligible = [r for r in validation_curve if quality_guard(r['quality'], reference_quality)]
    feasible = [r for r in eligible if r['parameters'] <= fraction * reference_parameters]
    candidates = feasible or eligible or validation_curve[-1:]
    if not candidates:
        raise ValueError('empty rank curve')
    selected = min(candidates, key=lambda r: (r['parameters'], r['rank']))
    return selected['rank'], bool(feasible)
