"""Owned, immutable inference snapshot; no design matrix or per-call CSR.

Preparation replaces COO rows with a leaf-indexed offset array. Requests stream
one tree at a time and gather only visited float32 coefficients into float64
accumulators. No dense M-by-C residual or expanded low-rank model is retained.
"""
from dataclasses import dataclass
import numpy as np

from experiments.shared_leaf_basis import LeafEnsemble, probabilities
from experiments.output_calibration_core import labels_from_raw


def frozen_array(value):
    # bytes-backed arrays cannot be made writable again; also break source aliases.
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class PreparedLeafInference:
    base: LeafEnsemble
    indptr: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    decision_shifts: np.ndarray

    @classmethod
    def prepare(cls, model):
        names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
        base = LeafEnsemble(**{n: frozen_array(getattr(model.base, n)) for n in names},
            n_features=model.base.n_features, task=model.base.task,
            representation=model.base.representation)
        # Source dataclasses are frozen, but their NumPy contents can be mutated.
        # Validate once before handing index arrays to unchecked compiled loops.
        from experiments.uniform_threshold_core import ThresholdModel
        ThresholdModel(base, model.rows, model.columns, model.values, model.decision_shifts)
        if len(model.values):
            order = np.lexsort((model.columns, model.rows))
            counts = np.bincount(model.rows, minlength=len(base.coefficients))
            if len(model.values) > np.iinfo(np.int32).max:
                raise ValueError('residual exceeds int32 offsets')
            ptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
        else:
            order, ptr = np.array([], np.int64), np.array([], np.int32)
        return cls(base, frozen_array(ptr), frozen_array(model.columns[order]),
                   frozen_array(model.values[order]), frozen_array(model.decision_shifts))

    @property
    def resident_bytes(self):
        return self.base.payload_bytes + sum(a.nbytes for a in
            (self.indptr, self.columns, self.values, self.decision_shifts))

    def raw(self, features):
        x = np.asarray(features, dtype=np.float32)
        b = self.base
        if x.ndim != 2 or x.shape[1] != b.n_features or np.isinf(x).any():
            raise ValueError('invalid features')
        latent = np.zeros((len(x), b.coefficients.shape[1]), np.float64)
        residual = np.zeros((len(x), len(b.bias)), np.float64) if len(self.values) else None
        for t in range(b.n_trees):
            nodes = b.nodes[b.node_offsets[t]:b.node_offsets[t+1]]
            index = np.zeros(len(x), np.int64)
            while np.any(index >= 0):
                objects = np.flatnonzero(index >= 0)
                current = nodes[index[objects]]
                feature = current[:, 0].astype(int)
                value = x[objects, np.abs(feature)-1]
                right = np.where(np.isnan(value), feature > 0, value > current[:, 1])
                index[objects] = current[np.arange(len(objects)), 2+right.astype(int)].astype(int)
            leaves = -index-1+b.leaf_offsets[t]
            latent += b.coefficients[leaves]
            if residual is not None:
                starts = self.indptr[leaves]
                counts = self.indptr[leaves+1]-starts
                objects = np.repeat(np.arange(len(x)), counts)
                # Ragged row gather; each object/output occurs once per tree.
                positions = (np.repeat(starts, counts) + np.arange(len(objects))
                             - np.repeat(np.cumsum(counts)-counts, counts))
                residual[objects, self.columns[positions]] += self.values[positions]
        if b.representation == 'low_rank':
            raw = latent @ b.basis.T
        elif b.representation == 'reference_class':
            raw = np.column_stack((latent, np.zeros(len(x))))
        else:
            raw = latent
        raw += b.bias
        if residual is not None:
            raw += residual
        return raw

    def predict(self, features):
        return probabilities(self.raw(features), self.base.task)

    def predict_labels(self, features):
        return labels_from_raw(self.raw(features), self.base.task, self.decision_shifts)
