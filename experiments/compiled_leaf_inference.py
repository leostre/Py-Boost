"""Single-thread compiled AB+L inference with the unchanged FP64 contract.

All request arrays are allocated by NumPy before entering Numba. The compiled
loops allocate no NRT arrays, so tracemalloc is not evaded by moving buffers
into an untracked allocator. Compilation and snapshot preparation are explicit.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
from numba import njit

from experiments.prepared_leaf_inference import PreparedLeafInference
from experiments.output_calibration_core import labels_from_raw


@njit(cache=False, fastmath=False, nogil=True)
def gather_into(x, nodes, node_offsets, leaf_offsets, coefficients, indptr,
                columns, values, latent, residual):
    has_residual = len(values) != 0
    for i in range(len(x)):
        for t in range(len(node_offsets)-1):
            pointer = 0
            while pointer >= 0:
                node = node_offsets[t]+pointer
                feature = int(nodes[node, 0])
                value = x[i, abs(feature)-1]
                right = feature > 0 if math.isnan(value) else value > nodes[node, 1]
                pointer = int(nodes[node, 3 if right else 2])
            leaf = leaf_offsets[t]-pointer-1
            for j in range(coefficients.shape[1]):
                latent[i, j] += float(coefficients[leaf, j])
            if has_residual:
                for p in range(indptr[leaf], indptr[leaf+1]):
                    residual[i, columns[p]] += float(values[p])


@njit(cache=False, fastmath=False, nogil=True)
def finish_into(decoded, bias, residual, out, multiclass, has_residual, probability):
    for i in range(len(out)):
        maximum = -math.inf
        for j in range(out.shape[1]):
            value = decoded[i, j] if j < decoded.shape[1] else 0.
            value += float(bias[j])
            if has_residual:
                value += residual[i, j]
            out[i, j] = value
            maximum = max(maximum, value)
        if not probability:
            continue
        if multiclass:
            total = 0.
            for j in range(out.shape[1]):
                out[i, j] = math.exp(out[i, j]-maximum)
                total += out[i, j]
            for j in range(out.shape[1]):
                out[i, j] = min(1.-1e-6, max(1e-6, out[i, j]/total))
        else:
            for j in range(out.shape[1]):
                value = out[i, j]
                z = math.exp(-abs(value))
                p = 1./(1.+z) if value >= 0 else z/(1.+z)
                out[i, j] = min(1.-1e-7, max(1e-7, p))


@dataclass(frozen=True)
class CompiledLeafInference:
    snapshot: PreparedLeafInference
    compilation_ms: float

    @classmethod
    def prepare(cls, model):
        snapshot = PreparedLeafInference.prepare(model)
        runtime = cls(snapshot, 0.)
        t = time.perf_counter()
        # Same dtypes/layouts as requests; zero objects trigger compilation only.
        runtime.predict(np.empty((0, snapshot.base.n_features), np.float32))
        return cls(snapshot, (time.perf_counter()-t)*1000)

    @property
    def resident_bytes(self):
        return self.snapshot.resident_bytes

    def _evaluate(self, features, probability):
        x = np.asarray(features, dtype=np.float32)
        s, b = self.snapshot, self.snapshot.base
        if x.ndim != 2 or x.shape[1] != b.n_features or np.isinf(x).any():
            raise ValueError('invalid features')
        # One C-layout signature for sliced inputs, too; copies are request costs.
        x = np.ascontiguousarray(x)
        if not x.flags.writeable:
            x = x.copy()  # Keep the precompiled signature; never modify caller input.
        latent = np.zeros((len(x), b.coefficients.shape[1]), np.float64)
        residual = np.zeros((len(x), len(b.bias) if len(s.values) else 0), np.float64)
        gather_into(x, b.nodes, b.node_offsets, b.leaf_offsets, b.coefficients,
                    s.indptr, s.columns, s.values, latent, residual)
        if b.representation == 'low_rank':
            decoded = latent @ b.basis.T
            del latent
        else:
            decoded = latent
        out = np.empty((len(x), len(b.bias)), np.float64)
        finish_into(decoded, b.bias, residual, out, b.task == 'multiclass', bool(len(s.values)), probability)
        return out

    def raw(self, features):
        return self._evaluate(features, False)

    def predict(self, features):
        return self._evaluate(features, True)

    def predict_labels(self, features):
        return labels_from_raw(self.raw(features), self.snapshot.base.task, self.snapshot.decision_shifts)
