"""FP64 GEMM accumulation into one owned residual/logit/probability buffer."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg.blas import dgemm

from experiments.compiled_leaf_inference import gather_into, finish_into
from experiments.prepared_leaf_inference import PreparedLeafInference, frozen_array
from experiments.output_calibration_core import labels_from_raw


@dataclass(frozen=True)
class AccumulatedLeafInference:
    snapshot: PreparedLeafInference
    decoder: np.ndarray
    compilation_ms: float

    @classmethod
    def prepare(cls, model):
        snapshot = PreparedLeafInference.prepare(model)
        # Exact FP32->FP64 conversion, not a refit. Retained cache is accounted.
        # Its transpose and latent.T are F-contiguous BLAS inputs without copies.
        decoder = frozen_array(snapshot.base.basis.T.astype(np.float64))
        runtime = cls(snapshot, decoder, 0.)
        started = time.perf_counter()
        runtime.predict(np.empty((0, snapshot.base.n_features), np.float32))
        return cls(snapshot, decoder, (time.perf_counter()-started)*1000)

    @property
    def resident_bytes(self):
        return self.snapshot.resident_bytes + self.decoder.nbytes

    def _evaluate(self, features, probability):
        x = np.asarray(features, dtype=np.float32)
        s, b = self.snapshot, self.snapshot.base
        if x.ndim != 2 or x.shape[1] != b.n_features or np.isinf(x).any():
            raise ValueError('invalid features')
        x = np.ascontiguousarray(x)
        if not x.flags.writeable:
            x = x.copy()
        out = np.zeros((len(x), len(b.bias)), np.float64)
        low_rank = b.representation == 'low_rank'
        latent = np.zeros((len(x), b.coefficients.shape[1]), np.float64) if low_rank else out
        gather_into(x, b.nodes, b.node_offsets, b.leaf_offsets, b.coefficients,
                    s.indptr, s.columns, s.values, latent, out)
        if low_rank and len(x):
            # out.T <- decoder.T @ latent.T + out.T. All three BLAS arguments
            # are F-contiguous views. No returned dense decoder product exists.
            result = dgemm(1., self.decoder.T, latent.T, beta=1. if len(s.values) else 0.,
                           c=out.T, overwrite_c=1)
            if result.ctypes.data != out.ctypes.data:
                raise RuntimeError('BLAS copied the owned accumulation buffer')
        del latent
        # The residual is already included. Each element is read before its own
        # overwrite; softmax subsequently operates only on this owned output.
        finish_into(out, b.bias, out, out, b.task == 'multiclass', False, probability)
        return out

    def raw(self, features):
        return self._evaluate(features, False)

    def predict(self, features):
        return self._evaluate(features, True)

    def predict_labels(self, features):
        return labels_from_raw(self.raw(features), self.snapshot.base.task, self.snapshot.decision_shifts)
