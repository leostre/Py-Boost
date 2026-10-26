"""Cooperative gather -> GEMM beta=1 -> in-place FP64 output processing."""
from experiments.fused_leaf_gpu import FusedLeafGpu, SOURCE as FUSED_SOURCE


# Keep the tested routing, unique sparse writes and stable output reductions.
# Only the destination stride and final gather stores change.
SOURCE = FUSED_SOURCE.replace(
    'int n, int d, int trees, int width, int outputs, int has_residual) {',
    'int n, int d, int trees, int width, int outputs, int has_residual, int latent_stride, int low_rank) {')
_OLD_STORES = '''    if (k < width) latent[obj*width+k] = acc;
    if (has_residual && k < outputs) residual[obj*outputs+k] = corrections[k];'''
_NEW_STORES = '''    if (k < latent_stride) {
        double value = k < width ? acc : 0.;
        if (!low_rank && has_residual) value += corrections[k];
        latent[obj*latent_stride+k] = value;
    }
    if (low_rank && has_residual && k < outputs)
        residual[obj*outputs+k] = corrections[k];'''
if SOURCE.count(_OLD_STORES) != 1:
    raise RuntimeError('parent gather source changed')
SOURCE = SOURCE.replace(_OLD_STORES, _NEW_STORES)


class AccumulatedLeafGpu(FusedLeafGpu):
    def __init__(self, model):
        import cupy as cp
        super().__init__(model)
        self._decoder = cp.array(self._basis, dtype=cp.float64, order='C', copy=True)
        self.resident_bytes += self._decoder.nbytes
        self.kernel = cp.RawKernel(SOURCE, 'gather')

    def _accumulate(self, x):
        import cupy as cp
        import numpy as np
        from cupy.cublas import gemm
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features or not x.flags.c_contiguous:
            raise ValueError('resident features must be contiguous float32 of correct width')
        if bool(cp.isinf(x).any()):
            raise ValueError('invalid features')
        out = cp.empty((len(x), self.outputs), cp.float64)
        low_rank = self.representation == 'low_rank'
        latent = cp.empty((len(x), self.width), cp.float64) if low_rank else out
        if len(x):
            self.kernel((len(x),), (self.threads,), (x, self._nodes, self._node_offsets,
                self._leaf_offsets, self._coefficients, self._indptr, self._columns,
                self._values, latent, out, *(np.int32(v) for v in
                (len(x), self.n_features, self.n_trees, self.width, self.outputs,
                 self.has_residual, latent.shape[1], low_rank))),
                shared_mem=self.outputs*8 if self.has_residual else 0)
            if low_rank:
                # Installed CuPy GEMM handles contiguous C output directly via
                # transposed cuBLAS operands, not an F-layout output copy.
                result = gemm('N', 'T', latent, self._decoder, out=out,
                              alpha=1., beta=1. if self.has_residual else 0.)
                if result.data.ptr != out.data.ptr:
                    raise RuntimeError('GEMM copied the owned accumulation buffer')
        return out

    def raw_device(self, x):
        out = self._accumulate(x)
        out += self._bias
        return out

    def predict_device(self, x):
        import numpy as np
        out = self._accumulate(x)
        if len(x):
            # No separate residual or probability allocation, including the
            # implicit reference-class column initialized by the gather kernel.
            self.finish((len(x),), (self.threads,), (out, self._bias, out, out,
                *(np.int32(v) for v in (self.outputs, self.outputs, 0,
                                       self.task == 'multiclass'))), shared_mem=self.threads*8)
        return out
