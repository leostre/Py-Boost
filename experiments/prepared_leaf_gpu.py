"""Fused streaming leaf gather: one block per object, no visit matrix/CSR.

Tree routing is shared inside a block. Components accumulate in float64 in tree
order; the low-rank decoder is applied exactly once after all trees.
"""
from experiments.prepared_leaf_inference import PreparedLeafInference

SOURCE = r'''
extern "C" __global__ void gather(
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, const float* a, const int* ptr,
    const int* cols, const float* vals, double* latent, double* residual,
    int n, int d, int trees, int width, int outputs, int has_residual) {
    int obj = blockIdx.x, k = threadIdx.x;
    __shared__ long long leaf;
    double acc = 0., correction = 0.;
    for (int t=0; t<trees; ++t) {
        if (k == 0) {
            int p = 0;
            while (p >= 0) {
                const float* node = nodes + 4*(noff[t]+p);
                int f = (int)node[0];
                float value = x[obj*d+abs(f)-1];
                bool right = isnan(value) ? f > 0 : value > node[1];
                p = (int)node[2+right];
            }
            leaf = loff[t]-p-1;
        }
        __syncthreads();
        if (k < width) acc += (double)a[leaf*width+k];
        if (has_residual && k < outputs)
            for (int s=ptr[leaf]; s<ptr[leaf+1]; ++s)
                if (cols[s] == k) correction += (double)vals[s];
        __syncthreads();
    }
    if (k < width) latent[obj*width+k] = acc;
    if (has_residual && k < outputs) residual[obj*outputs+k] = correction;
}
'''


class PreparedLeafGpu:
    def __init__(self, model):
        import cupy as cp
        snapshot = PreparedLeafInference.prepare(model)
        b = snapshot.base
        self.task, self.representation = b.task, b.representation
        self.n_features, self.n_trees = b.n_features, b.n_trees
        self.width, self.outputs = b.coefficients.shape[1], len(b.bias)
        self.threads = max(32, ((max(self.width, self.outputs)+31)//32)*32)
        if self.threads > 1024:
            raise ValueError('fused implementation supports up to 1024 outputs')
        for name in ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'):
            setattr(self, '_'+name, cp.array(getattr(b, name), copy=True))
        for name in ('indptr', 'columns', 'values', 'decision_shifts'):
            setattr(self, '_'+name, cp.array(getattr(snapshot, name), copy=True))
        self.has_residual = bool(len(snapshot.values))
        self.resident_bytes = snapshot.resident_bytes
        self.kernel = cp.RawKernel(SOURCE, 'gather')

    def raw_device(self, x):
        import cupy as cp
        import numpy as np
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features or not x.flags.c_contiguous:
            raise ValueError('resident features must be contiguous float32 of correct width')
        if bool(cp.isinf(x).any()):
            raise ValueError('invalid features')
        latent = cp.empty((len(x), self.width), cp.float64)
        residual = cp.empty((len(x), self.outputs), cp.float64) if self.has_residual else cp.empty(0, cp.float64)
        if len(x):
            self.kernel((len(x),), (self.threads,), (x, self._nodes, self._node_offsets,
                self._leaf_offsets, self._coefficients, self._indptr, self._columns,
                self._values, latent, residual, *(np.int32(v) for v in
                (len(x), self.n_features, self.n_trees, self.width, self.outputs, self.has_residual))))
        if self.representation == 'low_rank':
            raw = latent @ self._basis.T
        elif self.representation == 'reference_class':
            raw = cp.column_stack((latent, cp.zeros(len(x))))
        else:
            raw = latent
        raw += self._bias
        if self.has_residual:
            raw += residual
        return raw

    def predict_device(self, x):
        from experiments.inference_benchmark_gpu import GpuThresholdReference
        return GpuThresholdReference.probability(self, self.raw_device(x))

    def predict(self, x):
        import cupy as cp
        # CuPy ascontiguousarray accepts device arrays, not NumPy inputs.
        device = cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))
        return self.predict_device(device).get()
