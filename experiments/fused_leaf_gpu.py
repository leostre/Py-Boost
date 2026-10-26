"""Cooperative sparse gather + one decoder + fused FP64 output processing.

No sparse matrix, no dense M-by-C fallback. Each unique residual coordinate of a
visited leaf is processed by one lane. Barriers preserve tree accumulation order.
"""
from experiments.prepared_leaf_gpu import PreparedLeafGpu

SOURCE = r'''
extern "C" __global__ void gather(
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, const float* a, const int* ptr,
    const int* cols, const float* vals, double* latent, double* residual,
    int n, int d, int trees, int width, int outputs, int has_residual) {
    int obj = blockIdx.x, k = threadIdx.x;
    __shared__ long long leaf;
    extern __shared__ double corrections[];
    if (has_residual && k < outputs) corrections[k] = 0.;
    double acc = 0.;
    __syncthreads();
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
        // Unique columns within one leaf: no atomics or same-tree write races.
        if (has_residual)
            for (int s=ptr[leaf]+k; s<ptr[leaf+1]; s+=blockDim.x)
                corrections[cols[s]] += (double)vals[s];
        __syncthreads();
    }
    if (k < width) latent[obj*width+k] = acc;
    if (has_residual && k < outputs) residual[obj*outputs+k] = corrections[k];
}

extern "C" __global__ void finish(
    const double* decoded, const float* bias, const double* residual,
    double* out, int width, int outputs, int has_residual, int multiclass) {
    int obj = blockIdx.x, k = threadIdx.x;
    extern __shared__ double work[];
    double value = k < width ? decoded[obj*width+k] : 0.;
    if (k < outputs) {
        value += (double)bias[k];
        if (has_residual) value += residual[obj*outputs+k];
    }
    if (!multiclass) {
        if (k < outputs) {
            double z = exp(-fabs(value));
            double p = value >= 0 ? 1./(1.+z) : z/(1.+z);
            out[obj*outputs+k] = fmin(1.-1e-7, fmax(1e-7, p));
        }
        return;
    }
    // Exact IEEE-754 -infinity via an NVRTC builtin; no host math.h/macros.
    work[k] = k < outputs ? value : -__longlong_as_double(0x7ff0000000000000LL);
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (k < stride) work[k] = fmax(work[k], work[k+stride]);
        __syncthreads();
    }
    double e = k < outputs ? exp(value-work[0]) : 0.;
    __syncthreads();
    work[k] = e;
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (k < stride) work[k] += work[k+stride];
        __syncthreads();
    }
    if (k < outputs) out[obj*outputs+k] = fmin(1.-1e-6, fmax(1e-6, e/work[0]));
}
'''


class FusedLeafGpu(PreparedLeafGpu):
    def __init__(self, model):
        import cupy as cp
        super().__init__(model)
        self.threads = 32
        while self.threads < max(self.width, self.outputs):
            self.threads *= 2
        self.kernel = cp.RawKernel(SOURCE, 'gather')
        self.finish = cp.RawKernel(SOURCE, 'finish')

    def _gather(self, x):
        import cupy as cp
        import numpy as np
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features or not x.flags.c_contiguous:
            raise ValueError('resident features must be contiguous float32 of correct width')
        if bool(cp.isinf(x).any()):
            raise ValueError('invalid features')
        latent = cp.empty((len(x), self.width), cp.float64)
        residual = cp.empty((len(x), self.outputs if self.has_residual else 0), cp.float64)
        if len(x):
            self.kernel((len(x),), (self.threads,), (x, self._nodes, self._node_offsets,
                self._leaf_offsets, self._coefficients, self._indptr, self._columns,
                self._values, latent, residual, *(np.int32(v) for v in
                (len(x), self.n_features, self.n_trees, self.width, self.outputs, self.has_residual))),
                shared_mem=self.outputs*8 if self.has_residual else 0)
        decoded = latent @ self._basis.T if self.representation == 'low_rank' else latent
        return decoded, residual

    def raw_device(self, x):
        import cupy as cp
        decoded, residual = self._gather(x)
        if self.representation == 'reference_class':
            decoded = cp.column_stack((decoded, cp.zeros(len(x))))
        decoded += self._bias
        if self.has_residual:
            decoded += residual
        return decoded

    def predict_device(self, x):
        import cupy as cp
        import numpy as np
        decoded, residual = self._gather(x)
        out = cp.empty((len(x), self.outputs), cp.float64)
        if len(x):
            self.finish((len(x),), (self.threads,), (decoded, self._bias, residual, out,
                *(np.int32(v) for v in (decoded.shape[1], self.outputs, self.has_residual,
                                       self.task == 'multiclass'))), shared_mem=self.threads*8)
        return out
