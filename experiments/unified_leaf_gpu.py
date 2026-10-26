"""One cooperative CTA per object: route/gather -> shared decode -> output.

Only the N-by-C FP64 result is global request storage. Decoder B is stored as
r-by-C contiguous FP32: adjacent output lanes read adjacent coefficients.
No dense reconstruction of leaf values, CSR, per-request BLAS or N-by-r array.
"""
from experiments.prepared_leaf_gpu import PreparedLeafGpu


SOURCE = r'''
extern "C" __global__ void unified(
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, const float* a, const int* ptr,
    const int* cols, const float* vals, const float* basis,
    const float* bias, double* out, int n, int d, int trees, int probability) {
    int obj = blockIdx.x, k = threadIdx.x;
    __shared__ long long leaf;
    extern __shared__ double work[];
    double* corrections = work + WIDTH;
    if (HAS_RESIDUAL && k < OUTPUTS) corrections[k] = 0.;
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
        if (k < WIDTH) acc += (double)a[leaf*WIDTH+k];
        // Canonical unique sparse columns per leaf: one lane per coordinate.
        if (HAS_RESIDUAL)
            for (int s=ptr[leaf]+k; s<ptr[leaf+1]; s+=blockDim.x)
                corrections[cols[s]] += (double)vals[s];
        __syncthreads();
    }
    double value = k < WIDTH ? acc : 0.;
    if (LOW_RANK) {
        if (k < WIDTH) work[k] = acc;
        __syncthreads();
        value = 0.;
        if (k < OUTPUTS) {
            // Decode once AFTER all trees; no per-thread local rank array.
            #pragma unroll 1
            for (int q=0; q<WIDTH; ++q)
                value += work[q]*(double)basis[q*OUTPUTS+k];
        }
    }
    if (k < OUTPUTS) {
        if (HAS_RESIDUAL) value += corrections[k];
        value += (double)bias[k];
    }
    if (!probability) {
        if (k < OUTPUTS) out[obj*OUTPUTS+k] = value;
        return;
    }
    if (!MULTICLASS) {
        if (k < OUTPUTS) {
            double z = exp(-fabs(value));
            double p = value >= 0 ? 1./(1.+z) : z/(1.+z);
            out[obj*OUTPUTS+k] = fmin(1.-1e-7, fmax(1e-7, p));
        }
        return;
    }
    // All lanes must finish reading latent/residual before reusing shared work.
    __syncthreads();
    work[k] = k < OUTPUTS ? value : -__longlong_as_double(0x7ff0000000000000LL);
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (k < stride) work[k] = fmax(work[k], work[k+stride]);
        __syncthreads();
    }
    double e = k < OUTPUTS ? exp(value-work[0]) : 0.;
    __syncthreads();
    work[k] = e;
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (k < stride) work[k] += work[k+stride];
        __syncthreads();
    }
    if (k < OUTPUTS) out[obj*OUTPUTS+k] = fmin(1.-1e-6, fmax(1e-6, e/work[0]));
}
'''


class UnifiedLeafGpu(PreparedLeafGpu):
    def __init__(self, model):
        import cupy as cp
        import time
        super().__init__(model)
        self.threads = 32
        while self.threads < max(self.width, self.outputs):
            self.threads *= 2
        low_rank = self.representation == 'low_rank'
        # Replace, do not retain another decoder copy. Exact FP32 transpose.
        if low_rank:
            self._basis = cp.ascontiguousarray(self._basis.T)
        self.shared_bytes = max(self.width+(self.outputs if self.has_residual else 0),
                                self.threads)*8
        definitions = dict(WIDTH=self.width, OUTPUTS=self.outputs, LOW_RANK=int(low_rank),
                           HAS_RESIDUAL=int(self.has_residual), MULTICLASS=int(self.task == 'multiclass'))
        source = '\n'.join(f'#define {k} {v}' for k, v in definitions.items())+'\n'+SOURCE
        self.kernel = cp.RawKernel(source, 'unified', options=('--fmad=false',))
        started = time.perf_counter()
        self.kernel.compile()
        self.compilation_ms = (time.perf_counter()-started)*1000
        self.kernel_resources = dict(self.kernel.attributes)
        self.kernel_resources.update(threads=self.threads, dynamic_shared_bytes=self.shared_bytes,
                                     decoder_layout='r_by_outputs_C_FP32', **definitions)

    def _request(self, x, probability):
        import cupy as cp
        import numpy as np
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features or not x.flags.c_contiguous:
            raise ValueError('resident features must be contiguous float32 of correct width')
        if bool(cp.isinf(x).any()):
            raise ValueError('invalid features')
        out = cp.empty((len(x), self.outputs), cp.float64)
        if len(x):
            self.kernel((len(x),), (self.threads,), (x, self._nodes, self._node_offsets,
                self._leaf_offsets, self._coefficients, self._indptr, self._columns,
                self._values, self._basis, self._bias, out, *(np.int32(v) for v in
                (len(x), self.n_features, self.n_trees, probability))), shared_mem=self.shared_bytes)
        return out

    def raw_device(self, x):
        return self._request(x, False)

    def predict_device(self, x):
        return self._request(x, True)
