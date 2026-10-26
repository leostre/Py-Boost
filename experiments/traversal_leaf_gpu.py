"""Frozen-model traversal alternatives; no fitting or dense leaf reconstruction.

The current study supports at most 128 outputs, covering the frozen cohort.
The baseline UnifiedLeafGpu remains unchanged. Every path preserves FP32
parameters, FP64 accumulation/output, NaN routing and strict > comparisons.
"""
from dataclasses import dataclass
import time

from experiments.prepared_leaf_gpu import PreparedLeafGpu


@dataclass(frozen=True)
class TraversalPlan:
    variant: str
    threads: int
    groups: int
    tile_rows: int
    shared_bytes: int

    def index_workspace_bytes(self, rows, trees):
        if rows < 0 or trees < 1:
            raise ValueError('invalid request dimensions')
        return 4*min(rows, self.tile_rows)*trees if self.variant == 'tiled' else 0


def traversal_plan(variant, width, outputs, has_residual):
    if (variant not in ('warp', 'groups', 'tiled') or not 1 <= width <= outputs <= 128):
        raise ValueError('traversal study requires 1 <= width <= outputs <= 128')
    if variant == 'warp':
        return TraversalPlan(variant, 128, 4, 0, 4*(width+outputs*bool(has_residual))*8)
    if variant == 'groups':
        return TraversalPlan(variant, 128, 4, 0, max(4*(width+outputs*bool(has_residual)), 128)*8)
    threads = 32
    while threads < outputs:
        threads *= 2
    return TraversalPlan(variant, threads, 1, 256, max(width+outputs*bool(has_residual), threads)*8)


COMMON = r'''
__device__ __forceinline__ int route_leaf(
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, int obj, int d, int tree) {
    int p = 0;
    while (p >= 0) {
        const float* node = nodes + 4*(noff[tree]+p);
        int f = (int)node[0];
        float v = x[(long long)obj*d+abs(f)-1];
        bool right = isnan(v) ? f > 0 : v > node[1];
        p = (int)node[2+right];
    }
    return (int)(loff[tree]-p-1);
}

__device__ __forceinline__ void finish_block(
    double value, int obj, int k, int probability, double* work, double* out) {
    if (!probability) {
        if (k < OUTPUTS) out[(long long)obj*OUTPUTS+k] = value;
        return;
    }
    if (!MULTICLASS) {
        if (k < OUTPUTS) {
            double z = exp(-fabs(value));
            double p = value >= 0 ? 1./(1.+z) : z/(1.+z);
            out[(long long)obj*OUTPUTS+k] = fmin(1.-1e-7, fmax(1e-7,p));
        }
        return;
    }
    __syncthreads();
    work[k] = k < OUTPUTS ? value : -__longlong_as_double(0x7ff0000000000000LL);
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (k < stride) work[k] = fmax(work[k],work[k+stride]);
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
    if (k < OUTPUTS) out[(long long)obj*OUTPUTS+k] = fmin(1.-1e-6,fmax(1e-6,e/work[0]));
}
'''

ARGS = r'''
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, const float* a, const int* ptr,
    const int* cols, const float* vals, const float* basis,
    const float* bias, double* out, int n, int d, int trees, int probability
'''

WARP = 'extern "C" __global__ void traversal('+ARGS+r''') {
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int obj=blockIdx.x*4+warp;
    if (obj >= n) return;  // Entire warp exits together; no block barriers.
    extern __shared__ double storage[];
    double* latent=storage+warp*(WIDTH+HAS_RESIDUAL*OUTPUTS);
    double* corrections=latent+WIDTH;
    double acc[(WIDTH+31)/32];
    for (int q=0;q<(WIDTH+31)/32;++q) acc[q]=0.;
    if (HAS_RESIDUAL)
        for (int j=lane;j<OUTPUTS;j+=32) corrections[j]=0.;
    __syncwarp();
    for (int t=0;t<trees;++t) {
        int leaf=lane==0 ? route_leaf(x,nodes,noff,loff,obj,d,t) : 0;
        leaf=__shfl_sync(0xffffffff,leaf,0);
        for (int q=0;q<(WIDTH+31)/32;++q) {
            int j=lane+32*q;
            if (j<WIDTH) acc[q]+=(double)a[(long long)leaf*WIDTH+j];
        }
        if (HAS_RESIDUAL)
            for (int s=ptr[leaf]+lane;s<ptr[leaf+1];s+=32)
                corrections[cols[s]]+=(double)vals[s];
        __syncwarp();  // Unique columns per leaf, ordered across trees.
    }
    if (LOW_RANK) {
        for (int q=0;q<(WIDTH+31)/32;++q) {
            int j=lane+32*q;
            if (j<WIDTH) latent[j]=acc[q];
        }
        __syncwarp();
    }
    double values[(OUTPUTS+31)/32];
    double maximum=-__longlong_as_double(0x7ff0000000000000LL);
    for (int q=0;q<(OUTPUTS+31)/32;++q) {
        int j=lane+32*q;
        double value=0.;
        if (j<OUTPUTS) {
            if (LOW_RANK) {
                #pragma unroll 1
                for (int r=0;r<WIDTH;++r) value+=latent[r]*(double)basis[r*OUTPUTS+j];
            } else if (j<WIDTH) value=acc[q];
            if (HAS_RESIDUAL) value+=corrections[j];
            value+=(double)bias[j];
            maximum=fmax(maximum,value);
        }
        values[q]=value;
    }
    if (probability && MULTICLASS) {
        for (int s=16;s>0;s/=2) maximum=fmax(maximum,__shfl_down_sync(0xffffffff,maximum,s));
        maximum=__shfl_sync(0xffffffff,maximum,0);
        double sum=0.;
        for (int q=0;q<(OUTPUTS+31)/32;++q) {
            int j=lane+32*q;
            values[q]=j<OUTPUTS ? exp(values[q]-maximum) : 0.;
            sum+=values[q];
        }
        for (int s=16;s>0;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);
        sum=__shfl_sync(0xffffffff,sum,0);
        for (int q=0;q<(OUTPUTS+31)/32;++q) values[q]=fmin(1.-1e-6,fmax(1e-6,values[q]/sum));
    } else if (probability) {
        for (int q=0;q<(OUTPUTS+31)/32;++q) {
            double z=exp(-fabs(values[q]));
            double p=values[q]>=0 ? 1./(1.+z) : z/(1.+z);
            values[q]=fmin(1.-1e-7,fmax(1e-7,p));
        }
    }
    for (int q=0;q<(OUTPUTS+31)/32;++q) {
        int j=lane+32*q;
        if (j<OUTPUTS) out[(long long)obj*OUTPUTS+j]=values[q];
    }
}
'''

GROUPS = 'extern "C" __global__ void traversal('+ARGS+r''') {
    int obj=blockIdx.x, k=threadIdx.x, group=k/32, lane=k%32;
    extern __shared__ double work[];
    double* latent=work+group*WIDTH;
    double* corrections=work+4*WIDTH+group*OUTPUTS;
    double acc[(WIDTH+31)/32];
    for (int q=0;q<(WIDTH+31)/32;++q) acc[q]=0.;
    if (HAS_RESIDUAL)
        for (int j=lane;j<OUTPUTS;j+=32) corrections[j]=0.;
    __syncwarp();
    for (int t=trees*group/4;t<trees*(group+1)/4;++t) {
        int leaf=lane==0 ? route_leaf(x,nodes,noff,loff,obj,d,t) : 0;
        leaf=__shfl_sync(0xffffffff,leaf,0);
        for (int q=0;q<(WIDTH+31)/32;++q) {
            int j=lane+32*q;
            if (j<WIDTH) acc[q]+=(double)a[(long long)leaf*WIDTH+j];
        }
        if (HAS_RESIDUAL)
            for (int s=ptr[leaf]+lane;s<ptr[leaf+1];s+=32)
                corrections[cols[s]]+=(double)vals[s];
        __syncwarp();
    }
    for (int q=0;q<(WIDTH+31)/32;++q) {
        int j=lane+32*q;
        if (j<WIDTH) latent[j]=acc[q];
    }
    __syncthreads();
    if (k<WIDTH) {
        double sum=0.;
        for (int g=0;g<4;++g) sum+=work[g*WIDTH+k];
        work[k]=sum;
    }
    __syncthreads();
    double value=0.;
    if (k<OUTPUTS) {
        if (LOW_RANK) {
            #pragma unroll 1
            for (int r=0;r<WIDTH;++r) value+=work[r]*(double)basis[r*OUTPUTS+k];
        } else if (k<WIDTH) value=work[k];
        if (HAS_RESIDUAL)
            for (int g=0;g<4;++g) value+=work[4*WIDTH+g*OUTPUTS+k];
        value+=(double)bias[k];
    }
    finish_block(value,obj,k,probability,work,out);
}
'''

ROUTE = r'''
extern "C" __global__ void route_tile(
    const float* x, const float* nodes, const long long* noff,
    const long long* loff, int* leaves, int n, int d, int trees) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k<n*trees) {
        int obj=k%n, tree=k/n;
        leaves[k]=route_leaf(x,nodes,noff,loff,obj,d,tree);
    }
}
'''

TILED = 'extern "C" __global__ void traversal('+ARGS+r''', const int* leaves) {
    int obj=blockIdx.x, k=threadIdx.x;
    extern __shared__ double work[];
    double* corrections=work+WIDTH;
    if (HAS_RESIDUAL && k<OUTPUTS) corrections[k]=0.;
    double acc=0.;
    __syncthreads();
    for (int t=0;t<trees;++t) {
        int leaf=leaves[t*n+obj];
        if (k<WIDTH) acc+=(double)a[(long long)leaf*WIDTH+k];
        if (HAS_RESIDUAL)
            for (int s=ptr[leaf]+k;s<ptr[leaf+1];s+=blockDim.x)
                corrections[cols[s]]+=(double)vals[s];
        if (HAS_RESIDUAL) __syncthreads();
    }
    double value=k<WIDTH ? acc : 0.;
    if (LOW_RANK) {
        if (k<WIDTH) work[k]=acc;
        __syncthreads();
        value=0.;
        if (k<OUTPUTS) {
            #pragma unroll 1
            for (int r=0;r<WIDTH;++r) value+=work[r]*(double)basis[r*OUTPUTS+k];
        }
    }
    if (k<OUTPUTS) {
        if (HAS_RESIDUAL) value+=corrections[k];
        value+=(double)bias[k];
    }
    finish_block(value,obj,k,probability,work,out);
}
'''


class TraversalLeafGpu(PreparedLeafGpu):
    def __init__(self, model, variant):
        import cupy as cp
        self.plan = traversal_plan(variant, model.base.coefficients.shape[1],
                                   len(model.base.bias), bool(len(model.values)))
        super().__init__(model)
        if len(model.base.coefficients) > 2**31-1:
            raise ValueError('leaf indices exceed int32')
        if self.representation == 'low_rank':
            self._basis = cp.ascontiguousarray(self._basis.T)
        self.threads, self.shared_bytes = self.plan.threads, self.plan.shared_bytes
        defines = dict(WIDTH=self.width, OUTPUTS=self.outputs,
                       LOW_RANK=int(self.representation == 'low_rank'),
                       HAS_RESIDUAL=int(self.has_residual), MULTICLASS=int(self.task == 'multiclass'))
        header = '\n'.join(f'#define {k} {v}' for k,v in defines.items())+'\n'+COMMON
        self.kernel = cp.RawKernel(header+{'warp':WARP,'groups':GROUPS,'tiled':TILED}[variant],
                                   'traversal',options=('--fmad=false',))
        self.route_kernel = cp.RawKernel(header+ROUTE,'route_tile',options=('--fmad=false',)) if variant == 'tiled' else None
        started = time.perf_counter()
        self.kernel.compile()
        if self.route_kernel is not None:
            self.route_kernel.compile()
        self.compilation_ms = (time.perf_counter()-started)*1000
        self.kernel_resources = dict(main=dict(self.kernel.attributes), threads=self.threads,
            dynamic_shared_bytes=self.shared_bytes, variant=variant, tile_rows=self.plan.tile_rows)
        if self.route_kernel is not None:
            self.kernel_resources['route'] = dict(self.route_kernel.attributes)

    def launch_into(self, x, out, probability=True, workspace=None):
        """Prevalidated, preallocated launch for separately labelled stream timing."""
        import numpy as np
        n = len(x)
        if not n:
            return
        def args(features, result, rows):
            return (features,self._nodes,self._node_offsets,self._leaf_offsets,self._coefficients,
                    self._indptr,self._columns,self._values,self._basis,self._bias,result,
                    *(np.int32(v) for v in (rows,self.n_features,self.n_trees,probability)))
        if self.plan.variant != 'tiled':
            grid = (n+3)//4 if self.plan.variant == 'warp' else n
            self.kernel((grid,),(self.threads,),args(x,out,n),shared_mem=self.shared_bytes)
            return
        if workspace is None:
            raise ValueError('tiled launch requires its bounded index workspace')
        for start in range(0,n,self.plan.tile_rows):
            features,result = x[start:start+self.plan.tile_rows],out[start:start+self.plan.tile_rows]
            rows = len(features)
            self.route_kernel(((rows*self.n_trees+127)//128,),(128,),
                (features,self._nodes,self._node_offsets,self._leaf_offsets,workspace,
                 *(np.int32(v) for v in (rows,self.n_features,self.n_trees))))
            self.kernel((rows,),(self.threads,),(*args(features,result,rows),workspace),
                        shared_mem=self.shared_bytes)

    def workspace(self, rows):
        import cupy as cp
        return cp.empty((min(rows,self.plan.tile_rows),self.n_trees),cp.int32) if self.plan.variant == 'tiled' and rows else None

    def _request(self, x, probability):
        import cupy as cp
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features or not x.flags.c_contiguous:
            raise ValueError('resident features must be contiguous float32 of correct width')
        if bool(cp.isinf(x).any()):
            raise ValueError('invalid features')
        out = cp.empty((len(x),self.outputs),cp.float64)
        self.launch_into(x,out,probability,self.workspace(len(x)))
        return out

    def raw_device(self, x):
        return self._request(x,False)

    def predict_device(self, x):
        return self._request(x,True)


def baseline_launch_into(model, x, out):
    """Same unchanged unified CUDA kernel, excluding request validation/allocation."""
    import numpy as np
    if len(x):
        model.kernel((len(x),),(model.threads,),
            (x,model._nodes,model._node_offsets,model._leaf_offsets,model._coefficients,
             model._indptr,model._columns,model._values,model._basis,model._bias,out,
             *(np.int32(v) for v in (len(x),model.n_features,model.n_trees,True))),
            shared_mem=model.shared_bytes)
