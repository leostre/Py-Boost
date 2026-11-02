"""One shared output; cooperative tree gather plus beta=1 GEMM per basis."""
import numpy as np
from packed_serving_gpu import PackedGpu
from unified_serving_core import coalesce
from experiments.fused_leaf_gpu import SOURCE as PARENT

OLD='''    if (k < width) latent[obj*width+k] = acc;
    if (has_residual && k < outputs) residual[obj*outputs+k] = corrections[k];'''
NEW='''    if (k < width) {
        if (low_rank) latent[obj*width+k] = acc;
        else residual[obj*outputs+k] += acc;
    }
    if (has_residual && k < outputs) residual[obj*outputs+k] += corrections[k];'''
if PARENT.count(OLD)!=1:raise RuntimeError('pinned gather source changed')
SOURCE=PARENT.replace('int outputs, int has_residual) {','int outputs, int has_residual, int low_rank) {').replace(OLD,NEW)
FINISH=r'''
extern "C" __global__ void controls(double* out,const float* shifts,unsigned char* labels,long long size,int c,double temperature){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
 double z=out[i]/temperature,e=exp(-fabs(z)),p=z>=0?1./(1.+e):e/(1.+e);
 out[i]=fmin(1.-1e-7,fmax(1e-7,p));labels[i]=(unsigned char)(z>=(double)shifts[i%c]);
}
'''


class GatherGemmGpu(PackedGpu):
    def __init__(self,model,temperature,shifts):
        import cupy as cp
        from experiments.prepared_leaf_gpu import PreparedLeafGpu
        if model.task!='multilabel':raise ValueError('this comparison is multilabel only')
        self.model=coalesce(model);self.task=model.task;self.outputs=len(model.bias);self.features=model.n_features;self.temperature=float(temperature)
        self.bias=cp.array(model.bias,copy=True);self.shifts=cp.array(shifts,dtype=cp.float32,copy=True);self.blocks=[]
        for block in self.model.blocks:
            b=PreparedLeafGpu(block);b._decoder=cp.array(b._basis,dtype=cp.float64,order='C',copy=True);del b._basis
            self.blocks.append(b)
        self.gather=cp.RawKernel(SOURCE,'gather',options=('--fmad=false',));self.finish=cp.RawKernel(FINISH,'controls',options=('--fmad=false',))
        arrays=[self.bias,self.shifts]+[a for b in self.blocks for a in vars(b).values() if isinstance(a,cp.ndarray)]
        self.resident_device_array_bytes=sum({int(a.data.ptr):a.nbytes for a in arrays}.values())

    def launch_into(self,x,out,ml_labels,mc_labels,raw=False):
        import cupy as cp
        from cupy.cublas import gemm
        if x.dtype!=cp.float32 or x.ndim!=2 or x.shape[1]!=self.features:raise ValueError('invalid features')
        out[:]=self.bias
        for b in self.blocks:
            low=b.representation=='low_rank';latent=cp.empty((len(x),b.width),cp.float64) if low else out
            if len(x):
                self.gather((len(x),),(b.threads,),(x,b._nodes,b._node_offsets,b._leaf_offsets,b._coefficients,b._indptr,b._columns,b._values,
                     latent,out,*(np.int32(v) for v in (len(x),self.features,b.n_trees,b.width,self.outputs,b.has_residual,low))),
                     shared_mem=self.outputs*8 if b.has_residual else 0)
                if low:
                    result=gemm('N','T',latent,b._decoder,out=out,alpha=1.,beta=1.)
                    if result.data.ptr!=out.data.ptr:raise RuntimeError('GEMM copied accumulation buffer')
            del latent
        if len(x) and not raw:self.finish(((out.size+255)//256,),(256,),(out,self.shifts,ml_labels,np.int64(out.size),np.int32(self.outputs),np.float64(self.temperature)))
        return out
