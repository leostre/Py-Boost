"""Hardware FP16 conversion; fixed INT8 column scales outside inner loops."""
import numpy as np
from six_quant_gpu import QuantPackedGpu
from packed_serving_core import PackedPlan
from packed_serving_gpu import SOURCE

HALF=r'''
__device__ __forceinline__ double readq(const unsigned short* p,long long i,const float* scales,long long s){
 float v;asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(p[i]));return (double)v;
}
'''
INT8=r'''
__device__ __forceinline__ double readq(const signed char* p,long long i,const float* scales,long long s){return (double)p[i];}
'''


class FastQuantPackedGpu(QuantPackedGpu):
    def __init__(self,snapshot,temperature=1.,groups=4):
        import cupy as cp
        self.plan=PackedPlan(groups,'aos',snapshot.outputs,snapshot.max_width,snapshot.has_residual)
        self.task=snapshot.task;self.outputs=snapshot.outputs;self.features=snapshot.features;self.temperature=float(temperature)
        self.blocks=len(snapshot.arrays['meta']);self.arrays={k:cp.array(v,copy=True) for k,v in snapshot.arrays.items()}
        self.resident_device_array_bytes=sum(a.nbytes for a in self.arrays.values())
        qtype='unsigned short' if snapshot.codec=='fp16' else 'signed char'
        source=SOURCE.replace('const float* a,const float* basis,',f'const {qtype}* a,const {qtype}* basis,')
        source=source.replace('double* out,unsigned char* ml_labels','const long long* qmeta,const float* ascales,const float* bscales,\n double* out,unsigned char* ml_labels')
        source=source.replace('(double)a[aoff+(long long)leaf*width+j]','readq(a,aoff+(long long)leaf*width+j,ascales,qmeta[2*block]+j)')
        source=source.replace('(double)basis[boff+(long long)r*C+j]','readq(basis,boff+(long long)r*C+j,bscales,qmeta[2*block+1]+r)')
        if snapshot.codec=='int8':
            assert source.count('latent[j]=value;')==1
            source=source.replace('latent[j]=value;','latent[j]=value*(double)ascales[qmeta[2*block]+j]*(low?(double)bscales[qmeta[2*block+1]+j]:1.);')
        defines='\n'.join(f'#define {k} {v}' for k,v in dict(C=self.outputs,GROUPS=groups,MAX_WIDTH=snapshot.max_width,HAS_RESIDUAL=int(snapshot.has_residual),SOA=0,MULTICLASS=int(snapshot.task=='multiclass')).items())
        self.kernel=cp.RawKernel(defines+'\n'+(HALF if snapshot.codec=='fp16' else INT8)+'\n'+source,'packed',options=('--fmad=false',))


def half_preflight():
    import cupy as cp
    bits=np.arange(65536,dtype=np.uint16);values=bits.view(np.float16);bits=bits[np.isfinite(values)];expected=bits.view(np.float16).astype(float)
    kernel=cp.RawKernel(HALF+'\nextern "C" __global__ void check_half(const unsigned short* x,double* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=readq(x,i,(const float*)0,0);}','check_half')
    x=cp.asarray(bits);y=cp.empty(len(bits),cp.float64);kernel(((len(bits)+255)//256,),(256,),(x,y,np.int32(len(bits))))
    actual=y.get();np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))
    return dict(finite_half_patterns=len(bits),exact=True,negative_zero_preserved=True)
