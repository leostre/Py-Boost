"""Quantized packed execution. A/B stay compressed in device global memory."""
import numpy as np
from packed_serving_gpu import SOURCE,PackedGpu
from packed_serving_core import PackedPlan

HALF=r'''
__device__ __forceinline__ double readq(const unsigned short* p,long long i,const float* scales,long long s) {
 unsigned int h=p[i],sign=(h&32768u)<<16,ex=(h>>10)&31u,m=h&1023u,bits;
 if(ex==0u) {
  if(m==0u)bits=sign;
  else {int e=-14;while((m&1024u)==0u){m<<=1;--e;}m&=1023u;bits=sign|((unsigned int)(e+127)<<23)|(m<<13);}
 } else if(ex==31u)bits=sign|0x7f800000u|(m<<13);
 else bits=sign|((ex+112u)<<23)|(m<<13);
 return (double)__uint_as_float(bits);
}
'''
INT8=r'''
__device__ __forceinline__ double readq(const signed char* p,long long i,const float* scales,long long s) {
 return (double)p[i]*(double)scales[s];
}
'''


class QuantPackedGpu(PackedGpu):
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
        defines='\n'.join(f'#define {name} {value}' for name,value in dict(C=self.outputs,GROUPS=groups,MAX_WIDTH=snapshot.max_width,
             HAS_RESIDUAL=int(snapshot.has_residual),SOA=0,MULTICLASS=int(snapshot.task=='multiclass')).items())
        self.kernel=cp.RawKernel(defines+'\n'+(HALF if snapshot.codec=='fp16' else INT8)+'\n'+source,'packed',options=('--fmad=false',))

    def launch_into(self,x,out,ml_labels,mc_labels,raw=False):
        import cupy as cp
        if x.ndim!=2 or x.shape[1]!=self.features or x.dtype!=cp.float32 or not x.flags.c_contiguous:raise ValueError('invalid resident features')
        if out.shape!=(len(x),self.outputs) or out.dtype!=cp.float64:raise ValueError('invalid output buffer')
        if len(x):
            order=('nodes','feature','threshold','left','right','meta','noff','loff','coefficients','basis','ptr','columns','values','bias','shifts','qmeta','ascales','bscales')
            self.kernel((len(x),),(self.plan.threads,),(x,*[self.arrays[k] for k in order],out,ml_labels,mc_labels,
                         *(np.int32(v) for v in (len(x),self.features,self.blocks,int(raw))),np.float64(self.temperature)),shared_mem=self.plan.shared_bytes)
        return out
