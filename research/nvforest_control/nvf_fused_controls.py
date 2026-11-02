"""Equivalent fused probability+threshold boundary around the unmodified engine."""
import numpy as np

SOURCE=r'''
extern "C" __global__ void finish(const INPUT* raw,const float* shifts,double* out,unsigned char* labels,
                                  long long size,int c,double temperature){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
 double z=(double)raw[i]/temperature,e=exp(-fabs(z)),p=z>=0?1./(1.+e):e/(1.+e);
 out[i]=fmin(1.-1e-7,fmax(1e-7,p));labels[i]=(unsigned char)(z>=(double)shifts[i%c]);
}
'''

class OutputControls:
    def __init__(self,shifts,temperature):
        import cupy as cp
        self.shifts=shifts;self.temperature=float(temperature);self.outputs=len(shifts)
        self.kernels={dtype:cp.RawKernel(SOURCE.replace('INPUT',ctype),'finish',options=('--fmad=false',))
                      for dtype,ctype in ((cp.float32,'float'),(cp.float64,'double'))}

    def __call__(self,raw):
        import cupy as cp
        if raw.dtype.type not in self.kernels:raise ValueError('unexpected nvForest output dtype')
        raw=cp.ascontiguousarray(raw).reshape(-1,self.outputs);out=cp.empty(raw.shape,cp.float64);labels=cp.empty(raw.shape,cp.uint8)
        if raw.size:self.kernels[raw.dtype.type](((raw.size+255)//256,),(256,),(raw,self.shifts,out,labels,np.int64(raw.size),np.int32(self.outputs),np.float64(self.temperature)))
        return out,labels
