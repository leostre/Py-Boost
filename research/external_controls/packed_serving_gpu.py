"""One CUDA launch: all bases, tree routing, sparse gather, decode and controls."""
import numpy as np
from packed_serving_core import PackedPlan,prepare_snapshot


SOURCE=r'''
extern "C" __global__ void packed(
 const float* x,const float* nodes,const int* feature,const float* threshold,const int* left,const int* right,
 const long long* meta,const long long* noff,const long long* loff,const float* a,const float* basis,
 const int* ptr,const int* cols,const float* vals,const float* bias,const float* shifts,
 double* out,unsigned char* ml_labels,int* mc_labels,int n,int d,int blocks,int raw_mode,double temperature) {
 int obj=blockIdx.x,k=threadIdx.x,group=k/32,lane=k%32;
 extern __shared__ double shared[];
 double* total=shared;
 double* latent=shared+C;
 double* correction=latent+GROUPS*MAX_WIDTH;
 double* scratch=correction+GROUPS*C*HAS_RESIDUAL;
 for(int j=k;j<C;j+=blockDim.x) total[j]=(double)bias[j];
 __syncthreads();
 for(int block=0;block<blocks;++block) {
  const long long* m=meta+8*block;
  int begin=(int)m[0],trees=(int)m[1],width=(int)m[2],low=(int)m[3],has=(int)m[7];
  long long aoff=m[4],boff=m[5],poff=m[6];
  double acc[4]={0.,0.,0.,0.};
  for(int j=lane;j<width;j+=32) latent[group*MAX_WIDTH+j]=0.;
  if(has) for(int j=lane;j<C;j+=32) correction[group*C+j]=0.;
  __syncthreads();
  for(int t=begin+trees*group/GROUPS;t<begin+trees*(group+1)/GROUPS;++t) {
   int leaf=0;
   if(lane==0) {
    int p=0;
    while(p>=0) {
     long long idx=noff[t]+p;
     int f;float cut;int l,r;
     if(SOA) {f=feature[idx];cut=threshold[idx];l=left[idx];r=right[idx];}
     else {const float* node=nodes+4*idx;f=(int)node[0];cut=node[1];l=(int)node[2];r=(int)node[3];}
     float v=x[(long long)obj*d+abs(f)-1];
     bool go_right=isnan(v)?f>0:v>cut;
     p=go_right?r:l;
    }
    leaf=(int)(loff[t]-p-1);
   }
   leaf=__shfl_sync(0xffffffff,leaf,0);
   if(width<=128) {
    #pragma unroll
    for(int q=0;q<4;++q) {int j=lane+32*q;if(j<width)acc[q]+=(double)a[aoff+(long long)leaf*width+j];}
   } else {
    for(int j=lane;j<width;j+=32) latent[group*MAX_WIDTH+j]+=(double)a[aoff+(long long)leaf*width+j];
   }
   if(has) for(int s=ptr[poff+leaf]+lane;s<ptr[poff+leaf+1];s+=32) correction[group*C+cols[s]]+=(double)vals[s];
   __syncwarp();
  }
  if(width<=128) {
   #pragma unroll
   for(int q=0;q<4;++q) {int j=lane+32*q;if(j<width)latent[group*MAX_WIDTH+j]=acc[q];}
  }
  __syncthreads();
  for(int j=k;j<width;j+=blockDim.x) {
   double value=0.;for(int g=0;g<GROUPS;++g)value+=latent[g*MAX_WIDTH+j];
   latent[j]=value;
  }
  __syncthreads();
  for(int j=k;j<C;j+=blockDim.x) {
   double value=0.;
   if(low) {for(int r=0;r<width;++r)value+=latent[r]*(double)basis[boff+(long long)r*C+j];}
   else if(j<width)value=latent[j];
   if(has)for(int g=0;g<GROUPS;++g)value+=correction[g*C+j];
   total[j]+=value;
  }
  __syncthreads();
 }
 if(raw_mode) {
  for(int j=k;j<C;j+=blockDim.x)out[(long long)obj*C+j]=total[j];
  return;
 }
 if(MULTICLASS) {
  if(k==0) {
   double maximum=-__longlong_as_double(0x7ff0000000000000LL),decision=maximum;int label=0;
   for(int j=0;j<C;++j) {
    double z=total[j]/temperature;maximum=fmax(maximum,z);
    double q=z-(double)shifts[j];if(q>decision){decision=q;label=j;}
   }
   double sum=0.;for(int j=0;j<C;++j)sum+=exp(total[j]/temperature-maximum);
   scratch[0]=maximum;scratch[1]=sum;mc_labels[obj]=label;
  }
  __syncthreads();
  for(int j=k;j<C;j+=blockDim.x)out[(long long)obj*C+j]=fmin(1.-1e-7,fmax(1e-7,exp(total[j]/temperature-scratch[0])/scratch[1]));
 } else {
  for(int j=k;j<C;j+=blockDim.x) {
   double z=total[j]/temperature,e=exp(-fabs(z)),p=z>=0?1./(1.+e):e/(1.+e);
   out[(long long)obj*C+j]=fmin(1.-1e-7,fmax(1e-7,p));ml_labels[(long long)obj*C+j]=(unsigned char)(z>=(double)shifts[j]);
  }
 }
}
'''


class PackedGpu:
    def __init__(self,model,temperature,shifts,groups=4,layout='aos'):
        import cupy as cp
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('positive finite temperature required')
        snapshot=prepare_snapshot(model,shifts,layout)
        self.plan=PackedPlan(groups,layout,snapshot.outputs,snapshot.max_width,snapshot.has_residual)
        self.task=snapshot.task;self.outputs=snapshot.outputs;self.features=snapshot.features;self.temperature=float(temperature)
        self.blocks=len(snapshot.arrays['meta']);self.arrays={k:cp.array(v,copy=True) for k,v in snapshot.arrays.items()}
        self.resident_device_array_bytes=sum(a.nbytes for a in self.arrays.values())
        defines='\n'.join(f'#define {name} {value}' for name,value in dict(C=self.outputs,GROUPS=groups,MAX_WIDTH=snapshot.max_width,
             HAS_RESIDUAL=int(snapshot.has_residual),SOA=int(layout=='soa'),MULTICLASS=int(snapshot.task=='multiclass')).items())
        self.kernel=cp.RawKernel(defines+'\n'+SOURCE,'packed',options=('--fmad=false',))

    def buffers(self,n):
        import cupy as cp
        if n<0:raise ValueError('negative batch size')
        return (cp.empty((n,self.outputs),cp.float64),
                cp.empty((n,self.outputs),cp.uint8) if self.task=='multilabel' else cp.empty(0,cp.uint8),
                cp.empty(n,cp.int32) if self.task=='multiclass' else cp.empty(0,cp.int32))

    def launch_into(self,x,out,ml_labels,mc_labels,raw=False):
        import cupy as cp
        if x.ndim!=2 or x.shape[1]!=self.features or x.dtype!=cp.float32 or not x.flags.c_contiguous:raise ValueError('invalid resident features')
        if out.shape!=(len(x),self.outputs) or out.dtype!=cp.float64 or not out.flags.c_contiguous:raise ValueError('invalid output buffer')
        if not raw and ((self.task=='multilabel' and (ml_labels.shape!=out.shape or ml_labels.dtype!=cp.uint8)) or
                        (self.task=='multiclass' and (mc_labels.shape!=(len(x),) or mc_labels.dtype!=cp.int32))):raise ValueError('invalid label buffer')
        if len(x):
            order=('nodes','feature','threshold','left','right','meta','noff','loff','coefficients','basis','ptr','columns','values','bias','shifts')
            self.kernel((len(x),),(self.plan.threads,),(x,*[self.arrays[k] for k in order],out,ml_labels,mc_labels,
                         *(np.int32(v) for v in (len(x),self.features,self.blocks,int(raw))),np.float64(self.temperature)),shared_mem=self.plan.shared_bytes)
        return out

    def request(self,features):
        import cupy as cp
        x=np.asarray(features,np.float32)
        if x.ndim!=2 or x.shape[1]!=self.features or np.isinf(x).any():raise ValueError('invalid features; NaN routing is supported')
        device=cp.asarray(x,order='C');out,ml,mc=self.buffers(len(x));self.launch_into(device,out,ml,mc)
        return out.get(),ml.get().view(np.bool_) if self.task=='multilabel' else mc.get()
