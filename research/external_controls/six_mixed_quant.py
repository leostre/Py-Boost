"""Per-block byte packing permits independent A, B, warm-start ablations."""
import numpy as np
from six_quant_core import QuantSnapshot,quantize_columns
from packed_serving_core import prepare_snapshot

DTYPES=(np.float32,np.float16,np.int8)


class MixedSnapshot(QuantSnapshot):
    def raw(self,x):
        x=np.asarray(x,np.float32);a=self.arrays;c=self.outputs
        out=np.broadcast_to(a['bias'],(len(x),c)).astype(float,copy=True)
        for block,m in enumerate(a['meta']):
            begin,trees,w,low,ao,bo,po,has=map(int,m);sa,sb=map(int,a['qmeta'][block]);fa,fb=map(int,a['qformat'][block])
            end=ao+int(a['qlength'][block,0])*np.dtype(DTYPES[fa]).itemsize
            av=a['coefficients'][ao:end].view(DTYPES[fa]).reshape(-1,w).astype(float)
            if fa==2:av*=a['ascales'][sa:sa+w]
            latent=np.zeros((len(x),w));residual=np.zeros((len(x),c))
            for t in range(begin,begin+trees):
                p=np.zeros(len(x),np.int64)
                while np.any(p>=0):
                    rows=np.flatnonzero(p>=0);node=a['nodes'][a['noff'][t]+p[rows]];f=node[:,0].astype(int);v=x[rows,np.abs(f)-1]
                    right=np.where(np.isnan(v),f>0,v>node[:,1]);p[rows]=node[np.arange(len(rows)),2+right.astype(int)].astype(int)
                leaves=a['loff'][t]-p-1;latent+=av[leaves]
                if has:
                    for row,l in enumerate(leaves):
                        u,v=a['ptr'][po+l:po+l+2];residual[row,a['columns'][u:v]]+=a['values'][u:v]
            if low:
                size=w*c*np.dtype(DTYPES[fb]).itemsize;b=a['basis'][bo:bo+size].view(DTYPES[fb]).reshape(w,c).astype(float)
                if fb==2:b*=a['bscales'][sb:sb+w,None]
                out+=latent@b
            else:out[:,:w]+=latent
            out+=residual
        return out


def prepare_mixed(model,shifts,recipe):
    codec,scope=recipe.split('-');assert scope in ('a_only','b_only','warm_only','ab_only')
    snap=prepare_snapshot(model,shifts);arrays={k:v.copy() for k,v in snap.arrays.items()};original=arrays['meta'].copy()
    aa=[];bb=[];sa=[];sb=[];qm=[];qf=[];ql=[];apos=bpos=0
    for i,m in enumerate(original):
        begin,trees,w,low,ao,bo,po,has=map(int,m);end=int(original[i+1,4]) if i+1<len(original) else len(arrays['coefficients'])
        av=arrays['coefficients'][ao:end].reshape(-1,w);bv=arrays['basis'][bo:bo+w*snap.outputs].reshape(w,snap.outputs).T if low else np.empty((0,w),np.float32)
        qa=bool((low and scope in ('a_only','ab_only')) or (not low and scope=='warm_only'));qb=bool(low and scope in ('b_only','ab_only'))
        aq,ass=quantize_columns(av,codec) if qa else (av,np.empty(0,np.float32))
        bq,bss=quantize_columns(bv,codec) if qb else (bv,np.empty(0,np.float32))
        code=1 if codec=='fp16' else 2;qm.append([len(sa),len(sb)]);qf.append([code if qa else 0,code if qb else 0]);sa.extend(ass);sb.extend(bss)
        ql.append([aq.size,bq.size]);abytes=aq.ravel().view(np.uint8).copy();bbytes=bq.T.copy().ravel().view(np.uint8).copy()
        abytes=np.pad(abytes,(0,(-len(abytes))%4));bbytes=np.pad(bbytes,(0,(-len(bbytes))%4))
        arrays['meta'][i,4:6]=[apos,bpos];apos+=len(abytes);bpos+=len(bbytes);aa.append(abytes);bb.append(bbytes)
    arrays.update(coefficients=np.concatenate(aa),basis=np.concatenate(bb),qmeta=np.asarray(qm,np.int64),qformat=np.asarray(qf,np.int32),qlength=np.asarray(ql,np.int64),ascales=np.asarray(sa,np.float32),bscales=np.asarray(sb,np.float32))
    return MixedSnapshot(arrays,recipe,snap.task,snap.features,snap.outputs,snap.trees,snap.max_width,snap.has_residual)


class MixedGpu:
    def __new__(cls,snapshot,temperature=1.,groups=4):
        # Reuse the already tested half decoder and launch/buffer validation.
        from six_quant_gpu import QuantPackedGpu,HALF
        from packed_serving_gpu import SOURCE
        from packed_serving_core import PackedPlan
        import cupy as cp
        obj=object.__new__(QuantPackedGpu);obj.plan=PackedPlan(groups,'aos',snapshot.outputs,snapshot.max_width,snapshot.has_residual)
        obj.task=snapshot.task;obj.outputs=snapshot.outputs;obj.features=snapshot.features;obj.temperature=float(temperature);obj.blocks=len(snapshot.arrays['meta'])
        obj.arrays={k:cp.array(v,copy=True) for k,v in snapshot.arrays.items()};obj.resident_device_array_bytes=sum(a.nbytes for a in obj.arrays.values())
        read=HALF+'''\n__device__ __forceinline__ double readmixed(const unsigned char* p,long long i,int f,const float* s,long long j){
          if(f==0)return (double)((const float*)p)[i];
          if(f==1)return readq((const unsigned short*)p,i,s,j);
          return (double)((const signed char*)p)[i]*(double)s[j];}\n'''
        source=SOURCE.replace('const float* a,const float* basis,','const unsigned char* a,const unsigned char* basis,')
        source=source.replace('double* out,unsigned char* ml_labels','const long long* qmeta,const float* ascales,const float* bscales,const int* qformat,\n double* out,unsigned char* ml_labels')
        source=source.replace('(double)a[aoff+(long long)leaf*width+j]','readmixed(a+aoff,(long long)leaf*width+j,qformat[2*block],ascales,qmeta[2*block]+j)')
        source=source.replace('(double)basis[boff+(long long)r*C+j]','readmixed(basis+boff,(long long)r*C+j,qformat[2*block+1],bscales,qmeta[2*block+1]+r)')
        defines='\n'.join(f'#define {k} {v}' for k,v in dict(C=obj.outputs,GROUPS=groups,MAX_WIDTH=snapshot.max_width,HAS_RESIDUAL=int(snapshot.has_residual),SOA=0,MULTICLASS=int(snapshot.task=='multiclass')).items())
        obj.kernel=cp.RawKernel(defines+'\n'+read+source,'packed',options=('--fmad=false',))
        # Kernel extra argument is inserted by the narrowly scoped launch shell.
        obj.launch_into=lambda x,out,ml,mc,raw=False:launch(obj,x,out,ml,mc,raw)
        return obj


def launch(obj,x,out,ml,mc,raw=False):
    if len(x):
        order=('nodes','feature','threshold','left','right','meta','noff','loff','coefficients','basis','ptr','columns','values','bias','shifts','qmeta','ascales','bscales','qformat')
        obj.kernel((len(x),),(obj.plan.threads,),(x,*[obj.arrays[k] for k in order],out,ml,mc,*(np.int32(v) for v in (len(x),obj.features,obj.blocks,int(raw))),np.float64(obj.temperature)),shared_mem=obj.plan.shared_bytes)
    return out
