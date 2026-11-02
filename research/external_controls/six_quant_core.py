"""Physical FP16 / per-coordinate INT8 packed coefficients; FP32 sparse L."""
from dataclasses import dataclass
import numpy as np
from packed_serving_core import prepare_snapshot, PackedPlan


def quantize_columns(values, codec, scales=None):
    x=np.asarray(values,np.float32)
    if codec=='fp16':
        q=x.astype(np.float16)
        if not np.isfinite(q).all():raise ValueError('FP16 overflow')
        return q,np.empty(0,np.float32)
    if codec!='int8':raise ValueError('unsupported codec')
    s=np.maximum(np.max(np.abs(x),axis=0)/127.,np.finfo(np.float32).tiny).astype(np.float32) if scales is None else np.asarray(scales,np.float32)
    if s.shape!=(x.shape[1],) or not np.isfinite(s).all() or np.any(s<=0):raise ValueError('invalid scales')
    return np.clip(np.rint(x/s),-127,127).astype(np.int8),s


@dataclass(frozen=True)
class QuantSnapshot:
    arrays:dict
    codec:str
    task:str
    features:int
    outputs:int
    trees:int
    max_width:int
    has_residual:bool

    @property
    def resident_bytes(self):return int(sum(a.nbytes for a in self.arrays.values()))

    def save(self,path):
        np.savez(path,**self.arrays,codec=np.array(self.codec),task=np.array(self.task),features=np.array(self.features),
                 outputs=np.array(self.outputs),trees=np.array(self.trees),max_width=np.array(self.max_width),has_residual=np.array(self.has_residual))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            keys=('codec','task','features','outputs','trees','max_width','has_residual')
            meta={k:f[k].item() for k in keys};arrays={k:f[k].copy() for k in f.files if k not in keys}
        return cls(arrays,**meta)

    def raw(self,x):
        x=np.asarray(x,np.float32);a=self.arrays;c=self.outputs
        out=np.broadcast_to(a['bias'],(len(x),c)).astype(float,copy=True)
        for block,m in enumerate(a['meta']):
            begin,trees,width,low,ao,bo,po,has=map(int,m);sa,sb=map(int,a['qmeta'][block])
            latent=np.zeros((len(x),width));residual=np.zeros((len(x),c))
            for t in range(begin,begin+trees):
                p=np.zeros(len(x),np.int64)
                while np.any(p>=0):
                    rows=np.flatnonzero(p>=0);node=a['nodes'][a['noff'][t]+p[rows]];f=node[:,0].astype(int)
                    v=x[rows,np.abs(f)-1];right=np.where(np.isnan(v),f>0,v>node[:,1]);p[rows]=node[np.arange(len(rows)),2+right.astype(int)].astype(int)
                leaf=a['loff'][t]-p-1;indices=ao+leaf[:,None]*width+np.arange(width)[None,:]
                z=a['coefficients'][indices].astype(float)
                if self.codec=='int8':z*=a['ascales'][sa:sa+width]
                latent+=z
                if has:
                    for row,l in enumerate(leaf):
                        u,v=a['ptr'][po+l:po+l+2];residual[row,a['columns'][u:v]]+=a['values'][u:v]
            if low:
                basis=a['basis'][bo:bo+width*c].reshape(width,c).astype(float)
                if self.codec=='int8':basis*=a['bscales'][sb:sb+width,None]
                out+=latent@basis
            else:out[:,:width]+=latent
            out+=residual
        return out


def prepare_quant(model,shifts,codec,fixed_a=None):
    snap=prepare_snapshot(model,shifts,'aos');arrays={k:v.copy() for k,v in snap.arrays.items()}
    aq=[];bq=[];ascales=[];bscales=[];meta=[]
    for m in arrays['meta']:
        begin,trees,w,low,ao,bo,po,has=map(int,m)
        # A section ends at the next block offset (or full payload end).
        i=len(meta);end=int(arrays['meta'][i+1,4]) if i+1<len(arrays['meta']) else len(arrays['coefficients'])
        fixed=fixed_a if low and codec=='int8' and fixed_a is not None and len(fixed_a)==w else None
        q,s=quantize_columns(arrays['coefficients'][ao:end].reshape(-1,w),codec,fixed)
        meta.append([len(ascales),len(bscales)]);aq.append(q.ravel());ascales.extend(s)
        if low:
            q,s=quantize_columns(arrays['basis'][bo:bo+w*snap.outputs].reshape(w,snap.outputs).T,codec)
            bq.append(q.T.copy().ravel());bscales.extend(s)
    dtype=np.float16 if codec=='fp16' else np.int8
    arrays.update(coefficients=np.concatenate(aq).astype(dtype),basis=np.concatenate(bq).astype(dtype) if bq else np.empty(0,dtype),
                  qmeta=np.asarray(meta,np.int64),ascales=np.asarray(ascales,np.float32),bscales=np.asarray(bscales,np.float32))
    return QuantSnapshot(arrays,codec,snap.task,snap.features,snap.outputs,snap.trees,snap.max_width,snap.has_residual)
