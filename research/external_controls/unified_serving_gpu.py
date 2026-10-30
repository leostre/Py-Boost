"""Equally optimized full/compact blocks, one output plus one block workspace."""
import numpy as np
from unified_serving_core import coalesce


class BlockGpu:
    def __init__(self,model,temperature,shifts):
        import cupy as cp
        from experiments.traversal_leaf_gpu import TraversalLeafGpu
        from experiments.prepared_leaf_gpu import PreparedLeafGpu
        self.model=coalesce(model);self.task=model.task;self.outputs=len(model.bias);self.temperature=float(temperature)
        self.bias=cp.asarray(model.bias);self.shifts=cp.asarray(shifts,dtype=cp.float32)
        self.blocks=[TraversalLeafGpu(b,'groups') if self.outputs<=128 else PreparedLeafGpu(b) for b in self.model.blocks]
        self.backend='groups' if self.outputs<=128 else 'prepared_wide_fallback'
        zero=cp.zeros(self.outputs,cp.float32)
        for b in self.blocks:b._bias=zero;b._decision_shifts=zero
        if len(self.blocks)==1:self.blocks[0]._bias=self.bias
        arrays=[self.bias,self.shifts]+[a for b in self.blocks for a in vars(b).values() if isinstance(a,cp.ndarray)]
        seen={};
        for a in arrays:seen[int(a.data.ptr)]=max(seen.get(int(a.data.ptr),0),a.nbytes)
        self.resident_device_array_bytes=sum(seen.values())

    def raw_into(self,x,out,work):
        import cupy as cp
        if len(self.blocks)>1:out[:]=self.bias
        for b in self.blocks:
            target=out if len(self.blocks)==1 else work
            if self.backend=='groups':b.launch_into(x,target,probability=False)
            else:target[:]=b.raw_device(x)
            if len(self.blocks)>1:cp.add(out,target,out=out)
        return out

    def request(self,features):
        import cupy as cp
        features=np.asarray(features,dtype=np.float32)
        if features.ndim!=2 or features.shape[1]!=self.model.n_features or np.isinf(features).any():
            raise ValueError('invalid feature matrix; NaN routing remains supported')
        x=cp.asarray(features,dtype=cp.float32,order='C');n=len(x)
        out=cp.empty((n,self.outputs),cp.float64)
        work=cp.empty_like(out) if len(self.blocks)>1 else out
        self.raw_into(x,out,work)
        out/=self.temperature
        labels=(out-self.shifts).argmax(1) if self.task=='multiclass' else out>=self.shifts
        if self.task=='multiclass':
            out-=out.max(1,keepdims=True);cp.exp(out,out=out);out/=out.sum(1,keepdims=True)
        else:
            # expit(z) expressed through exp(-abs(z)) to avoid overflow.
            z=cp.exp(-cp.abs(out));out[:]=cp.where(out>=0,1/(1+z),z/(1+z))
        cp.clip(out,1e-7,1-1e-7,out=out)
        return cp.asnumpy(out),cp.asnumpy(labels)
