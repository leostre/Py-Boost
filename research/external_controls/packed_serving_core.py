"""Typed finite launch plans and owned packed arrays; CUDA is a separate shell."""
from dataclasses import dataclass
import numpy as np
from unified_serving_core import coalesce
from experiments.prepared_leaf_inference import frozen_array


@dataclass(frozen=True)
class PackedPlan:
    groups:int
    layout:str
    outputs:int
    max_width:int
    has_residual:bool

    def __post_init__(self):
        if self.groups not in (1,2,4) or self.layout not in ('aos','soa') or not 1<=self.max_width<=self.outputs<=1024:
            raise ValueError('unsupported packed launch domain')
        if self.shared_bytes>49152:raise ValueError('packed launch exceeds48KiB shared memory')

    @property
    def threads(self):return 32*self.groups

    @property
    def shared_bytes(self):
        return 8*(self.outputs+self.groups*self.max_width+self.groups*self.outputs*int(self.has_residual)+self.threads)


@dataclass(frozen=True)
class PackedSnapshot:
    arrays:dict
    task:str
    features:int
    outputs:int
    trees:int
    max_width:int
    has_residual:bool
    layout:str

    @property
    def resident_bytes(self):return sum(a.nbytes for a in self.arrays.values())


def prepare_snapshot(model,shifts,layout='aos'):
    model=coalesce(model);c=len(model.bias);shifts=np.asarray(shifts,np.float32)
    if layout not in ('aos','soa') or shifts.shape!=(c,) or not np.isfinite(shifts).all():raise ValueError('invalid snapshot controls/layout')
    nodes=[];noff=[];loff=[];coeff=[];basis=[];ptr=[];cols=[];vals=[];meta=[]
    n=t=aoff=boff=poff=voff=0
    for block in model.blocks:
        b=block.base;w=b.coefficients.shape[1];has=bool(len(block.values));low=b.representation=='low_rank'
        nodes.append(b.nodes);noff.extend(b.node_offsets[:-1]+n);loff.extend(b.leaf_offsets[:-1])
        coeff.append(b.coefficients.ravel());basis.append(b.basis.T.copy().ravel() if low else np.empty(0,np.float32))
        if has:
            order=np.lexsort((block.columns,block.rows));counts=np.bincount(block.rows,minlength=len(b.coefficients))
            pointers=np.r_[0,np.cumsum(counts)]+voff
            if pointers[-1]>np.iinfo(np.int32).max:raise ValueError('packed CSR exceeds32bit domain')
            ptr.extend(pointers);cols.extend(block.columns[order]);vals.extend(block.values[order])
        meta.append([t,b.n_trees,w,int(low),aoff,boff,poff,int(has)])
        t+=b.n_trees;n+=len(b.nodes);aoff+=b.coefficients.size;boff+=b.basis.size if low else 0
        if has:poff+=len(b.coefficients)+1;voff+=len(block.values)
    node_array=np.concatenate(nodes).astype(np.float32)
    arrays=dict(meta=np.asarray(meta,np.int64),noff=np.asarray(noff,np.int64),loff=np.asarray(loff,np.int64),
                coefficients=np.concatenate(coeff),basis=np.concatenate(basis),ptr=np.asarray(ptr,np.int32),
                columns=np.asarray(cols,np.int32),values=np.asarray(vals,np.float32),bias=model.bias,shifts=shifts)
    if layout=='aos':
        arrays.update(nodes=node_array,feature=np.empty(0,np.int32),threshold=np.empty(0,np.float32),left=np.empty(0,np.int32),right=np.empty(0,np.int32))
    else:
        arrays.update(nodes=np.empty(0,np.float32),feature=node_array[:,0].astype(np.int32),threshold=node_array[:,1].copy(),
                      left=node_array[:,2].astype(np.int32),right=node_array[:,3].astype(np.int32))
    return PackedSnapshot({k:frozen_array(v) for k,v in arrays.items()},model.task,model.n_features,c,t,
                          max(b.base.coefficients.shape[1] for b in model.blocks),bool(len(vals)),layout)
