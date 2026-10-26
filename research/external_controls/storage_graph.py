"""Reachable numerical byte intervals, not complete process memory or a pool."""
from dataclasses import dataclass
import types
import numpy as np


@dataclass(frozen=True)
class BufferSpan:
    device: str
    address: int
    nbytes: int

    def __post_init__(self):
        if (not isinstance(self.device,str) or not self.device.strip()
                or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (self.address,self.nbytes))
                or (self.nbytes and not self.address)):
            raise ValueError('invalid owned numerical buffer')


def union_bytes(spans):
    result=0
    for device in {s.device for s in spans}:
        end=0
        for s in sorted((a for a in spans if a.device==device),key=lambda a:a.address):
            result+=max(0,s.address+s.nbytes-max(end,s.address))
            end=max(end,s.address+s.nbytes)
    return result


def numerical_spans(objects):
    try: import torch
    except ImportError: torch=None
    result=[]
    visited=set()
    def walk(value):
        if id(value) in visited: return
        visited.add(id(value))
        if isinstance(value,np.ndarray):
            owner=value
            while isinstance(owner.base,np.ndarray): owner=owner.base
            result.append(BufferSpan('cpu',int(owner.__array_interface__['data'][0]),int(owner.nbytes)))
            return
        if torch is not None and isinstance(value,torch.Tensor):
            storage=value.untyped_storage()
            result.append(BufferSpan(str(value.device),int(storage.data_ptr()),int(storage.nbytes())))
            if (value.is_leaf or value.retains_grad) and value.grad is not None: walk(value.grad)
            return
        if isinstance(value,(str,bytes,int,float,bool,type(None),type,types.ModuleType,
                             types.FunctionType,types.BuiltinFunctionType,types.MethodType)): return
        if isinstance(value,dict):
            for v in value.values(): walk(v)
        elif isinstance(value,(list,tuple,set,frozenset)):
            for v in value: walk(v)
        elif hasattr(value,'__dict__'):
            walk(vars(value))
    walk(objects)
    return result


def inventory(objects):
    spans=numerical_spans(objects)
    return dict(cpu_numerical_bytes=union_bytes([s for s in spans if s.device=='cpu']),
        gpu_numerical_bytes=union_bytes([s for s in spans if s.device.startswith('cuda')]),
        numerical_bytes=union_bytes(spans),observed_spans=len(spans),
        limitation='tensor/NumPy storage reachable via Python fields only; excludes hidden native buffers, Python overhead and allocator pools')
