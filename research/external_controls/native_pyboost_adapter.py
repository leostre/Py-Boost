"""Rehydrate saved dense leaves for unchanged master prediction; no tree fitting."""
from pathlib import Path
import hashlib,json,sys,types,importlib.util
import numpy as np


def original_classes():
    root=Path(__file__).parent/'original_pyboost'
    pin=json.loads((root/'source.json').read_text())
    for path,expected in pin['files'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
    # The study's experiments.* modules are independent of py_boost inference.
    if 'py_boost' in sys.modules:
        raise RuntimeError('Original baseline must be imported before any other py_boost')
    sys.path.insert(0,str(root))
    # The frozen source-only environment has no installed py_boost distribution.
    # Skip only top-level optional exports / distribution-version lookup. All
    # inference submodules below execute byte-identical original source.
    package=types.ModuleType('py_boost');package.__path__=[str(root/'py_boost')]
    package.__file__=str(root/'py_boost/__init__.py')
    package.__spec__=importlib.util.spec_from_loader('py_boost',loader=None,is_package=True)
    sys.modules['py_boost']=package
    gpu=types.ModuleType('py_boost.gpu');gpu.__path__=[str(root/'py_boost/gpu')]
    gpu.__spec__=importlib.util.spec_from_loader('py_boost.gpu',loader=None,is_package=True)
    sys.modules['py_boost.gpu']=gpu
    from py_boost.gpu.base import Ensemble
    from py_boost.gpu.tree import Tree
    from py_boost.gpu.losses.losses import BCELoss
    return Ensemble,Tree,BCELoss,pin


class OriginalGpu:
    def __init__(self,model,temperature,shifts,classes):
        import cupy as cp
        Ensemble,Tree,BCELoss,pin=classes
        if model.task!='multilabel' or temperature!=1.:
            raise ValueError('This sealed study supports unchanged multilabel T=1 only')
        native=Ensemble();native.models=[];native.base_score=model.bias.copy();native.nfeats=model.n_features
        native.postprocess_fn=BCELoss().postprocess_output
        for block in model.blocks:
            base=block.base
            if base.representation!='dense' or len(block.values):raise ValueError('Native control requires dense saved leaves')
            for t in range(base.n_trees):
                a,b=base.node_offsets[t:t+2];lo,hi=base.leaf_offsets[t:t+2]
                tree=Tree.__new__(Tree)
                for attr in ('gains','feats','bin_splits','nans','split','val_splits','values','group_index','leaves','test_format','test_format_offsets','feature_importance_gain','feature_importance_split'):
                    setattr(tree,attr,None)
                tree.nout=len(model.bias);tree.ngroups=1;tree.max_nodes=int(hi-lo)
                tree.test_format=base.nodes[a:b].copy().ravel()
                tree.test_format_offsets=np.array([0],np.int32)
                tree.values=base.coefficients[lo:hi].copy()
                tree.group_index=np.zeros(tree.nout,np.uint64)
                native.models.append(tree)
        self.native=native;self.pin=pin;self.shifts=cp.asarray(shifts,dtype=cp.float32)
        self.threshold=1/(1+cp.exp(-self.shifts));native.to_device()
        arrays=[native.base_score,self.shifts,self.threshold]
        arrays += [a for tree in native.models for a in vars(tree).values() if isinstance(a,cp.ndarray)]
        self.resident_device_array_bytes=sum({int(a.data.ptr):a.nbytes for a in arrays}.values())

    def device(self,x):
        # Unchanged Ensemble.predict, including its float32 accumulation,
        # internal allocations, synchronization and original BCE postprocess.
        p=self.native.predict(x)
        return p,p>=self.threshold

    def request(self,x):
        import cupy as cp
        p,l=self.device(cp.asarray(x,dtype=cp.float32,order='C'))
        return p.get(),l.get()
