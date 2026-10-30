import numpy as np
from scipy.special import softmax
from budget_quality_worker import native_load
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel
from abl_protected_core import as_protected
from unified_training_core import coarsen


def test_saved_deep_tree_respects_caps_above_eight_and_full_cap_preserves_function():
    nodes=[];leaf=0
    def build(left,right,depth):
        nonlocal leaf
        if depth==0:
            leaf+=1;return -leaf
        index=len(nodes);mid=(left+right)/2;nodes.append(None)
        a,b=build(left,mid,depth-1),build(mid,right,depth-1);nodes[index]=[1.,mid,a,b];return index
    build(-1.,1.,5);values=np.random.default_rng(25).normal(size=(32,4)).astype(np.float32)
    base=LeafEnsemble(np.array(nodes,np.float32),np.array([0,31],np.int64),np.array([0,32],np.int64),values,np.empty((0,0),np.float32),np.array([.1,.2,-.1,0],np.float32),1,'multiclass')
    model=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(4,np.float32))
    x=np.linspace(-.99,.99,256,dtype=np.float32)[:,None];teacher=as_protected(model).raw(x)
    for cap in (8,16,32):
        result=coarsen(model,x,teacher,cap)
        assert max(np.diff(result.blocks[0].base.leaf_offsets))<=cap
        if cap==32:np.testing.assert_allclose(softmax(result.raw(x),axis=1),softmax(teacher,axis=1),atol=2e-7,rtol=0)
