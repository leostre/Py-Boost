import numpy as np
from packed_preflight import fixture
from six_r2 import weaken
from experiments.sparse_leaf_core import leaf_design

def test_residual_zero_removes_storage_and_preserves_base():
    model,x,_=fixture(6);block=model.blocks[1];zero=weaken(block,0.)
    assert len(zero.values)==len(zero.rows)==len(zero.columns)==0
    design=leaf_design(block.base.leaf_indices(x),len(block.base.coefficients))
    np.testing.assert_allclose(zero.raw_from_design(design),block.base.raw_from_leaves(block.base.leaf_indices(x)),atol=1e-14)

def test_residual_scale_is_affine_and_does_not_change_topology_or_basis():
    model,x,_=fixture(6);block=model.blocks[1]
    design=leaf_design(block.base.leaf_indices(x),len(block.base.coefficients))
    none=weaken(block,0.).raw_from_design(design);full=block.raw_from_design(design)
    for alpha in (.25,.5,1.):
        result=weaken(block,alpha)
        assert result.base is block.base
        np.testing.assert_allclose(result.raw_from_design(design),none+alpha*(full-none),atol=3e-8)
