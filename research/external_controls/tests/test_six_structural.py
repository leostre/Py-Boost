from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from six_structural_core import keep_trees, newton_values


def test_tree_subset_has_original_contributions_and_one_bias():
    model,x,_,_=fixture(8,5,7); model=as_protected(model)
    for ids in ([0,1,2,3,4,5,6],[6,2,4],[1]):
        got=keep_trees(model,ids); base=model.blocks[0].base; leaf=base.leaf_indices(x)
        expected=model.bias+sum(base.coefficients[leaf[t]].astype(float) for t in ids)
        np.testing.assert_allclose(got.raw(x),expected,atol=1e-12)
        assert got.n_trees==len(ids)


@pytest.mark.parametrize('indices',[[],[1,1],[-1],[7]])
def test_invalid_subset_rejected(indices):
    model,*_=fixture(8,5,7)
    with pytest.raises(ValueError):keep_trees(as_protected(model),indices)


def test_newton_update_zero_and_clip_rounding():
    values=np.array([[1.,2.]],np.float32)
    np.testing.assert_array_equal(newton_values(values,[[100.,-100.]],[[1.,1.]],0.),values)
    np.testing.assert_array_equal(newton_values(values,[[100.,-100.]],[[1.,1.]],.5),[[0.,3.]])
