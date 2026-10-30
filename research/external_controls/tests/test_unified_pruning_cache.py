import budget_quality_worker  # select pinned native_source before experiments imports
import numpy as np
import abl_budget_revision_core as core


def test_static_pruning_loss_cache_preserves_exact_uncached_merges(monkeypatch):
    nodes=np.array([[1,0,1,2],[1,-.5,3,4],[1,.5,5,6],[1,-.75,-1,-2],[1,-.25,-3,-4],[1,.25,-5,-6],[1,.75,-7,-8]],np.float32)
    rng=np.random.default_rng(32);targets=rng.normal(size=(8,4));weights=rng.uniform(1,3,size=(8,4));basis=np.eye(4)
    original=core.fit_error;calls=[]
    def tracked(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(core,'fit_error',tracked)
    cached=core.prune_tree(nodes,targets,weights,basis,2);cached_calls=len(calls);calls.clear()
    monkeypatch.setattr(core,'lru_cache',lambda maxsize=None:lambda function:function)
    uncached=core.prune_tree(nodes,targets,weights,basis,2)
    assert cached_calls<=15 and len(calls)>cached_calls
    assert cached['merges']==uncached['merges']
    for key in ('nodes','target','weights','coefficients','mapping','origin_nodes'):
        np.testing.assert_array_equal(cached[key],uncached[key])
