from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from abl_budget_revision_core import pack,decode_model
from experiments.shared_leaf_basis import probabilities
from unified_training_core import candidate_grid,coarsen,merge_increment_views,shortlist


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_full_representation_preserves_function_in_both_archive_modes(task):
    old,_,_,_=fixture(12,4,3);inc,_,_,_=fixture(11,4,2)
    if task=='multilabel':
        old=replace(old,base=replace(old.base,task=task));inc=replace(inc,base=replace(inc.base,task=task))
    checkpoint=as_protected(old);x=np.linspace(-1,1,72,dtype=np.float32)[:,None]
    y=np.arange(72)%4 if task=='multiclass' else (np.arange(72)[:,None]%np.arange(2,6)==0).astype(int)
    expected=checkpoint.raw(x)+as_protected(inc).raw(x)
    for protected in (False,True):
        candidates=list(candidate_grid(checkpoint,inc,{'train':x,'development':x}, {'train':y},None,
            protected=protected,ranks=(4-int(task=='multiclass'),),caps=(8,)))
        assert len(candidates)==1
        got=candidates[0]['raw']['train']
        np.testing.assert_allclose(probabilities(got,task),probabilities(expected,task),atol=2e-6)
        assert candidates[0]['model'].n_trees==5
        assert candidates[0]['train_mse']<1e-10
    candidates=list(candidate_grid(None,inc,{'train':x}, {'train':y},None,
        ranks=(4-int(task=='multiclass'),),caps=(8,)))
    np.testing.assert_allclose(probabilities(candidates[0]['raw']['train'],task),probabilities(as_protected(inc).raw(x),task),atol=2e-6)


def test_sparse_candidates_and_train_only_shortlist_obey_the_price_grid():
    old,_,_,_=fixture(13,4,3);inc,_,_,_=fixture(19,4,3);x=np.linspace(-1,1,80,dtype=np.float32)[:,None]
    candidates=list(candidate_grid(as_protected(old),inc,{'train':x}, {'train':np.arange(80)%4},10**6,
        ranks=(1,2,3),caps=(8,),residuals=(0,.05)))
    assert candidates and all(c['cost']['cost_bytes']<=10**6 for c in candidates)
    chosen=shortlist(candidates,3);assert len(chosen)<=3
    assert {c['uid'] for c in chosen}<={c['uid'] for c in candidates}
    assert not list(candidate_grid(as_protected(old),inc,{'train':x}, {'train':np.arange(80)%4},1,ranks=(1,),caps=(8,)))


def test_branch_merge_has_exact_weights_and_one_checkpoint():
    a,_,_,_=fixture(3,4,3);b,_,_,_=fixture(6,4,2);old,_,_,_=fixture(9,4,4)
    x=np.linspace(-1,1,50,dtype=np.float32)[:,None]
    merged,_=merge_increment_views([a,b],[.25,.75])
    merged_raw=as_protected(merged).raw(x)
    np.testing.assert_allclose(merged_raw,.25*as_protected(a).raw(x)+.75*as_protected(b).raw(x),atol=2e-6)
    cp=as_protected(old);out=pack(cp.blocks+as_protected(merged).blocks,cp.bias+merged.base.bias)
    assert out.n_trees==9
    np.testing.assert_allclose(out.raw(x),cp.raw(x)+merged_raw,atol=2e-6)
