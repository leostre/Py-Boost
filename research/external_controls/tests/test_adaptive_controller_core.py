from dataclasses import replace
import numpy as np
import budget_quality_worker
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from experiments.shared_leaf_basis import probabilities
from adaptive_controller_core import Action,candidates,local_actions,diagnose,extend_basis,promotion


def test_full_action_replays_original_probability_and_compact_action_has_valid_cost():
    a,x,_,_=fixture(9,4,3);b,_,_,_=fixture(17,4,2);x=np.linspace(-1,1,54,dtype=np.float32)[:,None]
    old=as_protected(a);y=np.arange(len(x))%4
    pool=list(candidates(old,b,dict(train=x,development=x),dict(train=y),[Action(3,8,0.),Action(1,8,.02)],'functional'))
    assert len(pool)==2
    expected=old.raw(x)+as_protected(b).raw(x)
    np.testing.assert_allclose(probabilities(pool[0]['model'].raw(x),'multiclass'),probabilities(expected,'multiclass'),atol=2e-6)
    for c in pool:np.testing.assert_allclose(c['raw']['development'],c['model'].raw(x),atol=0)
    assert pool[1]['cost']['cost_bytes']>0


def test_controller_and_extension_are_bounded_and_keep_original_span():
    b=np.linalg.qr(np.random.default_rng(5).normal(size=(8,8)))[0].astype(np.float32)
    q=extend_basis(b[:,:2],b[:,::-1]);np.testing.assert_allclose(q[:,:2]@q[:,:2].T,b[:,:2]@b[:,:2].T,atol=2e-6)
    np.testing.assert_allclose(q.T@q,np.eye(8),atol=2e-6)
    e=np.random.default_rng(7).normal(size=(40,8));d=diagnose(e,2*e)
    assert abs(d['structural']-3)<1e-10
    actions=local_actions(2,8,.02,[1,2,4,8],d)
    assert 1<=len(actions)<=6 and len(set(actions))==len(actions)


def test_empty_promotion_does_not_claim_a_winner():
    # Actual promotion requires a nonempty matched experiment, not missing data.
    import pytest
    with pytest.raises((ValueError,ZeroDivisionError)):promotion([])


def test_posthoc_without_checkpoint_preserves_bias_and_full_prediction():
    a,x,_,_=fixture(71,4,3);x=np.linspace(-1,1,54,dtype=np.float32)[:,None]
    a=replace(a,base=replace(a.base,bias=np.array([.2,-.7,1.1,-.3],np.float32)))
    pool=list(candidates(None,a,dict(train=x,development=x),dict(train=np.arange(len(x))%4),
        [Action(3,8,0.),Action(1,8,.02)],'risk_residual'))
    expected=as_protected(a).raw(x)
    np.testing.assert_allclose(probabilities(pool[0]['model'].raw(x),'multiclass'),probabilities(expected,'multiclass'),atol=2e-6)
    assert pool[0]['model'].n_trees==a.base.n_trees
