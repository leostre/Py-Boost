import numpy as np
import pytest
import budget_quality_worker
from broad_core import grouped_folds,assess,byte_limit,budget_reference,ranks_for,select_cases
from residual_gate_core import sufficient,quality


def test_grouped_calibration_has_disjoint_features_and_exact_cover():
    x=np.repeat(np.arange(21)[:,None],np.arange(21)%3+1,axis=0)
    folds=grouped_folds(x,71);seen=[]
    for f in folds:
        assert not set(x[list(f.fitting),0])&set(x[list(f.validation),0])
        seen.extend(f.validation)
    assert sorted(seen)==list(range(len(x)))


def test_common_assessment_replays_held_predictions_and_has_no_fitting_label_leak():
    rng=np.random.default_rng(7);x=rng.normal(size=(57,3));z=rng.normal(size=(57,7));y=rng.integers(0,2,z.shape)
    folds=grouped_folds(x,9);result,arrays=assess(z,y,folds,[0,.5,-.5,1,-1],[.25,.5,1])
    q=quality(y,*arrays,'multilabel',7)
    assert result['quality']==q
    for fold,state in zip(folds,result['states']):
        a,b=list(fold.fitting),list(fold.validation);changed=y.copy();changed[b]=1-changed[b]
        alternate,_=assess(z,changed,folds,[0,.5,-.5,1,-1],[.25,.5,1])
        assert alternate['states'][result['states'].index(state)]['policies']==state['policies']
        policy=state['policies'][result['policy']]
        replay=sufficient(y[b],z[b],'multilabel',policy)
        for actual,expected in zip(arrays,replay):np.testing.assert_array_equal(actual[b],expected)


@pytest.mark.parametrize('c',[6,14,53,159,374,983])
def test_ranks_and_integer_budgets_are_bounded(c):
    assert ranks_for(c)[-1]==c
    assert all(r<=64 or r==c for r in ranks_for(c))
    assert byte_limit(budget_reference(c),.7)*10<=budget_reference(c)*7


def test_selection_never_uses_test_and_preserves_unaffordable_status():
    def m(uid,family,cost,f1):return dict(uid=uid,family=family,cost_bytes=cost,assessment={'quality':{'f1_macro':f1,'loss':.2}})
    case=dict(case='toy',analytic_reference_bytes=100,models=[m('a','ordinary',30,.5),m('b','ordinary',70,.6),m('p','posthoc',50,.7)])
    seal=select_cases([case]);rows=seal['selected']
    assert not seal['test_evaluated']
    assert next(r for r in rows if r['family']=='ordinary' and r['ratio']==.25)['uid'] is None
    assert next(r for r in rows if r['family']=='mixed24' and r['ratio']==.7)['uid']=='p'
