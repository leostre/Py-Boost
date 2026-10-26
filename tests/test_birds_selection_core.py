import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.birds_selection_core import (contributions, weighted_metrics, as_quality,
    passes, pool_indices, select, bootstrap_weights, split_half_weights, leave_one_out_weights)
from experiments.output_calibration_core import calibrated_quality


@st.composite
def multilabel_cases(draw):
    n = draw(st.integers(2, 15)); c = draw(st.integers(2, 6)); k = draw(st.integers(1, 4))
    y = np.array(draw(st.lists(st.integers(0, 1), min_size=n*c, max_size=n*c))).reshape(n, c)
    z = np.array(draw(st.lists(st.integers(-8, 8), min_size=k*n*c, max_size=k*n*c)), float).reshape(k,n,c)/2
    shifts = np.array(draw(st.lists(st.integers(-2,2),min_size=k*c,max_size=k*c)),np.float32).reshape(k,c)
    indices = draw(st.lists(st.integers(0,n-1),min_size=1,max_size=25))
    return y,z,shifts,np.bincount(indices,minlength=n),indices


@given(multilabel_cases())
@settings(max_examples=45, deadline=None)
def test_weighted_sufficient_statistics_match_independent_sklearn_on_literal_repetitions(case):
    y,z,shifts,weights,indices = case
    actual = weighted_metrics(contributions(y,z,shifts),weights[None,:])[0]
    for i in range(len(z)):
        expected = calibrated_quality(y[indices],z[i,indices], 'multilabel', shifts[i])
        assert as_quality(actual[i]) == pytest.approx(expected, abs=2e-14)


def test_macro_keeps_zero_support_labels_and_weighted_handles_no_positives():
    y=np.zeros((3,4),int);z=np.full((1,3,4),-2.)
    actual=weighted_metrics(contributions(y,z,np.zeros((1,4))),np.ones((1,3),int))[0,0]
    assert actual[1] == actual[2] == 0
    y[0,0]=1;z[0,0,0]=2
    actual=weighted_metrics(contributions(y,z,np.zeros((1,4))),np.ones((1,3),int))[0,0]
    assert actual[1] == .25 and actual[2] == 1


def test_single_binary_output_is_not_silently_treated_as_a_multilabel_domain():
    with pytest.raises(ValueError):
        contributions(np.zeros((3,1)),np.zeros((1,3,1)),np.zeros((1,1)))


@given(n=st.integers(4,60),repeats=st.integers(1,15),seed=st.integers(0,10000))
def test_paired_sampling_conserves_rows_and_internal_halves_never_overlap(n,repeats,seed):
    boot=bootstrap_weights(n,repeats,seed)
    assert boot.shape==(repeats,n) and np.all(boot.sum(axis=1)==n) and (boot>=0).all()
    assert np.array_equal(boot,bootstrap_weights(n,repeats,seed))
    a,b=split_half_weights(n,repeats,seed)
    assert np.all(a+b==1) and np.all(a*b==0)
    assert np.all(a.sum(axis=1)+b.sum(axis=1)==n)
    assert np.array_equal(a[:repeats],b[repeats:])
    jack=leave_one_out_weights(n)
    assert np.all(jack.sum(axis=1)==n-1) and np.all(jack.diagonal()==0)


def rows():
    return [dict(candidate_id=f'b{b}-t{t}',block=b,threshold_fraction=t,parameters=10+b,
                 payload_bytes=100+b,validation='must not influence the shortlist')
            for b in range(5) for t in (0.,.25,.5,1.)]


def test_shortlists_are_metadata_only_and_factorial_and_do_not_mutate_input():
    candidates=rows()
    assert len(pool_indices(candidates,'all_all'))==20
    assert len(pool_indices(candidates,'all_fixed'))==5
    assert len(pool_indices(candidates,'recent_all'))==8
    assert len(pool_indices(candidates,'recent_fixed'))==2
    assert [candidates[i]['block'] for i in pool_indices(candidates,'recent_fixed')]==[3,4]
    assert all(r['validation']=='must not influence the shortlist' for r in candidates)
    assert len(pool_indices(candidates[:4],'recent_fixed'))==1


@given(st.permutations(tuple(range(20))))
def test_selection_is_order_invariant_and_robust_rule_cannot_accept_outside_point_gate(order):
    candidates=rows(); q=np.array([[1., .5+(i%4)*.02,.6] for i in range(20)])
    full=np.array([1.,.5,.6]);rates=np.linspace(.5,1.,20)
    expected=select(candidates,q,full,list(range(20)),rates)
    perm=list(order)
    actual=select([candidates[i] for i in perm],q[perm],full,list(range(20)),rates[perm])
    assert actual['candidate_id']==expected['candidate_id'] and actual['accepted']==expected['accepted']
    assert not actual['accepted'] or actual['point_pass']


def test_three_quality_guards_relative_boundary_and_ninety_percent_inclusive():
    ref=np.array([2.,.8,.6])
    assert passes(np.array([2.1,.76,.57]),ref)
    assert not passes(np.array([2.10001,.99,.99]),ref)
    assert not passes(np.array([1.,.75999,.99]),ref)
    assert not passes(np.array([1.,.99,.56999]),ref)
    r=rows()[:1];q=np.array([[2.,.8,.6]])
    assert select(r,q,ref,[0],np.array([.9]))['accepted']
    assert not select(r,q,ref,[0],np.array([.898]))['accepted']


@pytest.mark.parametrize('bad', [np.zeros((1,3)),np.array([[1,-1,1]]),np.array([[.5,1,1]])])
def test_invalid_multiplicities_are_rejected(bad):
    evidence=contributions(np.zeros((3,2)),np.zeros((1,3,2)),np.zeros((1,2)))
    with pytest.raises(ValueError):
        weighted_metrics(evidence,bad)


def test_diagnostic_candidate_is_never_misreported_as_accepted():
    q=np.array([[10.,1.,1.]])
    chosen=select(rows()[:1],q,np.array([1.,.5,.5]),[0])
    assert chosen['candidate_id'] and not chosen['accepted'] and not chosen['point_pass']


def test_stable_gate_cannot_accept_a_point_failure_even_with_perfect_bootstrap_rate():
    chosen=select(rows()[:1],np.array([[1.06,.9,.9]]),np.array([1.,.5,.5]),[0],np.array([1.]))
    assert not chosen['accepted']
