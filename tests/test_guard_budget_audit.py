import pytest
from hypothesis import given,strategies as st

from experiments.guard_budget_audit import independent_guard_step, best_row, equal_quality, independent_depth
from experiments.local_guard_core import ActiveGuard,GuardPolicy,Outcome,advance_guard,StopAndRollback


@given(policy=st.sampled_from(list(GuardPolicy)),events=st.lists(st.sampled_from(list(Outcome)),min_size=1,max_size=8))
def test_independent_auditor_matches_production_decisions_on_generated_traces(policy,events):
    state=ActiveGuard();confirmed=-1;failures=0
    for block,event in enumerate(events):
        action,c,f,reason=independent_guard_step(policy.value,block,confirmed,failures,
                                              event is Outcome.PASS,event is not Outcome.NO_CANDIDATE)
        decision=advance_guard(policy,state,block,event)
        if isinstance(decision,StopAndRollback):
            assert action=='rollback' and decision.state.retained_through==c and decision.state.reason.value==reason
            break
        assert action==('confirmed' if decision.locally_confirmed else 'provisional')
        assert decision.state.last_confirmed_block==c and decision.state.consecutive_failures==f
        state=decision.state;confirmed,failures=c,f


def test_audit_cost_guard_and_selection_oracles_keep_three_metrics():
    reference=dict(loss=1.,f1_macro=.5,f1_weighted=.5)
    good=dict(candidate_id='good',parameters=20,payload_bytes=100,validation=reference)
    bad=dict(candidate_id='bad',parameters=10,payload_bytes=80,
             validation=dict(loss=1.1,f1_macro=.9,f1_weighted=.9))
    assert best_row([bad,good],reference)==(good,True)
    assert best_row([],reference)==(None,False)
    with pytest.raises(RuntimeError):
        equal_quality(reference,bad['validation'])
    assert independent_depth((0,0,0),200,8,4,0,dict(parameters=1,payload_bytes=4)) is None


def test_test_scoring_requires_successful_audit_parent_before_loading_any_data(tmp_path,monkeypatch):
    import experiments.guard_budget_audit as audit
    from experiments.a100_runner import write_report,sha256
    parent=tmp_path/'report.json'
    write_report(parent,dict(status='failed',test_evaluated=False))
    monkeypatch.setattr(audit,'expanded_manifest',lambda *a:pytest.fail('must not load data without a successful parent'))
    with pytest.raises(RuntimeError,match='must complete first'):
        audit.guard_budget_audit(dict(audit_only=False,pretest_audit_report=str(parent),
            pretest_audit_report_sha256=sha256(parent)),{},tmp_path)


def test_paired_prefix_check_rejects_changed_rng_before_guard_divergence():
    from experiments.guard_budget_audit import paired_prefixes
    strict=dict(dataset='toy',seed=1,cycles=[dict(initial_sha256='same',fit=dict(sha256='increment'),state='rejected')])
    other=dict(dataset='toy',seed=1,cycles=[dict(initial_sha256='same',fit=dict(sha256='different'),state='accepted')])
    entries=[dict(guard_policy='strict',allocation_policy='periodic_fixed',trajectory=strict,reused=False),
             dict(guard_policy='warmup2',allocation_policy='periodic_fixed',trajectory=other,reused=False)]
    with pytest.raises(RuntimeError,match='diverged'):
        paired_prefixes(entries)
    other['cycles'][0]['fit']['sha256']='increment'
    assert paired_prefixes(entries)[0]['identical_prefix_blocks']==1


def test_frozen_final_choice_is_not_reselected_due_to_one_ulp_loss_rounding(tmp_path):
    import numpy as np
    from experiments.guard_budget_audit import verify_final_choice
    from experiments.closed_budget_study import choose_final_checkpoints
    from experiments.shared_leaf_basis import compress
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.sparse_leaf_core import Budget
    from test_sparse_leaf_core import teacher
    base=compress(teacher('multilabel',c=8),np.zeros(8),np.eye(8)[:,:2])
    model=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(8,np.float32))
    x=np.array([[-1],[1]]*12,np.float32);y=(np.arange(192).reshape(24,8)%3==0).astype(int)
    roles=dict(calibration=np.arange(8,16),selection=np.arange(16,24))
    cfg=dict(decision_grid=[0],decision_shrinkage=[.5,1.])
    reference=dict(loss=10.,f1_macro=0.,f1_weighted=0.)
    selected=choose_final_checkpoints([(0,model)],x,y,roles,reference,Budget('joint',.75,10000,40000),
                                      cfg,tmp_path,tmp_path)
    for row in selected['candidates']:
        row['validation']['loss']=float(np.nextafter(row['validation']['loss'],np.inf))
    stats=dict(models=0,maximum_selection_replay_error=0.)
    result=verify_final_choice(dict(choice=selected),[(0,model)],x,y,roles,reference,tmp_path,cfg,stats)
    assert result['candidate']['candidate_id']==selected['candidate']['candidate_id']
    assert stats['maximum_selection_replay_error']<1e-12
