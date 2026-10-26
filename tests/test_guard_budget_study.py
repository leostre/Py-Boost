from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from experiments.guard_budget_study import same_arrays, locked_json
from experiments.a100_runner import write_report, sha256
from experiments.uniform_threshold_core import ThresholdModel
from test_sparse_leaf_core import teacher


def test_gate_compares_arrays_not_only_probabilities_and_locks_input(tmp_path):
    dense = teacher('multilabel', c=8)
    model = ThresholdModel(dense, np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(8,np.float32))
    same_arrays(model,model)
    changed = replace(model, decision_shifts=np.ones(8,np.float32))
    with pytest.raises(AssertionError):
        same_arrays(model,changed)
    path = tmp_path/'manifest.json'
    write_report(path, dict(test_evaluated=False))
    digest = sha256(path)
    assert locked_json(path,digest)[0] == dict(test_evaluated=False)
    write_report(path,dict(test_evaluated=True))
    with pytest.raises(RuntimeError,match='changed'):
        locked_json(path,digest)


def test_strict_no_candidate_final_selection_never_has_a_fake_checkpoint(tmp_path):
    from experiments.closed_budget_study import choose_final_checkpoints
    result = choose_final_checkpoints([], Mock(), Mock(), {}, {}, Mock(), {},tmp_path,tmp_path)
    assert result['state']=='no_accepted_checkpoint' and result['candidate'] is None


def test_guard_shell_uses_provisional_feedback_but_discards_it_after_second_failure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import experiments.closed_budget_study as study
    from experiments.shared_leaf_basis import compress
    from experiments.sparse_leaf_core import leaf_design
    from experiments.history_basis_study import raw
    dense = teacher('multilabel', c=8)
    dense.save(tmp_path/'dense.npz')
    compact = compress(dense, np.zeros(8), np.eye(8)[:, :2])
    models = [ThresholdModel(replace(compact, coefficients=compact.coefficients.copy()+v),
        np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(8,np.float32))
        for v in (1,2,3)]
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(get_default_memory_pool=lambda:
        SimpleNamespace(free_all_blocks=lambda:None)))
    monkeypatch.setitem(sys.modules,'resource',SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda _:SimpleNamespace(ru_maxrss=1)))
    monkeypatch.setattr(study,'check_deadline',lambda *args:None)
    initial, fitted = [], []
    def fit(request):
        with np.load(request['initial_path']) as data:
            initial.append(data['train'].copy())
        study.empty_model(dense).save(request['output'])
        fitted.append(1)
        return dict(seconds=0,sha256=sha256(request['output']))
    monkeypatch.setattr(study,'fit_increment',fit)
    monkeypatch.setattr(study,'combine_actual',lambda *args:study.empty_model(dense))
    monkeypatch.setattr(study,'prefix',lambda *args:study.empty_model(dense))
    calls=[]
    def project(*args,**kwargs):
        assert not kwargs['enforce_local_quality'] and kwargs['prefer_local_quality']
        index=len(calls);calls.append(1)
        return models[index],dict(state='accepted',local_quality_pass=index==0)
    monkeypatch.setattr(study,'compact_proposal',project)
    def choose(checkpoints,*args):
        assert [b for b,_ in checkpoints]==[0]
        study.independent_snapshot(checkpoints[0][1])
        return dict(state='accepted',candidate=dict(block=0))
    monkeypatch.setattr(study,'choose_final_checkpoints',choose)
    x=np.array([[-1],[1]]*4,np.float32);y=np.zeros((8,8),np.int32)
    roles=dict(train=np.array([0,1]),calibration=np.array([2,3]),control=np.array([4,5]),
               selection=np.array([6]),test=np.array([7]))
    full=dict(reference_cost=dict(parameters=100000,payload_bytes=1000000),
        fits=[dict(depth=5,artifact='dense.npz',sha256=sha256(tmp_path/'dense.npz'))],validation_reference={})
    result=study.run_path('periodic_fixed',dict(dataset='toy',seed=1,task='multilabel'),full,tmp_path,
        dict(closed_max_blocks=5,own_pool_limit_bytes=1000000,lr=.1),tmp_path/'out',x,y,roles,guard_policy='confirm2')
    assert len(fitted)==3 and result['native_fit_jobs']==3
    np.testing.assert_array_equal(initial[1],raw(models[0],x[roles['train']]).astype(np.float32))
    np.testing.assert_array_equal(initial[2],raw(models[1],x[roles['train']]).astype(np.float32))
    assert result['cycles'][1]['guard_action']=='provisional'
    assert result['cycles'][2]['guard_action']=='rollback'
    assert result['final_guard_state']['retained_through']==0
