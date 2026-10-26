import json

import numpy as np
import pytest

from experiments.a100_runner import sha256
from experiments.birds_selection_study import (forbid_fitting, locked, birds_selection_test,
                                               selection_diagnostics)
from experiments.birds_selection_core import quality_vector


def test_fitting_forbidden_even_if_a_later_refactor_accidentally_calls_it():
    import experiments.output_calibration_core as calibration
    import experiments.leaf_budget_runtime as runtime
    import experiments.parallel_leaf_worker as worker
    import experiments.shared_leaf_basis as basis
    with forbid_fitting():
        for fit in (calibration.fit_decision_shifts,runtime.fit_dense,worker.fit_increment,basis.functional_basis):
            with pytest.raises(RuntimeError, match='fitting is forbidden'):
                fit()


def test_test_phase_refuses_an_incomplete_parent_before_loading_any_dataset(tmp_path,monkeypatch):
    import experiments.birds_selection_study as study
    parent=tmp_path/'parent.json'
    parent.write_text(json.dumps(dict(status='unfinished',test_evaluated=False)))
    monkeypatch.setattr(study,'expanded_manifest',lambda *a:pytest.fail('test data access before parent gate'))
    with pytest.raises(RuntimeError,match='must complete first'):
        birds_selection_test(dict(birds_selection_manifest=str(parent),
                                  birds_selection_manifest_sha256=sha256(parent)),{},tmp_path)


def test_input_fingerprint_change_is_fatal(tmp_path):
    file=tmp_path/'model.npz';file.write_bytes(b'original')
    expected=sha256(file);file.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='input changed'):
        locked(file,expected)


def test_full_reference_and_candidate_use_the_same_perturbation_not_independent_draws():
    rows=[dict(candidate_id='b0-t1',block=0,threshold_fraction=1.,parameters=20,payload_bytes=100)]
    point=np.array([[1.,.5,.6],[1.,.5,.6]])
    # Both models improve together in one draw and deteriorate together in the other.
    boot=np.array([[[.8,.8,.8],[.8,.8,.8]],[[2.,.2,.2],[2.,.2,.2]]])
    decisions,rates,_,_,_=selection_diagnostics(rows,point,boot,
        np.stack([point,point]),np.stack([point,point]),np.stack([point,point]),
        np.array([20,25]),np.zeros((2,3),int))
    assert rates.tolist()==[1.]
    assert all(d['accepted'] for d in decisions)
