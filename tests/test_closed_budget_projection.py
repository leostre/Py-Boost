from dataclasses import replace
import inspect
import numpy as np

from experiments.closed_budget_projection import compact_proposal
from experiments.shared_leaf_basis import probabilities
from experiments.sparse_leaf_core import Budget, Cost, leaf_design
from experiments.output_calibration_core import calibrated_quality
from test_sparse_leaf_core import teacher


def test_projection_guard_calibration_isolation_cost_and_roundtrip(monkeypatch, tmp_path):
    dense = teacher('multilabel', c=8)
    x = np.array([[-1], [1]]*15, np.float32)
    # Keep each role's distinct label object so calibration leakage is visible.
    train = (np.arange(80).reshape(10, 8)%3 == 0).astype(np.int32)
    cal = (np.arange(80).reshape(10, 8)%3 == 1).astype(np.int32)
    control = (np.arange(80).reshape(10, 8)%3 == 2).astype(np.int32)
    import experiments.uniform_threshold_core as core
    original = core.fit_decision_shifts
    calls = []
    def checked(raw, labels, *args):
        assert labels is cal
        calls.append(1)
        return original(raw, labels, *args)
    monkeypatch.setattr(core, 'fit_decision_shifts', checked)
    cfg = dict(margin_scale=.5, decision_grid=[0, -1, 1], decision_shrinkage=[.5, 1.])
    budget = Budget('joint', .75, 10000, 40000)
    model, record = compact_proposal(dense, x[:10], train, x[10:20], cal, x[20:], control,
                                      budget, 1000, cfg)
    assert record['state'] == 'accepted' and calls
    assert budget.allows(Cost(model.parameter_count, model.payload_bytes))
    design = leaf_design(model.base.leaf_indices(x[20:]), len(model.base.coefficients))
    q = calibrated_quality(control, model.raw_from_design(design), 'multilabel', model.decision_shifts)
    assert q == record['chosen']['validation']
    path = tmp_path/'model.npz'
    model.save(path)
    from experiments.uniform_threshold_core import ThresholdModel
    loaded = ThresholdModel.load(path)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    assert not any('test' in p or 'selection' in p for p in inspect.signature(compact_proposal).parameters)


def test_no_memory_proposal_has_no_fake_dense_fallback():
    dense = teacher('multilabel', c=8)
    x = np.array([[-1], [1]]*3, np.float32)
    y = np.arange(48).reshape(6, 8)%2
    cfg = dict(margin_scale=.5, decision_grid=[0], decision_shrinkage=[1.])
    model, record = compact_proposal(dense, x, y, x, y, x, y, Budget('joint', .75, 1, 4), 100, cfg)
    assert model is None and record['state'] == 'no_memory_feasible'


def test_local_guard_ablation_cannot_relax_the_hard_resource_cap(monkeypatch):
    import experiments.closed_budget_projection as projection
    monkeypatch.setattr(projection, 'quality_guard', lambda *args: False)
    dense = teacher('multilabel', c=8)
    x = np.array([[-1], [1]]*3, np.float32)
    y = np.arange(48).reshape(6, 8)%2
    cfg = dict(margin_scale=.5, decision_grid=[0], decision_shrinkage=[1.])
    budget = Budget('joint', .75, 10000, 40000)
    rejected, info = projection.compact_proposal(dense, x, y, x, y, x, y, budget, 100, cfg, fixed=True)
    assert rejected is None and info['state'] == 'local_quality_rejected'
    accepted, info = projection.compact_proposal(dense, x, y, x, y, x, y, budget, 100, cfg,
                                                fixed=True, enforce_local_quality=False)
    assert accepted is not None and not info['local_quality_enforced'] and not info['local_quality_pass']
    assert budget.allows(Cost(accepted.parameter_count, accepted.payload_bytes))
    rejected, info = projection.compact_proposal(dense, x, y, x, y, x, y, Budget('joint', .75, 1, 4), 100, cfg,
                                                fixed=True, enforce_local_quality=False)
    assert rejected is None and info['state'] == 'no_memory_feasible'


def test_tolerant_protocol_keeps_the_old_feasible_choice_before_a_diagnostic(monkeypatch):
    import experiments.closed_budget_projection as projection
    dense = teacher('multilabel', c=8)
    x = np.array([[-1], [1]]*3, np.float32)
    y = np.arange(48).reshape(6, 8)%2
    cfg = dict(margin_scale=.5, decision_grid=[0], decision_shrinkage=[1.])
    budget = Budget('joint', .75, 10000, 40000)
    strict, a = projection.compact_proposal(dense, x, y, x, y, x, y, budget, 100, cfg)
    tolerant, b = projection.compact_proposal(dense, x, y, x, y, x, y, budget, 100, cfg,
        enforce_local_quality=False, prefer_local_quality=True)
    assert strict is not None and a['chosen'] == b['chosen']
    np.testing.assert_array_equal(strict.predict(x), tolerant.predict(x))
    monkeypatch.setattr(projection, 'quality_guard', lambda *args: False)
    tolerant, b = projection.compact_proposal(dense, x, y, x, y, x, y, budget, 100, cfg,
        enforce_local_quality=False, prefer_local_quality=True)
    assert tolerant is not None and not b['local_quality_pass']
    assert b['chosen'] == min(b['candidates'], key=projection.choice_key)
