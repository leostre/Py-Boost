"""Storage, calibration isolation, budget and decision/probability contracts."""
import inspect
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.shared_leaf_basis import compress, functional_basis, reference_class_model
from experiments.sparse_leaf_core import Cost, leaf_design, select_candidate
from experiments.output_calibration_core import calibrated_quality
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants, family_allows
from experiments.uniform_threshold_study import uniform_plan, materialize_uniform
from test_sparse_leaf_core import teacher


@pytest.mark.parametrize('representation', ['dense', 'reference_class', 'low_rank'])
def test_all_representations_roundtrip_with_all_thresholds_counted(tmp_path, representation):
    dense = teacher('multiclass' if representation == 'reference_class' else 'multilabel')
    base = (reference_class_model(dense) if representation == 'reference_class' else
            compress(dense, np.zeros(8), np.eye(8)[:, :2]) if representation == 'low_rank' else dense)
    shifts = np.linspace(-.5, .5, 8).astype(np.float32)
    model = ThresholdModel(base, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), shifts)
    x = np.array([[-1], [1], [np.nan]], np.float32)
    design = leaf_design(base.leaf_indices(x), len(base.coefficients))
    np.testing.assert_array_equal(model.raw_from_design(design), base.raw_from_leaves(base.leaf_indices(x)))
    np.testing.assert_array_equal(model.predict(x), base.predict(x))
    assert model.parameter_count == base.parameter_count+8
    assert model.payload_bytes == base.payload_bytes+32
    path = tmp_path/'model.npz'
    model.save(path)
    loaded = ThresholdModel.load(path)
    assert loaded.base.representation == representation
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    np.testing.assert_array_equal(loaded.predict_labels(x), model.predict_labels(x))
    with np.load(path, allow_pickle=False) as f:
        assert model.payload_bytes == sum(f[k].nbytes for k in
            ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias',
             'rows', 'columns', 'values', 'decision_shifts'))
        counted = sum(f[k].size for k in ('coefficients', 'basis', 'bias', 'values', 'decision_shifts'))
        assert model.parameter_count == counted-(representation == 'reference_class')


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10000), task=st.sampled_from(['multilabel', 'multiclass']))
def test_threshold_variants_never_change_probability_loss_and_have_explicit_zero(seed, task):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(12, 4))
    y = rng.integers(0, 4, 12) if task == 'multiclass' else rng.integers(0, 2, (12, 4))
    variants = threshold_variants(raw[:7], y[:7], raw[7:], y[7:], task, [0, -1, 1], [.25, .5, 1])
    original = calibrated_quality(y[7:], raw[7:], task, np.empty(0, np.float32))
    assert len(variants) == 4
    np.testing.assert_array_equal(variants[0]['decision_shifts'], np.zeros(4))
    for v in variants:
        assert len(v['decision_shifts']) == 4
        assert v['validation']['loss'] == original['loss']


@given(parameters=st.integers(1, 100000), nnz=st.integers(0, 6000))
def test_family_caps_have_integer_boundary_oracle(parameters, nnz):
    for family, percent in [('sparse_01', 1), ('sparse_02', 2), ('sparse_05', 5)]:
        assert family_allows({'kind': 'sparse', 'nnz': nnz}, family, parameters) == (nnz <= percent*parameters//100)
        assert family_allows({'kind': 'rank_only', 'nnz': 0}, family, parameters)
    assert not family_allows({'kind': 'full', 'nnz': 0}, 'rank_only', parameters)


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
def test_planner_uses_only_calibration_labels_for_every_family_and_replays_selection(monkeypatch, task):
    dense = teacher(task, c=12)
    x = np.array([[-1], [1]]*24, np.float32)
    target = np.arange(48)%12 if task == 'multiclass' else (np.arange(576).reshape(48, 12)%3 == 0).astype(int)
    train, cal, select = np.arange(24), np.arange(24, 36), np.arange(36, 48)
    leaves = [dense.leaf_indices(x[v]) for v in (train, cal, select)]
    mean, basis, _ = functional_basis(dense.contributions(leaves[0]), task)
    config = {'budget_kinds': ['parameters', 'memory', 'joint'], 'budget_fractions': [.99],
              'base_rank_fractions': [.5, .75, .9], 'residual_capacity_fractions': [.25, .5, 1.],
              'tiny_selectors': ['balanced_mse', 'balanced_margin'], 'margin_scale': .5,
              'decision_grid': [0, -1, 1], 'decision_shrinkage': [.25, .5, 1.]}
    import experiments.uniform_threshold_core as core
    original = core.fit_decision_shifts
    calls = []
    def checked(raw, labels, *args):
        np.testing.assert_array_equal(labels, target[cal])
        calls.append(len(raw))
        return original(raw, labels, *args)
    monkeypatch.setattr(core, 'fit_decision_shifts', checked)
    plan = uniform_plan(dense, dense, mean, basis, leaves[0], target[train], leaves[1], target[cal],
                        leaves[2], target[select], Cost(dense.parameter_count, dense.payload_bytes), config)
    full = plan['full_choice']
    assert full == min(plan['full_candidates'], key=lambda r: (-r['validation']['f1_macro'],
                       r['validation']['loss'], r['payload_bytes'], r['parameters'], r['candidate_id']))
    assert full['parameters'] == dense.parameter_count+12  # full is outside compressed caps
    design = leaf_design(leaves[2], len(dense.coefficients))
    for selected in [{'candidate': full}, *plan['selections']]:
        chosen = selected['candidate']
        model = materialize_uniform(chosen, plan, dense, dense, mean, basis)
        assert model.parameter_count == chosen['parameters'] and model.payload_bytes == chosen['payload_bytes']
        q = calibrated_quality(target[select], model.raw_from_design(design), task, model.decision_shifts)
        for k in q:
            assert q[k] == pytest.approx(chosen['validation'][k], abs=1e-9)
    assert calls and all(n == len(cal) for n in calls)
    assert not any('test' in p for p in inspect.signature(uniform_plan).parameters)
    assert {r['kind'] for r in plan['rows']} == {'rank_only', 'sparse'}


def test_required_threshold_storage_cannot_be_omitted():
    dense = teacher()
    with pytest.raises(ValueError, match='exactly C'):
        ThresholdModel(dense, np.empty(0, np.int32), np.empty(0, np.int32),
                       np.empty(0, np.float32), np.empty(0, np.float32))


def test_calibration_label_changes_affect_fitted_thresholds_but_selection_labels_do_not():
    raw = np.array([[-.5], [-.2], [.1]])
    cal_y = np.array([[1], [1], [0]])
    first = threshold_variants(raw, cal_y, raw, cal_y, 'multilabel', [0, -1, 1], [1])
    other_selection = threshold_variants(raw, cal_y, raw, 1-cal_y, 'multilabel', [0, -1, 1], [1])
    other_calibration = threshold_variants(raw, 1-cal_y, raw, cal_y, 'multilabel', [0, -1, 1], [1])
    assert [v['decision_shifts'] for v in first] == [v['decision_shifts'] for v in other_selection]
    assert first[-1]['decision_shifts'] != other_calibration[-1]['decision_shifts']
