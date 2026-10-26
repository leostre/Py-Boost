"""Regression checks for opt-in application of exact shortlist refinement."""

from types import SimpleNamespace

import cupy as cp
import numpy as np

from experiments.benchmark_helena import _model_structure_fingerprint
from experiments.decision_aware_refinement_apply_core import assess_application_gate
from py_boost.gpu.utils import _apply_refinement_entry


def _entry(*, requested=True, applied=True, changes=2):
    return {
        'application_requested': requested,
        'applied_to_structure': applied,
        'application_refusal_reason': None,
        'applied_change_count': changes,
        'diagnostic_output_budget': 50,
        'shortlist_size': 64,
        'candidate_gains_allclose': True,
        'all_refined_winners_gain_equivalent': True,
        'nodes': [],
    }


def _record(*, apply, structure, prediction, seconds=1.0, log_loss=0.1, f1=0.5):
    return {
        'trees_built': 100,
        'split_refinement_apply': apply,
        'model_structure_fingerprint': structure,
        'prediction_fingerprint': prediction,
        'train_seconds': seconds,
        'log_loss': log_loss,
        'f1_weighted': f1,
        'f1_macro': 0.2,
        'f1_micro': 0.4,
    }


def test_apply_refinement_entry_returns_refined_splitters_and_marks_trace():
    entry = {
        'candidate_gains_allclose': True,
        'all_refined_winners_gain_equivalent': True,
        'applied_to_structure': False,
        'applied_change_count': 0,
        'nodes': [
            {
                'refined_identity': [7, 3, True],
                'refined_full_gain': 2.5,
                'would_change_structure_action': True,
                'applied_to_structure': False,
            },
            {
                'refined_identity': [2, 8, False],
                'refined_full_gain': 0.0,
                'would_change_structure_action': False,
                'applied_to_structure': False,
            },
        ],
    }
    feature, gain, split, nan_left = _apply_refinement_entry(entry, 0.1)
    assert feature.get().tolist() == [7, 2]
    assert gain.get().tolist() == [2.5, 0.0]
    assert split.get().tolist() == [3, 8]
    assert nan_left.get().tolist() == [True, False]
    assert entry['applied_to_structure'] is True
    assert entry['applied_change_count'] == 1
    assert entry['applied_valid_split_count'] == 1


def test_application_gate_accepts_exact_non_inferior_pair():
    records = {
        'baseline': _record(apply=False, structure='base', prediction='base'),
        'treatment': _record(apply=True, structure='new', prediction='same'),
        'repeat': _record(apply=True, structure='new', prediction='same'),
    }
    traces = {
        'baseline': [_entry(requested=False, applied=False, changes=0)],
        'treatment': [_entry()],
        'repeat': [_entry()],
    }
    gate = assess_application_gate(
        records,
        traces,
        expected_trees=100,
        expected_budget=50,
        expected_shortlist=64,
        maximum_log_loss_increase=0.002,
        maximum_f1_weighted_decrease=0.005,
        maximum_runtime_ratio=1.15,
    )
    assert gate['passed'] is True


def test_application_gate_rejects_nondeterministic_predictions():
    records = {
        'baseline': _record(apply=False, structure='base', prediction='base'),
        'treatment': _record(apply=True, structure='new', prediction='first'),
        'repeat': _record(apply=True, structure='new', prediction='second'),
    }
    traces = {
        'baseline': [_entry(requested=False, applied=False, changes=0)],
        'treatment': [_entry()],
        'repeat': [_entry()],
    }
    gate = assess_application_gate(
        records,
        traces,
        expected_trees=100,
        expected_budget=50,
        expected_shortlist=64,
        maximum_log_loss_increase=0.002,
        maximum_f1_weighted_decrease=0.005,
        maximum_runtime_ratio=1.15,
    )
    assert gate['passed'] is False
    assert gate['checks']['treatment_repeat_predictions_are_exact'] is False


def test_structure_fingerprint_uses_retained_inference_arrays():
    first = SimpleNamespace(
        models=[SimpleNamespace(
            feats=None,
            test_format=np.asarray([1.0, 0.5, -1.0, -2.0], dtype=np.float32),
            test_format_offsets=np.asarray([0], dtype=np.int32),
        )]
    )
    second = SimpleNamespace(
        models=[SimpleNamespace(
            feats=None,
            test_format=np.asarray([2.0, 0.5, -1.0, -2.0], dtype=np.float32),
            test_format_offsets=np.asarray([0], dtype=np.int32),
        )]
    )
    assert _model_structure_fingerprint(first) != _model_structure_fingerprint(second)
