import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.prove_atomic_flip_rate import replay_comparisons, summarize_arm
from experiments.proving_common import DEFAULT_TOLERANCES, compare_replica, replica_record


def _tree(feature=0, threshold=10, gain=1.0, values=(0.0, 1.0), *, reverse_ids=False):
    return SimpleNamespace(
        ngroups=1,
        feats=np.array([[feature, -1, -1]]),
        bin_splits=np.array([[threshold, 0, 0]]),
        gains=np.array([[gain, 0.0, 0.0]], dtype=np.float32),
        nans=np.zeros((1, 3), dtype=bool),
        split=np.array([[[2, 1] if reverse_ids else [1, 2], [0, 0], [0, 0]]]),
        leaves=np.array([[0], [1], [0]] if reverse_ids else [[0], [0], [1]]),
        group_index=np.array([0]),
        values=np.array(values, dtype=np.float32).reshape(-1, 1),
    )


def _record(trees, logit=0.0):
    def postprocess(_):
        raise AssertionError('comparison must capture raw logits')

    model = SimpleNamespace(
        models=trees, postprocess_fn=postprocess, _default_postprocess_fn=lambda value: value,
        to_cpu=lambda: None,
    )
    model.predict = lambda _: model.postprocess_fn(np.array([[logit]], dtype=np.float32))
    record = replica_record(model, np.zeros((1, 1), dtype=np.float32), 0.0)
    assert model.postprocess_fn is postprocess
    return record


def test_exact_structure_and_leaf_values_follow_paths_not_node_ids():
    original = _record([_tree()])
    renumbered = _record([_tree(reverse_ids=True)])

    pair = compare_replica(original, renumbered)

    assert original['structure_fingerprints'] == renumbered['structure_fingerprints']
    assert original['leaf_value_fingerprints'] == renumbered['leaf_value_fingerprints']
    assert pair['flipped_trees'] == 0
    assert pair['prediction_bitwise_equal']


def test_q3_replays_gain_tolerances_from_serialized_split_details():
    original = _record([_tree(), _tree(), _tree(), _tree(feature=-1), _tree()])
    replica = _record([
        _tree(threshold=11),
        _tree(feature=1, gain=1.001),
        _tree(feature=1, gain=2.0),
        _tree(gain=0.002),
        _tree(values=(-0.0, 1.0)),
    ], logit=-0.0)

    atomic = compare_replica(original, replica)
    control = compare_replica(original, original)
    report = json.loads(json.dumps({
        'config': {'replicas': 2},
        'detail': {
            'atomic': {'comparisons_vs_replica0': [atomic]},
            'deterministic_control': {'comparisons_vs_replica0': [control]},
        },
    }))
    baseline, _ = replay_comparisons(report, DEFAULT_TOLERANCES)
    tight, clean = replay_comparisons(report, {
        **DEFAULT_TOLERANCES, 'gain_atol': 1e-4, 'gain_rtol': 2e-6,
    })

    assert baseline == [atomic]
    assert not atomic['prediction_bitwise_equal']
    assert atomic['prediction_max_abs_diff'] == 0
    assert atomic['leaf_value_only_trees'] == 1
    assert atomic['tolerant']['values_allclose_trees'] == 1
    assert atomic['tolerant']['hard_flip_trees'] == 1
    assert atomic['tolerant']['gain_equivalent_trees'] == 2
    summary = summarize_arm(tight)
    assert summary['flipped_trees'] == 4
    assert summary['tolerant']['near_tie_trees'] == 1
    assert summary['tolerant']['hard_flip_trees'] == 3
    assert summary['tolerant']['gain_equivalent_trees'] == 0
    assert clean == [control]

    report['detail']['atomic']['comparisons_vs_replica0'][0]['trees'][0]['elementwise'].clear()
    with pytest.raises(ValueError, match='incomplete split details'):
        replay_comparisons(report, DEFAULT_TOLERANCES)
