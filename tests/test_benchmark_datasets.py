"""Pure dataset-registry and target-normalization tests."""

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark_datasets import (
    DATASETS,
    Task,
    _binary_targets,
    _labels_to_indicator,
    _load_age_prediction,
    _prepare_moa,
    _prepare_rt_iot,
    _subsample_indices,
)


def test_dataset_registry_has_expected_tasks_and_output_dimensions():
    assert DATASETS["helena"].task is Task.MULTICLASS
    assert DATASETS["helena"].expected_outputs == 100
    assert DATASETS["birds"].expected_outputs == 19
    assert DATASETS["yeast"].expected_outputs == 14
    assert DATASETS["genbase"].expected_features == 1185
    assert DATASETS["mnist"].expected_outputs == 10
    assert DATASETS["age_prediction"].expected_features == 64
    assert DATASETS["mediamill"].expected_outputs == 101
    assert DATASETS["moa"].expected_features == 875
    assert DATASETS["moa"].expected_outputs == 206
    assert DATASETS["dionis"].data_id == 41167
    assert DATASETS["dionis"].expected_outputs == 355
    assert DATASETS["rt_iot2022"].expected_features == 81
    assert DATASETS["rt_iot2022"].expected_outputs == 12


def test_binary_target_normalization_accepts_strings_and_preserves_shape():
    target = np.array([["no", "yes"], ["yes", "no"], ["no", "no"]])

    normalized = _binary_targets(target)

    assert normalized.shape == target.shape
    assert normalized.dtype == np.float32
    assert set(np.unique(normalized)) == {0.0, 1.0}


def test_binary_target_normalization_rejects_non_binary_columns():
    target = np.array([["a"], ["b"], ["c"]])

    with pytest.raises(ValueError, match="instead of 2"):
        _binary_targets(target)


def test_libsvm_labels_support_zero_and_one_based_indices():
    zero_based = _labels_to_indicator([(0.0, 2.0), (1.0,)], n_outputs=3)
    one_based = _labels_to_indicator([(1.0, 3.0), (2.0,)], n_outputs=3)

    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(zero_based.toarray(), expected)
    np.testing.assert_array_equal(one_based.toarray(), expected)


def test_multiclass_subsample_indices_preserve_source_identity():
    target = np.repeat(np.arange(4), 25)

    selected = _subsample_indices(target, Task.MULTICLASS, 20, seed=42)

    assert selected.shape == (20,)
    assert len(np.unique(selected)) == 20
    assert set(np.unique(target[selected])) == {0, 1, 2, 3}


def test_age_prediction_loader_uses_union_of_fold_zero(tmp_path):
    cache_dir = tmp_path / "repo" / "data" / "benchmarks"
    source_dir = tmp_path / "repo" / "experiments" / "data" / "age_pred" / "fold_0"
    source_dir.mkdir(parents=True)
    columns = {f"emb_{index:04d}": [index, index + 1] for index in range(64)}
    train = pd.DataFrame({"client_id": [1, 2], "bins": [0, 1], **columns})
    test = pd.DataFrame({"client_id": [3, 4], "bins": [2, 3], **columns})
    train.to_csv(source_dir / "emb_train.csv", index=False)
    test.to_csv(source_dir / "emb_test.csv", index=False)

    features, target = _load_age_prediction(cache_dir)

    assert features.shape == (4, 64)
    assert target.tolist() == [0, 1, 2, 3]
    assert (cache_dir / "age_prediction" / "coles64_fold0_union.npz").exists()


def test_rt_iot_preparation_removes_endpoint_ports_and_encodes_categories():
    frame = pd.DataFrame({
        "id.orig_p": [1, 2, 3],
        "id.resp_p": [4, 5, 6],
        "proto": ["tcp", "udp", "tcp"],
        "value": [0.5, 1.5, 2.5],
    })

    features, target = _prepare_rt_iot(frame, ["normal", "attack", "normal"])

    assert features.shape == (3, 2)
    assert features.dtype == np.float32
    assert target.dtype == np.int32
    assert target.tolist() == [1, 0, 1]


def test_moa_preparation_matches_sketchboost_encoding_and_target_order():
    features_frame = pd.DataFrame({
        "sig_id": ["b", "a"],
        "cp_type": ["trt_cp", "ctl_vehicle"],
        "cp_time": [24, 72],
        "cp_dose": ["D2", "D1"],
        "g-0": [1.5, -2.0],
    })
    target_frame = pd.DataFrame({
        "sig_id": ["a", "b"],
        "target_0": [1, 0],
        "target_1": [0, 1],
    })

    features, target = _prepare_moa(features_frame, target_frame)

    assert features.dtype == np.float32
    assert target.dtype == np.float32
    np.testing.assert_array_equal(
        features,
        np.array([[0, 24, 0, 1.5], [1, 72, 1, -2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        target,
        np.array([[0, 1], [1, 0]], dtype=np.float32),
    )


def test_moa_preparation_rejects_unmatched_targets():
    features_frame = pd.DataFrame({
        "sig_id": ["a", "b"],
        "cp_type": ["trt_cp", "trt_cp"],
        "cp_time": [24, 48],
        "cp_dose": ["D1", "D2"],
    })
    target_frame = pd.DataFrame({"sig_id": ["a", "c"], "target_0": [1, 0]})

    with pytest.raises(ValueError, match="align"):
        _prepare_moa(features_frame, target_frame)
