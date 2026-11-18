"""Pure dataset-registry and target-normalization tests."""

import numpy as np
import pytest

from experiments.benchmark_datasets import (
    DATASETS,
    Task,
    _binary_targets,
    _labels_to_indicator,
)


def test_dataset_registry_has_expected_tasks_and_output_dimensions():
    assert DATASETS["helena"].task is Task.MULTICLASS
    assert DATASETS["helena"].expected_outputs == 100
    assert DATASETS["birds"].expected_outputs == 19
    assert DATASETS["yeast"].expected_outputs == 14
    assert DATASETS["mediamill"].expected_outputs == 101


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
