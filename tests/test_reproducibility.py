"""Regression and conservation checks for persisted experiment identities."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.reproducibility import array_fingerprint, paired_seed_map
from experiments.benchmark_datasets import DATASETS, LoadedDataset, split_benchmark_indices, split_benchmark_dataset


def test_model_and_split_seeds_are_explicitly_independent():
    assert paired_seed_map([42, 52, 62], [132, 142, 152]) == {42: 132, 52: 142, 62: 152}
    with pytest.raises(ValueError, match='one split seed'):
        paired_seed_map([42, 52], [132])


def test_fingerprint_distinguishes_shape_dtype_and_content():
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert array_fingerprint(values.copy()) == array_fingerprint(values)
    assert array_fingerprint(values.T) != array_fingerprint(values)
    assert array_fingerprint(values.astype(np.float64)) != array_fingerprint(values)


@given(seed=st.integers(min_value=0, max_value=100_000))
def test_saved_split_indices_partition_rows_and_reconstruct_arrays(seed):
    X = np.arange(240, dtype=np.float32).reshape(80, 3)
    y = np.tile(np.arange(4, dtype=np.int32), 20)
    dataset = LoadedDataset(DATASETS['helena'], X, y)
    train, test = split_benchmark_indices(dataset, seed)
    X_train, X_test, y_train, y_test = split_benchmark_dataset(dataset, seed)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(80))
    np.testing.assert_array_equal(X_train, X[train])
    np.testing.assert_array_equal(X_test, X[test])
    np.testing.assert_array_equal(y_train, y[train])
    np.testing.assert_array_equal(y_test, y[test])
