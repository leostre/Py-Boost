"""GPU checks for the opt-in deterministic histogram reduction."""

import os

import numpy as np
import pytest

from py_boost import GradientBoosting


pytestmark = pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='orx GPU acceptance only',
)


def _cpu_cumulative_histogram(
    encoded,
    target,
    nodes,
    rows,
    columns,
    outputs,
    nnodes,
    nbins,
):
    result = np.zeros(
        (len(outputs), nnodes, len(columns), nbins),
        dtype=np.float64,
    )
    for output_position, output in enumerate(outputs):
        for node in range(nnodes):
            node_rows = rows[nodes[rows] == node]
            for feature_position, feature in enumerate(columns):
                for row in node_rows:
                    result[
                        output_position,
                        node,
                        feature_position,
                        encoded[row, feature],
                    ] += float(target[row, output])
    return result.cumsum(axis=-1)


def _problem(seed=91):
    rng = np.random.default_rng(seed)
    nrows, nfeatures, noutputs = 79, 7, 5
    encoded = rng.integers(
        0, 24, size=(nrows, nfeatures), dtype=np.uint8
    )
    target = rng.normal(size=(nrows, noutputs)).astype(np.float32)
    nodes = rng.integers(0, 3, size=nrows, dtype=np.int32)
    rows = np.arange(nrows, dtype=np.uint64)
    columns = np.asarray([6, 1, 4, 0], dtype=np.uint64)
    outputs = np.asarray([3, 0, 4], dtype=np.uint64)
    return encoded, target, nodes, rows, columns, outputs


def test_parameter_is_opt_in_and_preserved_in_model_params():
    default = GradientBoosting('multilabel')
    enabled = GradientBoosting(
        'multilabel',
        deterministic_histogram=True,
        deterministic_histogram_layout='perf1',
    )
    assert default.params['deterministic_histogram'] is False
    assert default.params['deterministic_histogram_layout'] == 'hybrid'
    assert enabled.params['deterministic_histogram'] is True
    assert enabled.params['deterministic_histogram_layout'] == 'perf1'


def test_deterministic_histogram_is_exact_across_row_permutations():
    import cupy as cp

    from py_boost.gpu.utils import histogram, pad_and_move

    encoded, target, nodes, rows, columns, outputs = _problem()
    rng = np.random.default_rng(13)
    common = (
        pad_and_move(encoded),
        cp.asarray(target),
        cp.asarray(nodes),
        cp.asarray(columns),
    )
    first = histogram(
        *common,
        cp.asarray(rng.permutation(rows)),
        cp.asarray(outputs),
        3,
        32,
        deterministic=True,
    ).get()
    second = histogram(
        *common,
        cp.asarray(rng.permutation(rows)),
        cp.asarray(outputs),
        3,
        32,
        deterministic=True,
    ).get()
    assert np.array_equal(first, second)


def test_deterministic_histogram_matches_float64_reference_for_subsets():
    import cupy as cp

    from py_boost.gpu.utils import histogram, pad_and_move

    encoded, target, nodes, rows, columns, outputs = _problem(seed=101)
    selected_rows = rows[(rows % 4) != 1]
    actual = histogram(
        pad_and_move(encoded),
        cp.asarray(target),
        cp.asarray(nodes),
        cp.asarray(columns),
        cp.asarray(selected_rows[::-1].copy()),
        cp.asarray(outputs),
        3,
        32,
        deterministic=True,
    ).get()
    expected = _cpu_cumulative_histogram(
        encoded,
        target,
        nodes,
        selected_rows,
        columns,
        outputs,
        3,
        32,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_deterministic_histogram_supports_multiple_node_chunks():
    import cupy as cp

    from py_boost.gpu.utils import histogram, pad_and_move

    rng = np.random.default_rng(131)
    encoded = rng.integers(0, 16, size=(137, 5), dtype=np.uint8)
    target = rng.normal(size=(137, 4)).astype(np.float32)
    nodes = rng.integers(0, 11, size=137, dtype=np.int32)
    rows = np.arange(137, dtype=np.uint64)
    columns = np.asarray([4, 2, 0], dtype=np.uint64)
    outputs = np.asarray([3, 1, 0], dtype=np.uint64)
    actual = histogram(
        pad_and_move(encoded),
        cp.asarray(target),
        cp.asarray(nodes),
        cp.asarray(columns),
        cp.asarray(rows),
        cp.asarray(outputs),
        11,
        16,
        deterministic=True,
    ).get()
    expected = _cpu_cumulative_histogram(
        encoded,
        target,
        nodes,
        rows,
        columns,
        outputs,
        11,
        16,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize(
    ('seed', 'nrows', 'nnodes', 'noutputs'),
    [
        (173, 79, 3, 3),
        (179, 137, 11, 5),
        (181, 193, 17, 7),
    ],
)
def test_optimized_kernel_bitwise_matches_frozen_reference(
    seed,
    nrows,
    nnodes,
    noutputs,
):
    import cupy as cp

    from py_boost.gpu.utils import (
        fill_histogram_deterministic,
        fill_histogram_deterministic_perf1_reference,
        fill_histogram_deterministic_perf2_reference,
        fill_histogram_deterministic_reference,
        pad_and_move,
    )

    rng = np.random.default_rng(seed)
    nfeatures = 7
    nbins = 32
    encoded = rng.integers(
        0,
        nbins,
        size=(nrows, nfeatures),
        dtype=np.uint8,
    )
    target = rng.normal(size=(nrows, noutputs + 2)).astype(np.float32)
    nodes = rng.integers(0, nnodes, size=nrows, dtype=np.int32)
    rows = np.arange(nrows, dtype=np.uint64)
    rows = rng.permutation(rows[(rows % 5) != 2]).copy()
    columns = np.asarray([6, 1, 4, 0], dtype=np.uint64)
    outputs = np.arange(noutputs, dtype=np.uint64)[::-1].copy()

    encoded_gpu = pad_and_move(encoded)
    target_gpu = cp.asarray(target)
    nodes_gpu = cp.asarray(nodes)
    columns_gpu = cp.asarray(columns)
    rows_gpu = cp.asarray(rows)
    outputs_gpu = cp.asarray(outputs)
    shape = (len(outputs), nnodes, len(columns), nbins)
    optimized = cp.zeros(shape, dtype=cp.float32)
    perf1_reference = cp.zeros_like(optimized)
    perf2_reference = cp.zeros_like(optimized)
    reference = cp.zeros_like(optimized)

    fill_histogram_deterministic(
        optimized,
        encoded_gpu,
        target_gpu,
        nodes_gpu,
        columns_gpu,
        rows_gpu,
        outputs_gpu,
    )
    fill_histogram_deterministic_reference(
        reference,
        encoded_gpu,
        target_gpu,
        nodes_gpu,
        columns_gpu,
        rows_gpu,
        outputs_gpu,
    )
    fill_histogram_deterministic_perf1_reference(
        perf1_reference,
        encoded_gpu,
        target_gpu,
        nodes_gpu,
        columns_gpu,
        rows_gpu,
        outputs_gpu,
    )
    fill_histogram_deterministic_perf2_reference(
        perf2_reference,
        encoded_gpu,
        target_gpu,
        nodes_gpu,
        columns_gpu,
        rows_gpu,
        outputs_gpu,
    )

    assert cp.array_equal(optimized, reference)
    assert cp.array_equal(optimized, perf1_reference)
    assert cp.array_equal(optimized, perf2_reference)


@pytest.mark.parametrize(
    ('nnodes', 'expected_kernel_name', 'expected_threads'),
    [
        (1, 'deterministic_histogram_perf1', 128),
        (8, 'deterministic_histogram_perf1', 128),
        (9, 'deterministic_histogram', 64),
        (16, 'deterministic_histogram', 64),
        (17, 'deterministic_histogram', 64),
    ],
)
def test_deterministic_histogram_dispatches_at_eight_node_boundary(
    nnodes,
    expected_kernel_name,
    expected_threads,
):
    from py_boost.gpu.utils import deterministic_histogram_launch

    kernel, threads, *_ = deterministic_histogram_launch(nnodes)
    assert kernel.name == expected_kernel_name
    assert threads == expected_threads


def test_perf1_policy_disables_node_count_dispatch():
    from py_boost.gpu.utils import deterministic_histogram_launch

    kernel, threads, *_ = deterministic_histogram_launch(16, layout='perf1')
    assert kernel.name == 'deterministic_histogram_perf1'
    assert threads == 128


def test_deterministic_histogram_layout_trace_counts_selected_kernels():
    from py_boost.gpu.utils import (
        deterministic_histogram_launch,
        start_deterministic_histogram_layout_trace,
        stop_deterministic_histogram_layout_trace,
    )

    start_deterministic_histogram_layout_trace()
    deterministic_histogram_launch(8, layout='hybrid')
    deterministic_histogram_launch(16, layout='hybrid')
    deterministic_histogram_launch(16, layout='perf1')
    counts = stop_deterministic_histogram_layout_trace()

    assert counts == {
        ('hybrid', 'perf1', 8): 1,
        ('hybrid', 'perf2', 16): 1,
        ('perf1', 'perf1', 16): 1,
    }


def test_deterministic_histogram_layout_rejects_unknown_policy():
    from py_boost.gpu.utils import deterministic_histogram_launch

    with pytest.raises(ValueError, match='unknown deterministic histogram layout'):
        deterministic_histogram_launch(8, layout='unknown')


def test_profile_records_all_deterministic_histogram_stages():
    import cupy as cp

    from py_boost.gpu.utils import (
        histogram,
        pad_and_move,
        start_deterministic_histogram_profile,
        stop_deterministic_histogram_profile,
    )

    encoded, target, nodes, rows, columns, outputs = _problem(seed=191)
    start_deterministic_histogram_profile()
    try:
        histogram(
            pad_and_move(encoded),
            cp.asarray(target),
            cp.asarray(nodes),
            cp.asarray(columns),
            cp.asarray(rows),
            cp.asarray(outputs),
            3,
            32,
            deterministic=True,
            profile_kind='structure',
        )
    finally:
        records = stop_deterministic_histogram_profile()

    assert len(records) == 1
    record = records[0]
    assert record['kind'] == 'structure'
    assert record['nodes'] == 3
    assert record['input_rows'] == len(rows)
    for field in (
        'allocation_ms',
        'row_filter_ms',
        'row_to_host_ms',
        'cpu_sort_ms',
        'row_to_device_ms',
        'kernel_ms',
        'cumsum_ms',
        'sibling_subtraction_ms',
        'total_ms',
    ):
        assert np.isfinite(record[field])
        assert record[field] >= 0.0
