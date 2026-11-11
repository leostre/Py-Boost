"""Run the preregistered H-HIST-PERF-2 kernel and full-fit checks."""

from __future__ import annotations

import gc

import cupy as cp
import numpy as np

from experiments.deterministic_histogram import run_deterministic_histogram
from experiments.deterministic_histogram_core import (
    analyze_histogram_perf2_microbenchmark,
)
from py_boost.gpu.utils import (
    fill_histogram_deterministic,
    fill_histogram_deterministic_perf1_reference,
    pad_and_move,
)


def _timed_kernel_call(function, result, arguments):
    profile = {}
    function(result, *arguments, profile_record=profile)
    return float(profile['kernel_ms'])


def _benchmark_node_count(protocol, nodes_count):
    seed = int(protocol['microbenchmark_seed']) + int(nodes_count)
    rng = np.random.default_rng(seed)
    nrows = int(protocol['microbenchmark_rows'])
    nfeatures = int(protocol['microbenchmark_features'])
    noutputs = int(protocol['microbenchmark_outputs'])
    nbins = int(protocol['microbenchmark_bins'])
    repeats = int(protocol['microbenchmark_repeats'])

    encoded = rng.integers(
        0,
        nbins,
        size=(nrows, nfeatures),
        dtype=np.uint8,
    )
    target = rng.normal(size=(nrows, noutputs)).astype(np.float32)
    nodes = (np.arange(nrows, dtype=np.int32) % nodes_count)
    rng.shuffle(nodes)
    rows = rng.permutation(np.arange(nrows, dtype=np.uint64)).copy()
    columns = np.arange(nfeatures, dtype=np.uint64)
    outputs = np.arange(noutputs, dtype=np.uint64)[::-1].copy()

    encoded_gpu = pad_and_move(encoded)
    target_gpu = cp.asarray(target)
    nodes_gpu = cp.asarray(nodes)
    rows_gpu = cp.asarray(rows)
    columns_gpu = cp.asarray(columns)
    outputs_gpu = cp.asarray(outputs)
    shape = (noutputs, nodes_count, nfeatures, nbins)
    perf1_result = cp.empty(shape, dtype=cp.float32)
    perf2_result = cp.empty_like(perf1_result)
    arguments = (
        encoded_gpu,
        target_gpu,
        nodes_gpu,
        columns_gpu,
        rows_gpu,
        outputs_gpu,
    )

    _timed_kernel_call(
        fill_histogram_deterministic_perf1_reference,
        perf1_result,
        arguments,
    )
    _timed_kernel_call(
        fill_histogram_deterministic,
        perf2_result,
        arguments,
    )
    bitwise_equal = bool(cp.array_equal(perf1_result, perf2_result).item())

    perf1_times = []
    perf2_times = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            perf1_times.append(_timed_kernel_call(
                fill_histogram_deterministic_perf1_reference,
                perf1_result,
                arguments,
            ))
            perf2_times.append(_timed_kernel_call(
                fill_histogram_deterministic,
                perf2_result,
                arguments,
            ))
        else:
            perf2_times.append(_timed_kernel_call(
                fill_histogram_deterministic,
                perf2_result,
                arguments,
            ))
            perf1_times.append(_timed_kernel_call(
                fill_histogram_deterministic_perf1_reference,
                perf1_result,
                arguments,
            ))

    del perf1_result, perf2_result
    del encoded_gpu, target_gpu, nodes_gpu, rows_gpu, columns_gpu, outputs_gpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return {
        'nodes': int(nodes_count),
        'bitwise_equal': bitwise_equal,
        'perf1_kernel_ms': perf1_times,
        'perf2_kernel_ms': perf2_times,
        'shape': list(shape),
        'rows': nrows,
        'features': nfeatures,
        'outputs': noutputs,
        'bins': nbins,
    }


def run_deterministic_histogram_perf2(config):
    """Benchmark both layouts, apply gates, then run two full fits."""

    protocol = config['deterministic_histogram_performance_2']
    cases = [
        _benchmark_node_count(protocol, nodes_count)
        for nodes_count in (8, 16)
    ]
    microbenchmark = analyze_histogram_perf2_microbenchmark(
        cases,
        protocol['minimum_16_node_speedup_fraction'],
        protocol['maximum_8_node_regression_fraction'],
    )
    if not microbenchmark['correctness_gate_passed']:
        raise RuntimeError(
            'H-HIST-PERF-2 differs bitwise from the frozen H-HIST-PERF-1 kernel'
        )
    return run_deterministic_histogram(
        config,
        microbenchmark=microbenchmark,
    )
