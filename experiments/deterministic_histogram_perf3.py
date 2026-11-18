"""Run the preregistered H-HIST-PERF-3 hybrid-layout experiment."""

from __future__ import annotations

from experiments.deterministic_histogram import run_deterministic_histogram
from experiments.deterministic_histogram_core import (
    analyze_histogram_perf3_microbenchmark,
)
from experiments.deterministic_histogram_perf2 import _benchmark_node_count


def run_deterministic_histogram_perf3(config):
    """Benchmark hybrid dispatch, apply gates, then run two full fits."""

    protocol = config['deterministic_histogram_performance_3']
    cases = []
    for nodes_count in (8, 16):
        case = _benchmark_node_count(protocol, nodes_count)
        case['hybrid_kernel_ms'] = case.pop('perf2_kernel_ms')
        cases.append(case)
    microbenchmark = analyze_histogram_perf3_microbenchmark(
        cases,
        protocol['minimum_16_node_speedup_fraction'],
        protocol['maximum_8_node_regression_fraction'],
    )
    if not microbenchmark['correctness_gate_passed']:
        raise RuntimeError(
            'H-HIST-PERF-3 differs bitwise from the frozen H-HIST-PERF-1 kernel'
        )
    return run_deterministic_histogram(
        config,
        microbenchmark=microbenchmark,
    )
