"""Guarded correctness gate before real-data compact optimization runs."""
import signal
import time

from experiments.a100_runner import emit,write_report


def online_leaf_preflight(config,report,output):
    from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver
    from threadpoolctl import threadpool_limits
    import pytest
    started = time.perf_counter()
    signal.alarm(900)
    free,reason,probe = gpu_probe('shared',config['inference_gpu_minimum_free_mib'])
    if not free:
        raise RuntimeError(reason)
    import cupy as cp
    cp.cuda.runtime.free(0)
    guard = SharedGpuObserver()
    try:
        with guard,threadpool_limits(limits=1):
            code = pytest.main(['-q','tests/test_online_leaf_core.py','tests/test_shared_leaf_basis.py',
                                'tests/test_sparse_leaf_core.py','tests/test_uniform_threshold_core.py'])
            if code != 0:
                raise RuntimeError('compact training preflight tests failed')
        report.update(status='done',scientific_data_fits=0,small_synthetic_fits=8,
            fixed_topology_and_identity_replay_tests='passed',gpu_probe=probe,gpu_guard_samples=guard.samples,
            elapsed_seconds=time.perf_counter()-started,
            interpretation='Correctness gate only; no scientific quality/economy hypothesis tested')
        write_report(output/'report.json',report)
        emit('online_leaf_preflight_complete',status='passed',synthetic_fits=8,
            elapsed_seconds=report['elapsed_seconds'],gpu_observations=len(guard.samples))
    finally:
        signal.alarm(0)
