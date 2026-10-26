"""Paired reference/prepared inference and isolated request-memory measurements."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from experiments.inference_benchmark_core import ROLES, batch_sizes, timing_summary
from experiments.inference_benchmark_worker import load_inputs, rss, reset_rss_peak
from experiments.prepared_leaf_inference import PreparedLeafInference
from experiments.uniform_threshold_core import ThresholdModel

PATHS = ('reference', 'prepared')


def build(source, path, backend):
    if backend == 'cpu':
        if path == 'reference':
            return source
        if path == 'prepared':
            return PreparedLeafInference.prepare(source)
        if path == 'compiled':
            return source.prepare_compiled_inference()
        if path == 'accumulated':
            from experiments.accumulated_leaf_inference import AccumulatedLeafInference
            return AccumulatedLeafInference.prepare(source)
        raise ValueError('unknown CPU inference path')
    from experiments.inference_benchmark_gpu import GpuThresholdReference
    from experiments.prepared_leaf_gpu import PreparedLeafGpu
    if path == 'reference':
        return GpuThresholdReference(source)
    if path == 'prepared':
        return PreparedLeafGpu(source)
    if path == 'fused':
        from experiments.fused_leaf_gpu import FusedLeafGpu
        return FusedLeafGpu(source)
    if path == 'accumulated':
        from experiments.accumulated_leaf_gpu import AccumulatedLeafGpu
        return AccumulatedLeafGpu(source)
    if path == 'unified':
        from experiments.unified_leaf_gpu import UnifiedLeafGpu
        return UnifiedLeafGpu(source)
    raise ValueError('unknown GPU inference path')


def prepare_measured(source, path, backend, cp):
    # Tracing compiler internals changes JIT time dramatically. Measure JIT setup
    # with wall time/RSS, not tracemalloc; request buffers remain NumPy-tracked.
    traced = not (backend == 'cpu' and path in ('compiled', 'accumulated'))
    if traced:
        tracemalloc.start()
    reset = reset_rss_peak()
    before = rss()
    t = time.perf_counter()
    model = build(source, path, backend)
    if cp:
        cp.cuda.get_current_stream().synchronize()
    elapsed = (time.perf_counter()-t)*1000
    after = rss()
    peak = tracemalloc.get_traced_memory()[1] if traced else None
    if traced:
        tracemalloc.stop()
    resident = model.resident_bytes if path != 'reference' else source.payload_bytes
    return model, {'prepare_ms': elapsed, 'prepare_traced_peak_bytes': peak,
        'resident_array_bytes': resident, 'saved_payload_bytes': source.payload_bytes,
        'prepare_rss_before_bytes': before['VmRSS'], 'prepare_rss_after_bytes': after['VmRSS'],
        'prepare_rss_peak_bytes': after['VmHWM'], 'prepare_rss_peak_reset': reset,
        'jit_ms': getattr(model, 'compilation_ms', None),
        'kernel_resources': getattr(model, 'kernel_resources', None)}


def calibrated_decisions(p, source):
    shifts = source.decision_shifts.astype(float)
    if source.base.task == 'multiclass':
        return (np.log(p)-shifts).argmax(axis=1)
    return p >= 1/(1+np.exp(-shifts))


def timing(job, cp=None):
    x = load_inputs(job)
    models, sources, setup = {}, {}, []
    sync = cp.cuda.get_current_stream().synchronize if cp else lambda: None
    for role in ROLES:
        sources[role] = ThresholdModel.load(job['models'][role]['path'])
        for path in job.get('paths', PATHS):
            sync()
            model, costs = prepare_measured(sources[role], path, job['backend'], cp)
            models[role, path] = model
            setup.append({'role': role, 'path': path, **costs})
    records, parity = [], []
    keys = [(role, path) for role in ROLES for path in job.get('paths', PATHS)]
    for n in batch_sizes(len(x)):
        host = np.ascontiguousarray(x[:n])
        device = cp.asarray(host) if cp else None
        for mode in (('device', 'host') if cp else ('host',)):
            def call(key):
                model = models[key]
                return model.predict_device(device) if mode == 'device' else model.predict(host)
            for key in keys:
                sync()
                t = time.perf_counter()
                out = call(key)
                sync()
                first_ms = (time.perf_counter()-t)*1000
                actual = out.get() if mode == 'device' else out
                expected = sources[key[0]].predict(host)
                error = float(np.max(np.abs(expected-actual)))
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
                differences = int(np.count_nonzero(calibrated_decisions(actual, sources[key[0]]) !=
                                                   calibrated_decisions(expected, sources[key[0]])))
                if differences:
                    raise RuntimeError('prepared inference changed calibrated decisions')
                parity.append({'role': key[0], 'path': key[1], 'batch': n, 'mode': mode,
                    'maximum_probability_error': error, 'decision_disagreements': differences,
                    'first_ms': first_ms})
                del out, actual, expected
                for _ in range(job['warmup']):
                    out = call(key)
                    sync()
                    del out
            samples = {key: [] for key in keys}
            orders = []
            rng = np.random.default_rng(job['seed']+n)
            for _ in range(job['repeats']):
                order = rng.permutation(len(keys)).tolist()
                orders.append([list(keys[i]) for i in order])
                for i in order:
                    key = keys[i]
                    sync()
                    t = time.perf_counter_ns()
                    out = call(key)
                    sync()
                    samples[key].append((time.perf_counter_ns()-t)/1e6)
                    del out
            for role, path in keys:
                records.append({'role': role, 'path': path, 'batch': n,
                    'batch_scope': 'all' if n == len(x) else str(n), 'mode': mode,
                    **timing_summary(samples[role, path], n), 'orders': orders})
    return {'timings': records, 'parity': parity, 'setup': setup, 'threadpools': threadpool_info()}


def memory(job, cp=None):
    x = load_inputs(job)[:job['batch']].copy()
    source = ThresholdModel.load(job['models'][job['role']]['path'])
    model, costs = prepare_measured(source, job['path'], job['backend'], cp)
    del source
    if cp:
        device = cp.asarray(x)
        function = lambda: model.predict_device(device)
        sync = cp.cuda.get_current_stream().synchronize
        pool = cp.get_default_memory_pool()
    else:
        function, sync = lambda: model.predict(x), lambda: None
    rows = []
    for phase in ('cold', 'warm'):
        gc.collect()
        if cp:
            pool.free_all_blocks()
        reset = reset_rss_peak()
        before = rss()
        tracemalloc.start()
        if cp:
            baseline = pool.used_bytes()
            allocator = cp.cuda.get_allocator()
            peak_gpu = [baseline]
            def tracked(size):
                pointer = allocator(size)
                peak_gpu[0] = max(peak_gpu[0], pool.used_bytes())
                return pointer
            cp.cuda.set_allocator(tracked)
        try:
            out = function()
            sync()
            after = rss()
            peak = tracemalloc.get_traced_memory()[1]
            row = {'phase': phase, 'rss_before_bytes': before['VmRSS'],
                'rss_peak_bytes': after['VmHWM'], 'rss_peak_reset': reset,
                'rss_increment_bytes': max(0, after['VmHWM']-before['VmRSS']) if reset else None,
                'traced_peak_increment_bytes': peak, 'output_bytes': int(out.nbytes)}
            if cp:
                row.update(gpu_baseline_used_bytes=baseline, gpu_peak_used_bytes=peak_gpu[0],
                    gpu_workspace_peak_bytes=peak_gpu[0]-baseline, gpu_reserved_bytes=pool.total_bytes())
            rows.append(row)
        finally:
            if cp:
                cp.cuda.set_allocator(allocator)
            tracemalloc.stop()
        del out
    return {'measurements': rows, **costs, 'process_isolation': True}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    cp, guard, probe = None, None, []
    bootstrap_started = time.perf_counter()
    bootstrap_before = rss()
    paths = job.get('paths', PATHS)
    if job['backend'] == 'cpu' and ((job['kind'] == 'timing' and set(paths) & {'compiled', 'accumulated'})
                                  or job.get('path') in ('compiled', 'accumulated')):
        import experiments.compiled_leaf_inference  # noqa: F401; outside request/setup
        if 'accumulated' in paths or job.get('path') == 'accumulated':
            import experiments.accumulated_leaf_inference  # noqa: F401
    if job['backend'] == 'gpu':
        from experiments.inference_gpu_guard import gpu_probe, ExclusiveGpuGuard, SharedGpuObserver
        policy = job.get('gpu_policy', 'exclusive')
        free, reason, probe = gpu_probe(policy, job.get('gpu_minimum_free_mib', 2048))
        if not free:
            Path(job['result']).write_text(json.dumps({'gpu_deferred': reason, 'gpu_probe': probe}))
            return
        # Optional PyBoost/ONNX bootstrap is outside timing, affinity and memory.
        if job['kind'] == 'timing' or job.get('path') == 'reference':
            from py_boost.gpu.tree import Tree  # noqa: F401
        import cupy as cp
        cp.cuda.runtime.free(0)
        guard = ExclusiveGpuGuard() if policy == 'exclusive' else SharedGpuObserver()
    bootstrap_ms = (time.perf_counter()-bootstrap_started)*1000
    bootstrap_after = rss()
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    from contextlib import nullcontext
    try:
        with guard if guard else nullcontext():
            with threadpool_limits(limits=1):
                if job.get('gpu_tests'):
                    import pytest
                    if pytest.main(['-q', 'tests/test_prepared_leaf_inference.py',
                                    'tests/test_inference_benchmark.py', '-k', 'gpu']) != 0:
                        raise RuntimeError('guarded GPU regression tests failed')
                if job['kind'] == 'stage_profile':
                    from experiments.gpu_stage_profile import profile_worker
                    result = profile_worker(job, cp)
                else:
                    result = timing(job, cp) if job['kind'] == 'timing' else memory(job, cp)
    except Exception as error:
        if guard and guard.error:
            result = {'gpu_deferred': guard.error}
        else:
            raise error
    result.update(job=job, affinity=sorted(os.sched_getaffinity(0)),
                  gpu_probe=probe, gpu_guard_samples=guard.samples if guard else [],
                  runtime_bootstrap_ms=bootstrap_ms, bootstrap_rss_before_bytes=bootstrap_before['VmRSS'],
                  bootstrap_rss_after_bytes=bootstrap_after['VmRSS'])
    Path(job['result']).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
