"""One isolated measurement process; called only by the registered orx run."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.inference_benchmark_core import ROLES, batch_sizes, timing_summary
from experiments.uniform_threshold_core import ThresholdModel


def rss():
    entries = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                   if ':' in line)
    return {key: int(entries[key].split()[0])*1024 for key in ('VmRSS', 'VmHWM')}


def reset_rss_peak():
    try:
        Path('/proc/self/clear_refs').write_text('5')
        return True
    except OSError:
        return False


def gpu_status():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def load_inputs(job):
    with np.load(job['data_file'], allow_pickle=False) as f:
        features = f['features']
    with np.load(job['split_file'], allow_pickle=False) as f:
        test = f['test']
    return np.ascontiguousarray(features[test], dtype=np.float32)


def cpu_profile(model, x):
    from experiments.shared_leaf_basis import probabilities
    from experiments.sparse_leaf_core import leaf_design
    times = {}
    t = time.perf_counter()
    leaves = model.base.leaf_indices(x)
    times['routing_ms'] = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    design = leaf_design(leaves, len(model.base.coefficients))
    times['design_ms'] = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    raw = model.raw_from_design(design)
    times['leaf_decode_residual_ms'] = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    out = probabilities(raw, model.base.task)
    times['probabilities_ms'] = (time.perf_counter()-t)*1000
    return out, times


def gpu_profile(model, x):
    import cupy as cp
    times = {}
    def part(name, function):
        cp.cuda.get_current_stream().synchronize()
        t = time.perf_counter()
        out = function()
        cp.cuda.get_current_stream().synchronize()
        times[name] = (time.perf_counter()-t)*1000
        return out
    leaves = part('routing_ms', lambda: model.route(x))
    design = part('design_ms', lambda: model.design(leaves))
    raw = part('leaf_decode_residual_ms', lambda: model.raw(design))
    out = part('probabilities_ms', lambda: model.probability(raw))
    return out, times


def timing(job):
    x = load_inputs(job)
    setup = {}
    cpu, gpu = {}, {}
    before = gpu_status() if job['backend'] == 'gpu' else None
    for role in ROLES:
        t = time.perf_counter()
        cpu[role] = ThresholdModel.load(job['models'][role]['path'])
        setup[role] = {'load_ms': (time.perf_counter()-t)*1000}
    if job['backend'] == 'gpu':
        import cupy as cp
        from experiments.inference_benchmark_gpu import GpuThresholdReference
        for role in ROLES:
            cp.cuda.get_current_stream().synchronize()
            t = time.perf_counter()
            gpu[role] = GpuThresholdReference(cpu[role])
            cp.cuda.get_current_stream().synchronize()
            setup[role]['gpu_setup_ms'] = (time.perf_counter()-t)*1000
    records, parity, profiles = [], [], []
    for n in batch_sizes(len(x)):
        host = np.ascontiguousarray(x[:n])
        device = cp.asarray(host) if gpu else None
        modes = ('device', 'host') if gpu else ('host',)
        for mode in modes:
            functions = {role: (lambda role=role: gpu[role].predict_device(device))
                         if mode == 'device' else (lambda role=role: gpu[role].predict(host))
                         if gpu else (lambda role=role: cpu[role].predict(host)) for role in ROLES}
            for role in ROLES:
                sync = cp.cuda.get_current_stream().synchronize if gpu else lambda: None
                sync()
                t = time.perf_counter()
                out = functions[role]()
                sync()
                first = (time.perf_counter()-t)*1000
                expected = cpu[role].predict(host)
                actual = out.get() if mode == 'device' else out
                error = float(np.max(np.abs(expected-actual)))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
                if cpu[role].base.task == 'multiclass':
                    labels_a = (np.log(actual)-cpu[role].decision_shifts).argmax(axis=1)
                    labels_e = (np.log(expected)-cpu[role].decision_shifts).argmax(axis=1)
                else:
                    threshold = 1/(1+np.exp(-cpu[role].decision_shifts.astype(float)))
                    labels_a, labels_e = actual >= threshold, expected >= threshold
                disagreement = int(np.count_nonzero(labels_a != labels_e))
                if disagreement:
                    raise RuntimeError('GPU adapter changed calibrated decisions')
                parity.append({'batch': n, 'mode': mode, 'role': role,
                               'maximum_probability_error': error, 'decision_disagreements': disagreement})
                del out, actual, expected
                for _ in range(job['warmup']):
                    out = functions[role]()
                    sync()
                    del out
                setup[role][f'first_{mode}_{n}_ms'] = first
            samples = {role: [] for role in ROLES}
            orders = []
            rng = np.random.default_rng(job['seed']+n)
            for _ in range(job['repeats']):
                order = rng.permutation(ROLES).tolist()
                orders.append(order)
                for role in order:
                    sync()
                    t = time.perf_counter_ns()
                    out = functions[role]()
                    sync()
                    samples[role].append((time.perf_counter_ns()-t)/1e6)
                    del out
            for role in ROLES:
                records.append({'batch': n, 'mode': mode, 'role': role,
                                **timing_summary(samples[role], n), 'orders': orders})
        for role in ROLES:
            profiled, stages = gpu_profile(gpu[role], device) if gpu else cpu_profile(cpu[role], host)
            actual = profiled.get() if gpu else profiled
            np.testing.assert_allclose(actual, cpu[role].predict(host), atol=2e-5, rtol=0)
            profiles.append({'batch': n, 'role': role, **stages})
            del profiled, actual
    return {'timings': records, 'parity': parity, 'profiles': profiles, 'setup': setup,
            'gpu_before': before, 'gpu_after': gpu_status() if gpu else None,
            'threadpools': threadpool_info()}


def memory(job):
    x = load_inputs(job)[:job['batch']].copy()
    t = time.perf_counter()
    cpu = ThresholdModel.load(job['model']['path'])
    load_ms = (time.perf_counter()-t)*1000
    backend = job['backend']
    if backend == 'gpu':
        import cupy as cp
        from experiments.inference_benchmark_gpu import GpuThresholdReference
        model = GpuThresholdReference(cpu)
        device = cp.asarray(x)
        pool = cp.get_default_memory_pool()
        function = lambda: model.predict_device(device)
        sync = cp.cuda.get_current_stream().synchronize
    else:
        function, sync = lambda: cpu.predict(x), lambda: None
    measured = []
    for phase in ('cold', 'warm'):
        gc.collect()
        if backend == 'gpu':
            pool.free_all_blocks()
        reset = reset_rss_peak()
        before = rss()
        tracemalloc.start()
        baseline_trace = tracemalloc.get_traced_memory()[0]
        if backend == 'gpu':
            baseline_gpu = pool.used_bytes()
            previous_allocator = cp.cuda.get_allocator()
            peak = [baseline_gpu]
            def tracked_allocator(size):
                pointer = previous_allocator(size)
                peak[0] = max(peak[0], pool.used_bytes())
                return pointer
            cp.cuda.set_allocator(tracked_allocator)
        try:
            out = function()
            sync()
            after = rss()
            _, peak_trace = tracemalloc.get_traced_memory()
            row = {'phase': phase, 'rss_before_bytes': before['VmRSS'],
                'rss_peak_bytes': after['VmHWM'], 'rss_peak_reset': reset,
                'rss_increment_bytes': max(0, after['VmHWM']-before['VmRSS']) if reset else None,
                'traced_peak_increment_bytes': max(0, peak_trace-baseline_trace),
                'output_bytes': int(out.nbytes)}
            if backend == 'gpu':
                row.update(gpu_baseline_used_bytes=int(baseline_gpu),
                    gpu_peak_used_bytes=int(peak[0]),
                    gpu_workspace_peak_bytes=int(peak[0]-baseline_gpu),
                    gpu_reserved_bytes=int(pool.total_bytes()))
            measured.append(row)
        finally:
            if backend == 'gpu':
                cp.cuda.set_allocator(previous_allocator)
            tracemalloc.stop()
        del out
    return {'measurements': measured, 'load_ms': load_ms,
            'payload_bytes': cpu.payload_bytes, 'process_isolation': True}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    bootstrap_started = time.perf_counter()
    if job['backend'] == 'gpu':
        # Py-Boost's package initializer also imports optional inference tools.
        # Load the runtime before pinning and before request-memory boundaries.
        from py_boost.gpu.tree import Tree  # noqa: F401
    bootstrap_ms = (time.perf_counter()-bootstrap_started)*1000
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    with threadpool_limits(limits=1):
        result = timing(job) if job['kind'] == 'timing' else memory(job)
    result.update(job=job, affinity=sorted(os.sched_getaffinity(0)), runtime_bootstrap_ms=bootstrap_ms)
    Path(job['result']).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
