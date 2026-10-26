"""H-K5 sealed saved-model banks, isolated workers and real process accounting."""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.model_bank_core import (cohort_choices, executor_name, gpu_items, cpu_items,
                                         owned_bytes, latency_summary)


def locked_json(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise RuntimeError('frozen input changed: '+str(path))
    return json.loads(path.read_text())


def seal_bank(config, output):
    from experiments.leaf_budget_runtime import expanded_manifest
    source = Path(config['joint_development_manifest'])
    original = locked_json(source, config['joint_development_manifest_sha256'])
    pref = Path(config['prefix_development_manifest'])
    prefix = locked_json(pref, config['prefix_development_manifest_sha256'])
    expanded = expanded_manifest(config)
    rows, locks = [], {str(source): sha256(source), str(pref): sha256(pref),
                      config['expanded_manifest']: config['expanded_manifest_sha256']}
    # Dataset order and seeds are fixed in the committed config, not in test scores.
    for name in config['datasets']:
        for seed in config['bank_seeds_by_dataset'][name]:
            record = next(r for r in original['records'] if r['dataset'] == name and r['seed'] == seed)
            precord = next(r for r in prefix['records'] if r['dataset'] == name and r['seed'] == seed)
            case = next(r for r in expanded['cases'] if r['dataset'] == name and r['seed'] == seed)
            for family, choice in cohort_choices(record, precord).items():
                root = pref.parent if family == 'native_prefix' else source.parent
                path = root/choice['artifact']
                if sha256(path) != choice['sha256']:
                    raise RuntimeError('saved bank model changed')
                locks[str(path)] = choice['sha256']
                candidate = choice['candidate']
                rows.append(dict(dataset=name, seed=seed, family=family, task=record['task'],
                    task_id=f'{name}_{seed}', independent_dataset=name,
                    path=str(path), sha256=choice['sha256'],
                    candidate_id=candidate['candidate_id'], validation=candidate['validation'],
                    reference_validation=record['validation_reference'],
                    development_pass=choice['state'] == 'accepted',
                    selection_state=choice['state'], parameters=candidate['parameters'],
                    coo_bytes=choice['coo_bytes'], csr_bytes=choice['csr_bytes'],
                    data=expanded['datasets'][name], split_file=case['split_file'],
                    split_sha256=case['split_sha256']))
    result = dict(records=rows, file_locks=locks, test_evaluated=False,
                  test_labels_used_for_selection=False, scientific_fits=0,
                  interpretation='18 task versions from 6 datasets, not 18 independent datasets')
    write_report(output/'bank_manifest.json', result)
    return result


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Linux RSS unavailable')


class RssPeak:
    def __init__(self):
        self.peak = rss_bytes()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(.005):
            self.peak = max(self.peak, rss_bytes())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, rss_bytes())


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(line.split(',')[0]): int(line.split(',')[1])*1024**2
            for line in text.splitlines() if line.strip() and 'N/A' not in line}


def run_worker(manifest_path, family, backend, output):
    """Only called as a subprocess of the registered fixed scientific command."""
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.prepared_leaf_inference import PreparedLeafInference
    cohort = json.loads(Path(manifest_path).read_text())
    records = [r for r in cohort['records'] if r['family'] == family]
    features, runtimes, measurements, all_items = {}, [], [], []
    before_gpu = gpu_processes() if backend == 'gpu' else {}
    cp, own_pid = None, None
    if backend == 'gpu':
        import cupy as cp
        from experiments.inference_gpu_guard import gpu_probe
        ready, reason, _ = gpu_probe('shared', 2048)
        if not ready:
            raise RuntimeError(reason)
        cp.cuda.Device(0).use()
        cp.zeros(1).sum().item()
        cp.cuda.get_current_stream().synchronize()
        added = set(gpu_processes())-set(before_gpu)
        if len(added) != 1:
            raise RuntimeError('cannot identify own host GPU PID without ambiguity')
        own_pid = added.pop()
        cp.get_default_memory_pool().free_all_blocks()
    else:
        from experiments.compiled_leaf_inference import CompiledLeafInference
    baseline = dict(rss_bytes=rss_bytes(), gpu_process_bytes=gpu_processes().get(own_pid)
                    if cp is not None else None)
    maximum_error = 0.
    with threadpool_limits(limits=1), RssPeak() as peak:
        for record in records:
            name = record['dataset']
            if name not in features:
                dp = Path(record['data']['path'])
                if sha256(dp) != record['data']['sha256']:
                    raise RuntimeError('bank feature input changed')
                with np.load(dp, allow_pickle=False) as f:
                    # Inputs only. No target or test roles loaded by this worker.
                    features[name] = np.ascontiguousarray(f['features'][:1024], dtype=np.float32)
            probes = features[name][:17].copy()
            if sha256(record['path']) != record['sha256']:
                raise RuntimeError('bank artifact changed')
            source = ThresholdModel.load(record['path'])
            expected = source.predict(probes)
            if cp is not None:
                from experiments.traversal_leaf_gpu import TraversalLeafGpu
                from experiments.unified_leaf_gpu import UnifiedLeafGpu
                engine = executor_name(len(source.base.bias))
                runtime = (TraversalLeafGpu(source, 'groups') if engine == 'groups'
                           else UnifiedLeafGpu(source))
                items = gpu_items(runtime)
            else:
                engine = 'compiled_cpu'
                runtime = CompiledLeafInference.prepare(source)
                items = cpu_items(runtime.snapshot)
            np.testing.assert_allclose(runtime.predict(probes), expected, atol=2e-12, rtol=0)
            error = float(np.max(np.abs(runtime.predict(probes)-expected)))
            maximum_error = max(maximum_error, error)
            # Original COO/model objects MUST NOT remain in the serving bank.
            del source, expected, probes
            runtimes.append(runtime)
            all_items.extend([type(item)(record['task_id']+'/'+item.name, item.pointer, item.size)
                              for item in items])
            exact = owned_bytes(items)
            if exact != runtime.resident_bytes:
                raise RuntimeError('runtime resident_bytes does not match owned arrays')
            owned_bytes(all_items)  # Detect cross-model aliases as well.
            gc.collect()
            if cp is not None:
                cp.cuda.get_current_stream().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            memory = dict(rss_bytes=rss_bytes(),
                gpu_process_bytes=gpu_processes().get(own_pid) if cp is not None else None,
                pool_used_bytes=cp.get_default_memory_pool().used_bytes() if cp is not None else None,
                pool_reserved_bytes=cp.get_default_memory_pool().total_bytes() if cp is not None else None)
            measurements.append(dict(task_id=record['task_id'], dataset=name, family=family,
                backend=backend, k=len(runtimes), engine=engine, resident_bytes=exact,
                total_resident_bytes=owned_bytes(all_items),
                development_pass=record['development_pass'], probability_error=error,
                arrays=[asdict(i) for i in items], **memory))
            emit('bank_model_loaded', **{k:v for k,v in measurements[-1].items() if k != 'arrays'})
        request_rows = []
        for record, runtime in zip(records, runtimes):
            x = features[record['dataset']][:32].copy()
            samples = []
            for _ in range(5):
                begin = time.perf_counter_ns()
                result = runtime.predict(x)
                samples.append((time.perf_counter_ns()-begin)/1e6)
                del result
            request_rows.append(dict(task_id=record['task_id'], **latency_summary(samples)))
    result = dict(status='complete', family=family, backend=backend, baseline=baseline,
        measurements=measurements, request_rows=request_rows, maximum_probability_error=maximum_error,
        rss_sampled_peak_bytes=peak.peak, lifetime_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        feature_buffer_bytes=sum(x.nbytes for x in features.values()),
        gpu_host_pid=own_pid, test_labels_loaded=False, scientific_fits=0,
        array_ownership_checked=True, original_coo_models_retained=False,
        shared_gpu_timing=backend == 'gpu')
    write_report(Path(output), result)
    emit('bank_worker_complete', family=family, backend=backend, models=len(runtimes),
         rss_peak_bytes=peak.peak, maximum_probability_error=maximum_error,
         resident_bytes=owned_bytes(all_items), test_labels_loaded=False)


def model_bank_inventory(config, report, output):
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_model_bank_core.py',
                    'tests/test_model_bank_gpu.py'], cwd=ROOT, check=True, timeout=600)
    bank = seal_bank(config, output)
    emit('bank_cohort_sealed', records=len(bank['records']),
         sha256=sha256(output/'bank_manifest.json'), test_evaluated=False)
    results = []
    for backend in ('cpu', 'gpu'):
        for family in ('full', 'rank_residual', 'joint_static', 'native_prefix'):
            path = output/f'inventory_{backend}_{family}.json'
            subprocess.run([sys.executable, '-u', '-m', 'experiments.model_bank_study',
                str(output/'bank_manifest.json'), family, backend, str(path)],
                cwd=ROOT, check=True, timeout=1800)
            results.append(json.loads(path.read_text()))
    for path, digest in bank['file_locks'].items():
        if sha256(path) != digest:
            raise RuntimeError('bank source changed during inventory')
    report.update(status='complete', workers=results, scientific_fits=0, tree_fits=0,
        threshold_fits=0, test_evaluated=False, inventory_sha256=sha256(output/'bank_manifest.json'))
    write_report(output/'report.json', report)
    emit('bank_inventory_complete', workers=len(results), records=len(bank['records']),
         scientific_fits=0, max_probability_error=max(r['maximum_probability_error'] for r in results),
         final_resident_bytes={r['backend']+'/'+r['family']: r['measurements'][-1]['total_resident_bytes']
                               for r in results})


if __name__ == '__main__':
    run_worker(*sys.argv[1:])
