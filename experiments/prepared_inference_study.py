"""Compare two implementations, never reselect models or consult test labels."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.inference_benchmark_core import ROLES, paired_ratios
from experiments.inference_gpu_guard import gpu_probe


def summarize(timings):
    results = []
    groups = sorted({(r['dataset'], r['backend'], r['batch_scope'], r['mode']) for r in timings})
    for dataset, backend, batch, mode in groups:
        rows = [r for r in timings if (r['dataset'], r['backend'], r['batch_scope'], r['mode']) ==
                (dataset, backend, batch, mode)]
        seeds = sorted({r['seed'] for r in rows})
        lookup = {(r['seed'], r['role'], r['path']): r for r in rows}
        paths = {r['path'] for r in rows}
        for role in ROLES:
            comparisons = [
                ('prepared', role, 'reference', 'prepared_vs_reference'),
                ('prepared', 'full', 'prepared', 'prepared_vs_prepared_full'),
                ('reference', 'full', 'reference', 'reference_vs_reference_full')] if {'reference', 'prepared'} <= paths else []
            for path in sorted(paths-{'reference', 'prepared'}):
                controls = ('reference', 'prepared', 'compiled' if backend == 'cpu' else 'fused')
                if path == 'unified':
                    controls += ('accumulated',)
                for control in controls:
                    if control in paths and control != path:
                        comparisons.append((path, role, control, f'{path}_vs_{control}'))
                comparisons.append((path, 'full', path, f'{path}_vs_{path}_full'))
            for path, control_role, control_path, comparison in comparisons:
                ratios = [paired_ratios(lookup[s, control_role, control_path]['samples_ms'],
                                       lookup[s, role, path]['samples_ms']) for s in seeds]
                results.append({'dataset': dataset, 'backend': backend, 'batch_scope': batch,
                    'mode': mode, 'role': role, 'comparison': comparison, 'pairs': len(seeds),
                    'median_time_ratio': float(np.median(ratios)), 'min_time_ratio': min(ratios),
                    'max_time_ratio': max(ratios), 'per_seed_ratios': dict(zip(map(str, seeds), ratios)),
                    'candidate_median_ms': float(np.median([lookup[s, role, path]['median_ms'] for s in seeds]))})
    return results


def prepared_inference_study(config, report, output):
    started = time.perf_counter()
    signal.alarm(config['walltime_seconds'])
    report.update(status='running', teacher_fits=0, basis_fits=0, threshold_fits=0,
                  test_labels_used=False, gpu_status='not_attempted',
                  gpu_policy=config.get('inference_gpu_policy', 'exclusive'))
    timings, memory, parity, setup, gpu_checks = [], [], [], [], []
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_prepared_leaf_inference.py',
            'tests/test_inference_benchmark.py', 'tests/test_shared_leaf_basis.py',
            'tests/test_sparse_leaf_core.py', 'tests/test_uniform_threshold_core.py',
            '-k', 'not gpu and not actual_cupy'], cwd=ROOT, check=True)
        source = Path(config['prepared_source_manifest'])
        if sha256(source) != config['prepared_source_manifest_sha256']:
            raise RuntimeError('previous benchmark cohort changed')
        manifest = json.loads(source.read_text())
        cases = manifest['cases']
        if len(cases) != 15 or sum(len(c['models']) for c in cases) != 45:
            raise RuntimeError('cohort is not the previous 45 entries')
        locks = {**manifest['file_locks'], str(source): sha256(source)}
        for path, expected in locks.items():
            if sha256(path) != expected:
                raise RuntimeError(f'frozen input changed: {path}')
        write_report(output/'benchmark_manifest.json', manifest)
        number = 0
        def worker(job):
            nonlocal number
            number += 1
            request = output/'jobs'/f'{number:04d}.json'
            result = output/'measurements'/f'{number:04d}.json'
            write_report(request, {**job, 'result': str(result)})
            result.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'experiments.prepared_inference_worker', str(request)],
                           cwd=ROOT, check=True, timeout=240)
            return json.loads(result.read_text())
        emit('prepared_cohort_sealed', cases=15, saved_entries=45, file_locks=len(locks),
             source_manifest_sha256=sha256(source), teacher_fits=0, threshold_fits=0)
        # Fail fast on device regressions before repeating the CPU cohort.
        for backend in config.get('inference_backends', ('gpu', 'cpu')):
            if backend == 'gpu':
                free, reason, samples = gpu_probe(report['gpu_policy'], config.get('inference_gpu_minimum_free_mib', 2048))
                gpu_checks.append({'stage': 'before_gpu_suite', 'eligible': free, 'reason': reason,
                                   'policy': report['gpu_policy'], 'samples': samples})
                if not free:
                    report['gpu_status'] = 'deferred_gpu_resource_gate'
                    emit('gpu_deferred', reason=reason, samples=samples)
                    continue
                # All real GPU tests execute inside the same guarded timing worker,
                # not as an unguarded pytest process before/after an idle check.
            for case in cases:
                common = {k: case[k] for k in ('dataset', 'seed', 'data_file', 'split_file', 'models')}
                common.update(paths=config.get(f'inference_{backend}_paths', ['reference', 'prepared']),
                              gpu_policy=report['gpu_policy'],
                              gpu_minimum_free_mib=config.get('inference_gpu_minimum_free_mib', 2048))
                tags = {k: case[k] for k in ('dataset', 'seed')}
                measured = worker({**common, 'kind': 'timing', 'backend': backend,
                    'gpu_tests': backend == 'gpu' and case is cases[0],
                    'repeats': config['inference_repeats'], 'warmup': config['inference_warmup']})
                if measured.get('gpu_deferred'):
                    gpu_checks.append(measured)
                    report['gpu_status'] = 'deferred_gpu_resource_gate'
                    break
                for key, target in (('timings', timings), ('parity', parity), ('setup', setup)):
                    target.extend({**r, **tags, 'backend': backend} for r in measured[key])
                if backend == 'gpu':
                    gpu_checks.append({'stage': 'timing', **tags, 'samples': measured['gpu_guard_samples']})
                emit('prepared_timing_complete', backend=backend, **tags,
                     max_error=max(r['maximum_probability_error'] for r in measured['parity']))
                blocked = False
                for role in ROLES:
                    for path in common['paths']:
                        for batch in (1, case['test_rows']):
                            measured = worker({**common, 'kind': 'memory', 'backend': backend,
                                'role': role, 'path': path, 'batch': batch})
                            if measured.get('gpu_deferred'):
                                gpu_checks.append(measured)
                                report['gpu_status'], blocked = 'deferred_gpu_resource_gate', True
                                break
                            memory.extend({**r, **tags, 'backend': backend, 'role': role, 'path': path,
                                'batch': batch, 'batch_scope': 'all' if batch == case['test_rows'] else '1',
                                **{k: measured[k] for k in ('prepare_ms', 'prepare_traced_peak_bytes',
                                                           'resident_array_bytes', 'saved_payload_bytes',
                                                           'prepare_rss_before_bytes', 'prepare_rss_after_bytes',
                                                           'prepare_rss_peak_bytes', 'prepare_rss_peak_reset', 'jit_ms',
                                                           'runtime_bootstrap_ms', 'bootstrap_rss_before_bytes',
                                                           'bootstrap_rss_after_bytes', 'kernel_resources')}}
                                for r in measured['measurements'])
                            if backend == 'gpu':
                                gpu_checks.append({'stage': 'memory', **tags, 'role': role, 'path': path,
                                                   'batch': batch, 'samples': measured['gpu_guard_samples']})
                        if blocked:
                            break
                    if blocked:
                        break
                write_report(output/'progress.json', {'workers': number, 'timings': timings,
                    'memory': memory, 'parity': parity, 'setup': setup, 'gpu_checks': gpu_checks})
                emit('prepared_case_complete', backend=backend, **tags, workers=number,
                     elapsed_seconds=round(time.perf_counter()-started, 2))
                if blocked:
                    break
            if backend == 'gpu' and report['gpu_status'] != 'deferred_gpu_resource_gate':
                report['gpu_status'] = 'done_'+report['gpu_policy']
        for path, expected in locks.items():
            if sha256(path) != expected:
                raise RuntimeError(f'input changed during comparison: {path}')
        summaries = summarize(timings)
        report.update(status='done', timings=timings, memory=memory, parity=parity, setup=setup,
            summaries=summaries, gpu_checks=gpu_checks, workers=number,
            file_locks_verified=len(locks), benchmark_manifest_sha256=sha256(output/'benchmark_manifest.json'),
            elapsed_seconds=time.perf_counter()-started,
            cpu_info=subprocess.check_output(['lscpu'], text=True))
        write_report(output/'report.json', report)
        emit('prepared_inference_complete', gpu_status=report['gpu_status'], gpu_policy=report['gpu_policy'], workers=number,
            maximum_probability_error=max(r['maximum_probability_error'] for r in parity),
            decision_disagreements=sum(r['decision_disagreements'] for r in parity),
            file_locks_verified=len(locks), summaries=[r for r in summaries if r['batch_scope'] in ('1', 'all')
                and r['comparison'].split('_vs_')[0] not in ('reference', 'prepared')])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
