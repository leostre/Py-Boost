"""Bounded inference-only benchmark on 45 existing, hash-locked artifacts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.inference_benchmark_core import ROLES, frozen_choices, batch_sizes, paired_ratios


def run_worker(job, output, number):
    request = output/'jobs'/f'{number:04d}.json'
    result = output/'measurements'/f'{number:04d}.json'
    job = {**job, 'result': str(result)}
    write_report(request, job)
    subprocess.run([sys.executable, '-m', 'experiments.inference_benchmark_worker', str(request)],
                   cwd=ROOT, check=True, timeout=180)
    return json.loads(result.read_text())


def summary(timings):
    out = []
    for name, backend, batch, mode in sorted({(r['dataset'], r['backend'], r['batch'], r['mode'])
                                             for r in timings}):
        rows = [r for r in timings if (r['dataset'], r['backend'], r['batch'], r['mode'])
                == (name, backend, batch, mode)]
        for role in ('rank', 'hybrid'):
            pairs = []
            for seed in sorted({r['seed'] for r in rows}):
                reference = next(r for r in rows if r['seed'] == seed and r['role'] == 'full')
                candidate = next(r for r in rows if r['seed'] == seed and r['role'] == role)
                pairs.append(paired_ratios(reference['samples_ms'], candidate['samples_ms']))
            out.append({'dataset': name, 'backend': backend, 'batch': batch, 'mode': mode,
                'role': role, 'pairs': len(pairs), 'median_time_ratio': float(np.median(pairs)),
                'minimum_time_ratio': float(min(pairs)), 'maximum_time_ratio': float(max(pairs))})
    return out


def inference_benchmark_study(config, report, output):
    signal.alarm(config['walltime_seconds'])
    started = time.perf_counter()
    report.update(status='running', teacher_fits=0, basis_fits=0, threshold_fits=0,
        gpu_training_runs=0, inference_gpu=True,
        interpretation='Current CPU code and unoptimized GPU design/COO reference port; not native deployment kernels')
    (output/'jobs').mkdir()
    (output/'measurements').mkdir()
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_inference_benchmark.py',
            'tests/test_shared_leaf_basis.py', 'tests/test_sparse_leaf_core.py',
            'tests/test_uniform_threshold_core.py'], cwd=ROOT, check=True)
        parent = Path(config['inference_parent_dir'])
        manifest_file = parent/'selection_manifest.json'
        if sha256(manifest_file) != config['inference_parent_manifest_sha256']:
            raise RuntimeError('frozen parent manifest changed')
        previous = json.loads((parent/'report.json').read_text())
        manifest = json.loads(manifest_file.read_text())
        data_manifest = json.loads((ROOT/'experiments/a100_data_manifest.json').read_text())
        cases, locks = [], {str(manifest_file): sha256(manifest_file)}
        for record in manifest['selections']:
            name, seed = record['dataset'], record['seed']
            if name not in config['datasets'] or seed not in config['seeds_by_dataset'][name]:
                continue
            data_file = Path(config['data_dir'])/f'{name}.npz'
            if sha256(data_file) != data_manifest[name]['sha256'] or record['data_sha256'] != data_manifest[name]['sha256']:
                raise RuntimeError('data fingerprint changed')
            source = Path(config['source_dir'])/f'{name}_{seed}'
            split_file = source/'splits.npz'
            locks[str(data_file)], locks[str(split_file)] = sha256(data_file), sha256(split_file)
            choices = frozen_choices(record)
            models = {}
            for role, choice in choices.items():
                path = parent/choice['artifact']
                if sha256(path) != choice['sha256']:
                    raise RuntimeError('selected saved model changed')
                locks[str(path)] = choice['sha256']
                old = next(r for r in previous['full_records'] if r['dataset'] == name and r['seed'] == seed) if role == 'full' else next(
                    r for r in previous['records'] if r['dataset'] == name and r['seed'] == seed
                    and r['budget']['kind'] == 'joint' and r['budget']['fraction'] == .75
                    and r['family'] == ('rank_only' if role == 'rank' else 'sparse_05'))
                models[role] = {'path': str(path), 'sha256': choice['sha256'],
                    'candidate': choice['candidate'], 'prior_test_quality': old['test'],
                    'prior_joint_success': True if role == 'full' else old['joint_success']}
            with np.load(split_file, allow_pickle=False) as f:
                n = len(f['test'])
                test_sha = __import__('hashlib').sha256(f['test'].tobytes()).hexdigest()
            cases.append({'dataset': name, 'seed': seed, 'data_file': str(data_file),
                'split_file': str(split_file), 'test_index_sha256': test_sha,
                'models': models, 'test_rows': n, 'batches': batch_sizes(n)})
        assert len(cases) == 15 and sum(len(c['models']) for c in cases) == 45
        sealed = {'cases': cases, 'file_locks': locks, 'new_quality_selection': False,
                  'test_labels_used': False, 'parent_manifest_sha256': sha256(manifest_file)}
        write_report(output/'benchmark_manifest.json', sealed)
        manifest_sha = sha256(output/'benchmark_manifest.json')
        emit('benchmark_sealed', cases=15, saved_model_entries=45, sha256=manifest_sha,
             repeats=config['inference_repeats'], warmup=config['inference_warmup'],
             cpu_threads=1, test_labels_used=False)
        timings, memories, profiles, parity = [], [], [], []
        number = 0
        for case in cases:
            common = {k: case[k] for k in ('dataset', 'seed', 'data_file', 'split_file', 'models')}
            for backend in ('cpu', 'gpu'):
                number += 1
                result = run_worker({**common, 'kind': 'timing', 'backend': backend,
                    'repeats': config['inference_repeats'], 'warmup': config['inference_warmup']}, output, number)
                for key, target in (('timings', timings), ('profiles', profiles), ('parity', parity)):
                    target.extend({**r, 'dataset': case['dataset'], 'seed': case['seed'], 'backend': backend}
                                  for r in result[key])
                emit('timing_complete', dataset=case['dataset'], seed=case['seed'], backend=backend,
                     maximum_probability_error=max(r['maximum_probability_error'] for r in result['parity']))
                for role in ROLES:
                    for batch in (1, case['test_rows']):
                        number += 1
                        measured = run_worker({**common, 'kind': 'memory', 'backend': backend,
                            'role': role, 'model': case['models'][role], 'batch': batch}, output, number)
                        memories.extend({**r, 'dataset': case['dataset'], 'seed': case['seed'],
                            'backend': backend, 'role': role, 'batch': batch,
                            'payload_bytes': measured['payload_bytes']} for r in measured['measurements'])
                emit('memory_complete', dataset=case['dataset'], seed=case['seed'], backend=backend,
                     elapsed_seconds=round(time.perf_counter()-started, 2))
                write_report(output/'progress.json', {'status': 'running', 'workers_done': number,
                    'timings': timings, 'memory': memories, 'parity': parity})
        for path, expected in locks.items():
            if sha256(path) != expected:
                raise RuntimeError('input file changed during measurements')
        if sha256(output/'benchmark_manifest.json') != manifest_sha:
            raise RuntimeError('benchmark manifest changed')
        summaries = summary(timings)
        report.update(status='done', timings=timings, memory=memories, profiles=profiles,
            parity=parity, summaries=summaries, benchmark_manifest_sha256=manifest_sha,
            file_locks_verified=len(locks), workers=number, elapsed_seconds=time.perf_counter()-started,
            hardware={'uname': os.uname()._asdict() if hasattr(os.uname(), '_asdict') else list(os.uname()),
                      'cpu': subprocess.check_output(['lscpu'], text=True),
                      'gpu': subprocess.check_output(['nvidia-smi'], text=True)})
        write_report(output/'report.json', report)
        emit('inference_complete', workers=number, teacher_fits=0, basis_fits=0, threshold_fits=0,
            maximum_probability_error=max(r['maximum_probability_error'] for r in parity),
            decision_disagreements=sum(r['decision_disagreements'] for r in parity), summaries=summaries)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        raise
    finally:
        signal.alarm(0)
