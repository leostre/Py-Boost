"""Profile H-K2 unchanged operators, not a replacement performance baseline."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.inference_benchmark_core import ROLES, batch_sizes


def staged_request(model, x, cp):
    """CUDA event intervals include stream idle time caused by Python dispatch.

    No synchronization between gather/GEMM/finish. Validation's necessary host
    synchronization is unchanged. Allocation and validation have wall timers.
    Events perturb the request: report an uninstrumented request separately.
    """
    from cupy.cublas import gemm
    start = time.perf_counter_ns()
    if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != model.n_features or not x.flags.c_contiguous:
        raise ValueError('invalid resident features')
    if bool(cp.isinf(x).any()):
        raise ValueError('invalid features')
    checked = time.perf_counter_ns()
    out = cp.empty((len(x), model.outputs), cp.float64)
    low_rank = model.representation == 'low_rank'
    latent = cp.empty((len(x), model.width), cp.float64) if low_rank else out
    allocated = time.perf_counter_ns()
    events = [cp.cuda.Event() for _ in range(4)]
    events[0].record()
    model.kernel((len(x),), (model.threads,), (x, model._nodes, model._node_offsets,
        model._leaf_offsets, model._coefficients, model._indptr, model._columns,
        model._values, latent, out, *(np.int32(v) for v in
        (len(x), model.n_features, model.n_trees, model.width, model.outputs,
         model.has_residual, latent.shape[1], low_rank))),
        shared_mem=model.outputs*8 if model.has_residual else 0)
    events[1].record()
    if low_rank:
        result = gemm('N', 'T', latent, model._decoder, out=out,
                      alpha=1., beta=1. if model.has_residual else 0.)
        if result.data.ptr != out.data.ptr:
            raise RuntimeError('decoder copied destination')
    events[2].record()
    model.finish((len(x),), (model.threads,), (out, model._bias, out, out,
        *(np.int32(v) for v in (model.outputs, model.outputs, 0,
                               model.task == 'multiclass'))), shared_mem=model.threads*8)
    events[3].record()
    events[3].synchronize()
    elapsed = (time.perf_counter_ns()-start)/1e6
    return out, dict(validation_wall_ms=(checked-start)/1e6,
        allocation_wall_ms=(allocated-checked)/1e6,
        gather_event_ms=float(cp.cuda.get_elapsed_time(events[0], events[1])),
        decoder_event_ms=float(cp.cuda.get_elapsed_time(events[1], events[2])) if low_rank else 0.,
        finish_event_ms=float(cp.cuda.get_elapsed_time(events[2], events[3])),
        pipeline_event_ms=float(cp.cuda.get_elapsed_time(events[0], events[3])),
        instrumented_wall_ms=elapsed)


def profile_worker(job, cp):
    from experiments.inference_benchmark_worker import load_inputs
    from experiments.prepared_inference_worker import calibrated_decisions, prepare_measured
    from experiments.uniform_threshold_core import ThresholdModel
    x = load_inputs(job)
    sources = {r: ThresholdModel.load(job['models'][r]['path']) for r in ROLES}
    models, setup = {}, []
    for role in ROLES:
        models[role], costs = prepare_measured(sources[role], 'accumulated', 'gpu', cp)
        setup.append(dict(role=role, **costs))
    records, parity = [], []
    sync = cp.cuda.get_current_stream().synchronize
    for n in batch_sizes(len(x)):
        host = np.ascontiguousarray(x[:n])
        device = cp.asarray(host)
        for role in ROLES:
            model, source = models[role], sources[role]
            expected = source.predict(host)
            out, _ = staged_request(model, device, cp)
            actual = out.get()
            error = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
            differences = int(np.count_nonzero(calibrated_decisions(actual, source) !=
                                               calibrated_decisions(expected, source)))
            if differences:
                raise RuntimeError('profile changed calibrated decisions')
            parity.append(dict(role=role, batch=n, maximum_probability_error=error,
                               decision_disagreements=differences))
            del out, actual, expected
            for _ in range(job['warmup']):
                out, _ = staged_request(model, device, cp)
                del out
                out = model.predict_device(device)
                sync()
                del out
        samples = {r: [] for r in ROLES}
        rng = np.random.default_rng(job['seed']+n)
        for _ in range(job['repeats']):
            for role in rng.permutation(ROLES):
                row = {}
                # Alternate instrumentation and unchanged request to limit drift.
                for mode in rng.permutation(['staged', 'device', 'host']):
                    sync()
                    t = time.perf_counter_ns()
                    if mode == 'staged':
                        out, row_stages = staged_request(models[role], device, cp)
                        row.update(row_stages)
                    else:
                        out = (models[role].predict_device(device) if mode == 'device'
                               else models[role].predict(host))
                        sync()
                        row[mode+'_wall_ms'] = (time.perf_counter_ns()-t)/1e6
                    del out
                samples[role].append(row)
        for role in ROLES:
            records.append(dict(role=role, batch=n, batch_scope='all' if n == len(x) else str(n),
                rank=models[role].width if models[role].representation == 'low_rank' else None,
                outputs=models[role].outputs, trees=models[role].n_trees, samples=samples[role],
                medians={k: float(np.median([r[k] for r in samples[role]])) for k in samples[role][0]}))
    return dict(stages=records, parity=parity, setup=setup)


def gpu_stage_study(config, report, output):
    started = time.perf_counter()
    signal.alarm(config['walltime_seconds'])
    report.update(status='running', teacher_fits=0, basis_fits=0, threshold_fits=0,
                  test_labels_used=False)
    try:
        manifest_path = Path(config['prepared_source_manifest'])
        if sha256(manifest_path) != config['prepared_source_manifest_sha256']:
            raise RuntimeError('source manifest changed')
        manifest = json.loads(manifest_path.read_text())
        cases = manifest['cases']
        if len(cases) != 15 or sum(len(c['models']) for c in cases) != 45:
            raise RuntimeError('frozen cohort changed')
        locks = {**manifest['file_locks'], str(manifest_path): sha256(manifest_path)}
        def verify():
            for path, expected in locks.items():
                if sha256(path) != expected:
                    raise RuntimeError(f'frozen input changed: {path}')
        verify()
        write_report(output/'benchmark_manifest.json', manifest)
        results = []
        for number, case in enumerate(cases, 1):
            request = output/'jobs'/f'{number:04d}.json'
            result = output/'measurements'/f'{number:04d}.json'
            job = {k: case[k] for k in ('dataset', 'seed', 'data_file', 'split_file', 'models')}
            job.update(kind='stage_profile', backend='gpu', paths=['accumulated'],
                gpu_tests=number == 1, repeats=config['inference_repeats'],
                warmup=config['inference_warmup'], gpu_policy=config['inference_gpu_policy'],
                gpu_minimum_free_mib=config['inference_gpu_minimum_free_mib'], result=str(result))
            write_report(request, job)
            result.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'experiments.prepared_inference_worker', str(request)],
                           check=True, cwd=ROOT, timeout=240)
            measured = json.loads(result.read_text())
            if measured.get('gpu_deferred'):
                raise RuntimeError(measured['gpu_deferred'])
            results.append(dict(dataset=case['dataset'], seed=case['seed'], **measured))
            write_report(output/'progress.json', dict(workers=number, results=results))
            emit('stage_profile_case', dataset=case['dataset'], seed=case['seed'], workers=number,
                 stages=measured['stages'])
        verify()
        summary = []
        for dataset in config['datasets']:
            for role in ROLES:
                for batch in ('1', '32', '256', 'all'):
                    rows = [r for v in results if v['dataset'] == dataset for r in v['stages']
                            if r['role'] == role and r['batch_scope'] == batch]
                    summary.append(dict(dataset=dataset, role=role, batch_scope=batch,
                        medians={k: float(np.median([r['medians'][k] for r in rows]))
                                 for k in rows[0]['medians']}))
        report.update(status='done', workers=len(results), results=results, summary=summary,
            file_locks_verified=len(locks), elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json', report)
        emit('stage_profile_complete', workers=len(results), file_locks_verified=len(locks),
            maximum_probability_error=max(r['maximum_probability_error'] for v in results for r in v['parity']),
            decision_disagreements=sum(r['decision_disagreements'] for v in results for r in v['parity']),
            summary=summary)
    finally:
        signal.alarm(0)
