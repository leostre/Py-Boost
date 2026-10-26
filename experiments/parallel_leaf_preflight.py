"""Synthetic continuation + real two-process orchestration correctness gate."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import signal
import subprocess
import sys
import time
import numpy as np
from experiments.a100_runner import ROOT,emit,write_report


def parallel_leaf_preflight(config,report,output):
    from experiments.inference_gpu_guard import SharedGpuObserver,gpu_probe
    from experiments.parallel_leaf_worker import fit_increment
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.parallel_leaf_core import average_increments
    from experiments.history_basis_study import raw
    ready,reason,probe = gpu_probe('shared',8192)
    if not ready:
        raise RuntimeError(reason)
    started = time.perf_counter()
    signal.alarm(900)
    guard = SharedGpuObserver()
    try:
        with guard:
            subprocess.run([sys.executable,'-m','pytest','-q','tests/test_parallel_leaf_core.py',
                'tests/test_continuation_leaf_gpu.py','tests/test_history_study.py',
                'tests/test_nominal_rank_budget.py','tests/test_online_leaf_core.py'],cwd=ROOT,check=True,timeout=500)
            rng = np.random.default_rng(412)
            x = rng.normal(size=(256,5)).astype(np.float32)
            y = (x[:,:4]>0).astype(np.float32)
            np.savez(output/'data.npz',train_x=x[:192],train_y=y[:192],validation_x=x[192:],validation_y=y[192:])
            initial = np.tile(np.array([.3,-.2,.5,-.7],np.float32),(256,1))
            np.savez(output/'initial.npz',train=initial[:192],validation=initial[192:].astype(float))
            requests = [dict(data_path=str(output/'data.npz'),initial_path=str(output/'initial.npz'),
                config=config,task='multilabel',seed=601+j,trees=8,lr=.1,output=str(output/f'branch{j}.npz')) for j in range(2)]
            with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                records = list(pool.map(fit_increment,requests))
            assert len({r['pid'] for r in records}) == 2
            assert records[0]['initial_sha256'] == records[1]['initial_sha256']
            first,second = [ThresholdModel.load(output/f'branch{j}.npz') for j in range(2)]
            # Use a zero-output dummy tree checkpoint, keeping nonzero global bias.
            from dataclasses import replace
            checkpoint = replace(first,base=replace(first.base,coefficients=np.zeros_like(first.base.coefficients),
                bias=initial[0].copy()))
            merged = average_increments(checkpoint,[first,second])
            np.testing.assert_allclose(merged.raw(x),initial+.5*raw(first,x)+.5*raw(second,x),atol=2e-14,rtol=0)
        overlap_ns = max(0,min(r['fit_finished_ns'] for r in records)-max(r['fit_started_ns'] for r in records))
        report.update(status='done',scientific_fits=0,synthetic_gate='passed',workers=records,
            worker_fit_overlap_seconds=overlap_ns/1e9,gpu_probe=probe,gpu_guard_samples=guard.samples,
            elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        emit('parallel_preflight_complete',scientific_fits=0,workers=2,fit_overlap_seconds=overlap_ns/1e9)
    finally:
        signal.alarm(0)
