"""Only numerical/serialization and synthetic training gate for H6."""
import signal
import subprocess
import sys
import time

from experiments.a100_runner import ROOT,emit,write_report


def history_basis_preflight(config,report,output):
    from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver
    ready,reason,probe = gpu_probe('shared',config['inference_gpu_minimum_free_mib'])
    if not ready:
        raise RuntimeError(reason)
    started = time.perf_counter()
    signal.alarm(900)
    guard = SharedGpuObserver()
    try:
        with guard:
            subprocess.run([sys.executable,'-m','pytest','-q','tests/test_gradient_basis_core.py',
                'tests/test_block_leaf_model.py','tests/test_history_leaf_gpu.py','tests/test_online_leaf_core.py'],
                cwd=ROOT,check=True,timeout=840)
        report.update(status='done',scientific_fits=0,synthetic_gate='passed',
            elapsed_seconds=time.perf_counter()-started,gpu_probe=probe,gpu_guard_samples=guard.samples,
            interpretation='History moment/multibasis correctness only; no benefit claim yet')
        write_report(output/'report.json',report)
        emit('history_basis_preflight_complete',status='passed',scientific_fits=0,
             elapsed_seconds=report['elapsed_seconds'],gpu_observations=len(guard.samples))
    finally:
        signal.alarm(0)
