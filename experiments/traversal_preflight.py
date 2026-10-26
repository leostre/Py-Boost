"""Guarded GPU/CPU correctness only; synthetic trees are constructed, not fit."""
import json
from pathlib import Path
import subprocess
import sys

from experiments.a100_runner import ROOT,emit,write_report,sha256
from experiments.inference_gpu_guard import gpu_probe,SharedGpuObserver


def traversal_preflight(config,report,output):
    manifest_path=Path(config['prepared_source_manifest'])
    if sha256(manifest_path)!=config['prepared_source_manifest_sha256']:
        raise RuntimeError('frozen cohort manifest changed')
    manifest=json.loads(manifest_path.read_text())
    locks={**manifest['file_locks'],str(manifest_path):sha256(manifest_path)}
    for path,expected in locks.items():
        if sha256(path)!=expected: raise RuntimeError('frozen input changed: '+path)
    eligible,reason,samples=gpu_probe('shared',config['inference_gpu_minimum_free_mib'])
    if not eligible: raise RuntimeError(reason)
    with SharedGpuObserver() as observer:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_traversal_inference.py'],
                       cwd=ROOT,check=True,timeout=600)
    for path,expected in locks.items():
        if sha256(path)!=expected: raise RuntimeError('frozen input changed: '+path)
    report.update(status='done',scientific_fits=0,tree_fits=0,threshold_fits=0,
                  cases=14,variants=['warp','groups','tiled'],gpu_checks=samples+observer.samples,
                  file_locks_verified=len(locks))
    write_report(output/'report.json',report)
    emit('traversal_correctness_complete',variants=report['variants'],scientific_fits=0,
         tree_fits=0,threshold_fits=0,file_locks_verified=len(locks),
         environment_sha256=report['environment_sha256'])
