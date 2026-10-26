"""Thin effect shell shared by the expanded joint-budget studies."""
import gc
import json
from pathlib import Path
import time

import numpy as np

from experiments.a100_runner import sha256, build_model
from experiments.shared_leaf_basis import LeafEnsemble, export_teacher
from experiments.expanded_budget_data import validate_roles


def expanded_manifest(config):
    path = Path(config['expanded_manifest'])
    if sha256(path) != config['expanded_manifest_sha256']:
        raise RuntimeError('expanded data/role manifest changed')
    return json.loads(path.read_text())


def load_expanded(name, manifest):
    entry = manifest['datasets'][name]
    path = Path(entry['path'])
    if sha256(path) != entry['sha256']:
        raise RuntimeError('expanded dataset changed: '+name)
    with np.load(path, allow_pickle=False) as f:
        return np.ascontiguousarray(f['features'], dtype=np.float32), f['target'].copy()


def load_roles(case, count):
    path = Path(case['split_file'])
    if sha256(path) != case['split_sha256']:
        raise RuntimeError('expanded role partition changed')
    with np.load(path, allow_pickle=False) as f:
        roles = {k: f[k].copy() for k in ('train', 'calibration', 'control', 'selection', 'test')}
    validate_roles(roles, count)
    return roles


def check_deadline(config, reserve_seconds=0):
    if 'autonomous_deadline_utc' not in config:
        if config.get('bounded_tree_execution') is True:
            # Explicit alternative contract: the caller checks a committed tree cap
            # before every fit. Do not resurrect a completed session's calendar limit.
            return
        raise ValueError('a calendar deadline or explicit bounded-tree execution is required')
    from datetime import datetime, timezone
    deadline = datetime.fromisoformat(config['autonomous_deadline_utc'].replace('Z', '+00:00'))
    if (deadline-datetime.now(timezone.utc)).total_seconds() < reserve_seconds:
        raise TimeoutError('autonomous session deadline/reserve reached')


def fit_dense(case, depth, config, x, y, roles, trees=None):
    import cupy as cp
    from py_boost import GradientBoosting
    from experiments.inference_gpu_guard import gpu_probe
    minimum = 16384 if case['dataset'] == 'bibtex' else 8192
    ready, reason, observations = gpu_probe('shared', minimum)
    if not ready:
        raise RuntimeError(reason)
    check_deadline(config, 900)
    class MeasuredBoosting(GradientBoosting):
        def _fit(self, builder, info):
            self.pool = info['mempool']
            self.peak = self.pool.used_bytes()
            allocator = cp.cuda.get_allocator()
            def tracked(size):
                pointer = allocator(size)
                self.peak = max(self.peak, self.pool.used_bytes())
                if self.peak > config.get('own_pool_limit_bytes', 16*1024**3):
                    raise MemoryError('own training pool exceeded preregistered limit')
                return pointer
            # Nested allocator scopes are supported; the global setter is not
            # legal inside GradientBoosting.fit's existing allocator context.
            with cp.cuda.using_allocator(tracked):
                return super()._fit(builder, info)
    template = build_model('full', case['task'], case['outputs'], config, case['seed'],
                           depth, config['lambda_l2'], trees=trees)
    native = MeasuredBoosting(**template.params)
    del template
    started = time.perf_counter()
    native.fit(x[roles['train']], y[roles['train']],
               eval_sets=[dict(X=x[roles['calibration']], y=y[roles['calibration']])])
    cp.cuda.get_current_stream().synchronize()
    dense = export_teacher(native, x.shape[1], case['task'])
    probes = x[roles['control'][:min(64, len(roles['control']))]]
    actual = native.predict(probes)
    actual = actual.get() if hasattr(actual, 'get') else np.asarray(actual)
    error = float(np.max(np.abs(dense.predict(probes)-actual)))
    if error > 2e-5:
        raise RuntimeError('new dense export differs from training predictor')
    record = dict(depth=depth, trees=dense.n_trees, seconds=time.perf_counter()-started,
                  own_pool_peak_bytes=int(native.peak), probability_replay_error=error,
                  leaves=len(dense.coefficients), internal_nodes=len(dense.nodes),
                  gpu_probe=observations)
    del native
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return dense, record


def budget_preflight(config, report, output):
    import subprocess
    import sys
    from experiments.a100_runner import ROOT, emit
    from experiments.inference_gpu_guard import SharedGpuObserver
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_expanded_budget_data.py',
                    'tests/test_sparse_leaf_core.py', 'tests/test_rank_residual_map.py',
                    'tests/test_leaf_budget_runtime.py'],
                   cwd=ROOT, check=True, timeout=600)
    manifest = expanded_manifest(config)
    with SharedGpuObserver() as observer:
        old_case = next(c for c in manifest['cases'] if c['dataset'] == 'yeast' and c['seed'] == 1304)
        x, y = load_expanded('yeast', manifest)
        roles = load_roles(old_case, len(x))
        dense, fit = fit_dense(old_case, 5, config, x, y, roles)
        old_path = Path(config['source_dir'])/'yeast_1304'/'dense_teacher.npz'
        old_hash = sha256(old_path)
        old = LeafEnsemble.load(old_path)
        probes = x[np.r_[roles['calibration'], roles['control']]]
        error = float(np.max(np.abs(dense.predict(probes)-old.predict(probes))))
        structural = all(np.array_equal(getattr(dense, k), getattr(old, k))
                         for k in ('nodes', 'node_offsets', 'leaf_offsets'))
        reuse = structural and error <= 2e-5
        dense.save(output/'refit_yeast_1304.npz')
        if sha256(old_path) != old_hash:
            raise RuntimeError('old teacher changed')
        emit('budget_control_reuse', reusable=reuse, structure_identical=structural,
             max_probability_error=error, fit=fit)
        case = next(c for c in manifest['cases'] if c['dataset'] == 'bibtex')
        x, y = load_expanded('bibtex', manifest)
        roles = load_roles(case, len(x))
        model, capacity = fit_dense(case, 5, config, x, y, roles, trees=6)
        model.save(output/'bibtex_capacity_probe.npz')
        emit('budget_bibtex_capacity', **capacity)
    report.update(status='complete', old_depth5_reusable=reuse, old_structure_identical=structural,
                  old_probability_replay_error=error, old_control_fit=fit, bibtex_capacity=capacity,
                  full_scientific_fits=1, resource_probe_fits=1, resource_probe_trees=6,
                  expanded_manifest_sha256=config['expanded_manifest_sha256'],
                  gpu_observations=observer.samples, elapsed_seconds=time.perf_counter()-started)
    emit('budget_preflight_complete', reusable=reuse, bibtex_pool_peak=capacity['own_pool_peak_bytes'],
         elapsed_seconds=report['elapsed_seconds'])
