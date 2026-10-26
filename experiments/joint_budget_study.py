"""H-B2: development map with independent topology and global actual-array caps.

No test evaluation takes place in this module. Selection labels cannot affect
tree fitting, bases, residual scores, calibration, or the frozen full reference.
"""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, reference_class_model, functional_basis
from experiments.sparse_leaf_core import Cost, budget_for, leaf_design
from experiments.uniform_threshold_study import uniform_plan, materialize_uniform
from experiments.uniform_threshold_core import ThresholdModel
from experiments.output_calibration_core import calibrated_quality
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, fit_dense, check_deadline
from experiments.joint_budget_core import choose_joint, choice_record


def prepare_joint_case(case, config, output, x, y, roles):
    started = time.perf_counter()
    case_id = f"{case['dataset']}_{case['seed']}"
    directory = output/case_id
    directory.mkdir(parents=True, exist_ok=True)
    old = case['historical_outer_split']
    source_path = Path(config['source_dir'])/case_id/'dense_teacher.npz'
    locks = {}
    dense_by_depth, fits = {}, []
    # The global reference is always the same depth5 200tree method, never the
    # shallower model currently being compressed.
    for depth in (5, 3, 4):
        check_deadline(config, 1800)
        if depth == 5 and old and config['old_depth5_reusable']:
            locks[str(source_path)] = sha256(source_path)
            dense = LeafEnsemble.load(source_path)
            fit = dict(depth=5, trees=dense.n_trees, reused=True, source=str(source_path),
                       source_sha256=locks[str(source_path)], seconds=0,
                       leaves=len(dense.coefficients), internal_nodes=len(dense.nodes))
        else:
            dense, fit = fit_dense(case, depth, config, x, y, roles)
            fit['reused'] = False
        path = directory/f'dense_depth{depth}.npz'
        dense.save(path)
        fit.update(artifact=str(path.relative_to(output)), sha256=sha256(path))
        dense_by_depth[depth] = dense
        fits.append(fit)
        write_report(directory/'fits.json', fits)
        emit('joint_dense_ready', dataset=case['dataset'], seed=case['seed'], **fit)
    global_dense = dense_by_depth[5]
    global_reference = reference_class_model(global_dense) if case['task'] == 'multiclass' else global_dense
    reference_cost = Cost(global_reference.parameter_count, global_reference.payload_bytes)
    rows, contexts = [], {}
    for depth in (5, 3, 4):
        check_deadline(config, 1800)
        dense = dense_by_depth[depth]
        reference = reference_class_model(dense) if case['task'] == 'multiclass' else dense
        leaves = {k: dense.leaf_indices(x[roles[k]]) for k in ('train', 'calibration', 'selection')}
        mean, basis, singular = functional_basis(dense.contributions(leaves['train']), case['task'])
        np.savez(directory/f'basis_depth{depth}.npz', mean=mean, basis=basis, singular=singular)
        plan = uniform_plan(dense, reference, mean, basis, leaves['train'], y[roles['train']],
            leaves['calibration'], y[roles['calibration']], leaves['selection'], y[roles['selection']],
            reference_cost, config,
            lambda stage, rank, count: emit('joint_map_progress', dataset=case['dataset'], seed=case['seed'],
                                            depth=depth, stage=stage, rank=rank, candidates=count),
            build_selections=False)
        for row in [*plan['full_candidates'], *plan['rows']]:
            row.update(candidate_id=f'd{depth}-'+row['candidate_id'], depth=depth,
                       leaves=len(dense.coefficients), internal_nodes=len(dense.nodes), trees=dense.n_trees)
            rows.append(row)
        contexts[depth] = (plan, dense, reference, mean, basis, leaves['selection'])
        if depth == 5:
            full_choice = plan['full_choice']
    validation_reference = full_choice['validation']
    saved = {}
    def save(chosen):
        cid, depth = chosen['candidate_id'], chosen['depth']
        if cid in saved:
            return saved[cid]
        plan, dense, reference, mean, basis, leaves = contexts[depth]
        model = materialize_uniform(chosen, plan, dense, reference, mean, basis)
        path = directory/(cid+'.npz')
        model.save(path)
        loaded = ThresholdModel.load(path)
        design = leaf_design(leaves, len(dense.coefficients))
        quality = calibrated_quality(y[roles['selection']], loaded.raw_from_design(design), dense.task,
                                     loaded.decision_shifts)
        if any(abs(quality[k]-chosen['validation'][k]) > 1e-7 for k in quality):
            raise RuntimeError('joint chosen-model metric replay mismatch')
        if loaded.parameter_count != chosen['parameters'] or loaded.payload_bytes != chosen['payload_bytes']:
            raise RuntimeError('joint chosen-model storage replay mismatch')
        prepared = loaded.prepare_inference()
        probes = x[roles['selection'][:min(17, len(roles['selection']))]]
        np.testing.assert_allclose(loaded.predict(probes), prepared.predict(probes), atol=2e-12, rtol=0)
        saved[cid] = dict(artifact=str(path.relative_to(output)), sha256=sha256(path),
                          coo_bytes=loaded.payload_bytes, csr_bytes=prepared.resident_bytes,
                          file_bytes=path.stat().st_size)
        return saved[cid]
    full = dict(candidate=full_choice, **save(full_choice))
    decisions = []
    for fraction in config['budget_fractions']:
        budget = budget_for(reference_cost, 'joint', fraction)
        for family in ('leaf_only', 'rank_only', 'rank_residual', 'joint_static'):
            for percent in ((0,) if family in ('leaf_only', 'rank_only') else (0, 1, 2, 5)):
                choice = choose_joint(rows, validation_reference, budget, family, percent, reference_cost.parameters)
                record = dict(fraction=fraction, budget=asdict(budget), family=family, cap_percent=percent,
                              **choice_record(choice))
                if choice.candidate is not None:
                    record.update(save(choice.candidate))
                decisions.append(record)
    for path, expected in locks.items():
        if sha256(path) != expected:
            raise RuntimeError('frozen full teacher modified')
    record = dict(dataset=case['dataset'], seed=case['seed'], task=case['task'],
                  split_file=case['split_file'], split_sha256=case['split_sha256'],
                  data_sha256=case['data_sha256'], reference_cost=asdict(reference_cost),
                  denominator='original exact uncalibrated depth5 model; candidate thresholds counted',
                  validation_reference=validation_reference, full=full, fits=fits,
                  decisions=decisions, candidate_rows=rows, source_locks=locks,
                  test_evaluated=False, prepare_seconds=time.perf_counter()-started)
    write_report(directory/'selection.json', record)
    emit('joint_case_sealed', dataset=case['dataset'], seed=case['seed'], candidates=len(rows),
         decisions=len(decisions), unique_models=len(saved), seconds=record['prepare_seconds'])
    return {k: v for k, v in record.items() if k != 'candidate_rows'}


def joint_budget_study(config, report, output):
    from experiments.inference_gpu_guard import SharedGpuObserver
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_joint_budget_core.py',
                    'tests/test_uniform_threshold_core.py', 'tests/test_leaf_budget_runtime.py',
                    'tests/test_expanded_budget_data.py'], cwd=ROOT, check=True, timeout=600)
    manifest = expanded_manifest(config)
    records, complete = [], True
    with SharedGpuObserver() as observer:
        for name in config['datasets']:
            x, y = load_expanded(name, manifest)
            for case in (c for c in manifest['cases'] if c['dataset'] == name):
                try:
                    check_deadline(config, 2400)
                except TimeoutError:
                    complete = False
                    break
                roles = load_roles(case, len(x))
                records.append(prepare_joint_case(case, config, output, x, y, roles))
                write_report(output/'development_manifest.json', dict(records=records, test_evaluated=False,
                    expanded_manifest_sha256=config['expanded_manifest_sha256']))
                gc.collect()
            if not complete:
                break
    report.update(status='development_complete' if complete else 'partial_deadline', records=records,
                  full_scientific_fits=sum(not f['reused'] for r in records for f in r['fits']),
                  reused_full_controls=sum(f['reused'] for r in records for f in r['fits']),
                  test_evaluated=False, gpu_observations=observer.samples,
                  development_manifest_sha256=sha256(output/'development_manifest.json'),
                  elapsed_seconds=time.perf_counter()-started)
    emit('joint_development_complete', cases=len(records), fits=report['full_scientific_fits'],
         reused=report['reused_full_controls'], test_evaluated=False, seconds=report['elapsed_seconds'])
