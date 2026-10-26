"""One guard decision per immutable node; development only, never test scoring."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.local_guard_core import GuardPolicy
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.closed_budget_study import run_path
from experiments.uniform_threshold_core import ThresholdModel
from experiments.guard_reuse_core import all_pass_equivalent


def locked_json(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise RuntimeError('frozen guard input changed: '+str(path))
    return json.loads(path.read_text()), path


def same_arrays(left, right):
    for name in ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'):
        np.testing.assert_array_equal(getattr(left.base, name), getattr(right.base, name))
    for name in ('rows', 'columns', 'values', 'decision_shifts'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def guard_budget_study(config, report, output):
    from experiments.inference_gpu_guard import SharedGpuObserver
    started = time.perf_counter()
    tests = ['tests/test_local_guard_core.py', 'tests/test_closed_budget_core.py',
             'tests/test_closed_budget_projection.py', 'tests/test_guard_budget_study.py',
             'tests/test_continuation_contract.py', 'tests/test_guard_reuse_core.py']
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT, check=True, timeout=600)
    data = expanded_manifest(config)
    static, static_path = locked_json(config['joint_development_manifest'], config['joint_development_manifest_sha256'])
    baseline, baseline_path = locked_json(config['guard_baseline_manifest'], config['guard_baseline_manifest_sha256'])
    if baseline['test_evaluated'] or baseline['parent_manifest_sha256'] != sha256(static_path):
        raise RuntimeError('guard baseline is not the same frozen development reference')
    baseline_report = json.loads((baseline_path.parent/'report.json').read_text())
    for key in ('lr', 'lambda_l2', 'ntrees', 'decision_grid', 'decision_shrinkage', 'margin_scale', 'own_pool_limit_bytes'):
        if config[key] != baseline_report['config'][key]:
            raise RuntimeError('guard-only comparison changed '+key)
    records, cases = [], []
    strict_entries = []
    if not config.get('guard_gate') and config['guard_policy'] != 'strict' and config.get('guard_strict_manifest'):
        strict, _ = locked_json(config['guard_strict_manifest'], config['guard_strict_manifest_sha256'])
        if strict['test_evaluated'] or strict['static_manifest_sha256'] != sha256(static_path):
            raise RuntimeError('replay reference used a different frozen static control')
        if len(strict['records']) != 36 or any(r['guard_policy'] != 'strict' for r in strict['records']):
            raise RuntimeError('incomplete or non-strict replay reference')
        strict_entries = strict['records']
    with SharedGpuObserver() as observer, threadpool_limits(limits=1):
        if config.get('guard_gate', False):
            case = next(c for c in data['cases'] if c['dataset'] == 'yeast' and c['seed'] == 1304)
            x,y = load_expanded('yeast', data)
            roles = load_roles(case, len(x))
            full = next(r for r in static['records'] if r['dataset']=='yeast' and r['seed']==1304)
            old = next(r for r in baseline['records'] if r['dataset']=='yeast' and r['seed']==1304 and r['method']=='periodic_fixed')
            old_cycle = old['cycles'][0]
            old_path = baseline_path.parent/old_cycle['artifact']
            if sha256(old_path) != old_cycle['sha256']:
                raise RuntimeError('correctness gate old snapshot changed')
            expected = ThresholdModel.load(old_path)
            for policy in GuardPolicy:
                result = run_path('periodic_fixed', case, full, static_path.parent,
                                  config, output, x,y,roles, guard_policy=policy)
                cycle = result['cycles'][0]
                if cycle['guard_action'] != 'confirmed' or result['new_built_trees'] != 40:
                    raise RuntimeError('all-pass guard gate unexpectedly diverged')
                actual = ThresholdModel.load(output/cycle['artifact'])
                same_arrays(expected, actual)
                records.append(dict(allocation_policy='periodic_fixed', guard_policy=policy.value,
                    artifact_root=str(output), reused=False, trajectory=result))
                emit('guard_gate_identical', policy=policy.value, arrays_identical=True,
                     file_identical=sha256(output/cycle['artifact'])==old_cycle['sha256'])
            cases.append(dict(dataset='yeast',seed=1304))
        else:
            policy = GuardPolicy(config['guard_policy'])
            methods = config['guard_allocation_policies']
            if methods != ['periodic_fixed', 'joint_adaptive']:
                raise ValueError('guard-only protocol requires the two registered allocation policies')
            for name in config['datasets']:
                x,y = load_expanded(name, data)
                for case in (c for c in data['cases'] if c['dataset']==name and c['seed'] in config['guard_seeds_by_dataset'][name]):
                    check_deadline(config, 3600)
                    roles = load_roles(case, len(x))
                    full = next(r for r in static['records'] if r['dataset']==name and r['seed']==case['seed'])
                    for allocation in methods:
                        old = next((r for r in baseline['records'] if r['dataset']==name and
                                    r['seed']==case['seed'] and r['method']==allocation), None)
                        strict_source = next((r for r in strict_entries if r['trajectory']['dataset']==name and
                            r['trajectory']['seed']==case['seed'] and r['allocation_policy']==allocation), None)
                        proof = strict_source is not None and all_pass_equivalent(strict_source['trajectory'])
                        reused = (policy is GuardPolicy.STRICT and old is not None) or proof
                        source = strict_source['trajectory'] if proof else old
                        result = source if reused else run_path(allocation, case, full, static_path.parent,
                            config, output, x,y,roles, guard_policy=policy)
                        root = (strict_source['artifact_root'] if proof else str(baseline_path.parent)) if reused else str(output)
                        entry = dict(allocation_policy=allocation, guard_policy=policy.value,
                            artifact_root=root, reused=reused,
                            reuse_reason=('all_pass_equivalence' if proof else ('frozen_strict_reference' if reused else None)),
                            trajectory=result)
                        records.append(entry)
                        emit('guard_path_complete', dataset=name, seed=case['seed'], allocation=allocation,
                             policy=policy.value, reused=reused, selection_state=result['choice']['state'],
                             new_trees=0 if reused else result['new_built_trees'])
                        write_report(output/'progress.json', dict(records=records,test_evaluated=False))
                    cases.append(dict(dataset=name,seed=case['seed']))
    fits = sum(0 if r['reused'] else r['trajectory']['native_fit_jobs'] for r in records)
    trees = sum(0 if r['reused'] else r['trajectory']['new_built_trees'] for r in records)
    if trees > config['guard_node_tree_cap']:
        raise RuntimeError('guard node exceeded preregistered tree cap')
    manifest = dict(version=1, records=records, cases=cases, test_evaluated=False,
        static_manifest=str(static_path), static_manifest_sha256=sha256(static_path),
        baseline_manifest=str(baseline_path), baseline_manifest_sha256=sha256(baseline_path),
        config=config, new_built_trees=trees, native_fit_jobs=fits)
    path = output/'guard_development_manifest.json'
    write_report(path, manifest)
    report.update(status='guard_gate_complete' if config.get('guard_gate') else 'guard_development_complete',
        records=records, cases=cases, test_evaluated=False, native_fit_jobs=fits, new_built_trees=trees,
        reused_paths=sum(r['reused'] for r in records), development_manifest=str(path),
        development_manifest_sha256=sha256(path), gpu_observations=observer.samples,
        elapsed_seconds=time.perf_counter()-started)
    emit('guard_development_complete', cases=len(cases),paths=len(records),new_trees=trees,
         reused_paths=report['reused_paths'],manifest_sha256=sha256(path),test_evaluated=False)
