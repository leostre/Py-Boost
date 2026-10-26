"""Strong no-fit early-stopping control against compact feedback policies."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, reference_class_model
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants, choice_key
from experiments.sparse_leaf_core import Cost, budget_for, leaf_design
from experiments.parallel_leaf_core import prefix
from experiments.joint_budget_core import choose_joint, choice_record
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.output_calibration_core import calibrated_quality


def prefix_grid(dense_by_depth, calibration_features, calibration_labels, selection_features, selection_labels, config):
    """The function cannot consume test rows or labels."""
    rows, bases = [], {}
    for depth, dense in dense_by_depth.items():
        reference = reference_class_model(dense) if dense.task == 'multiclass' else dense
        entire = ThresholdModel(reference, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32),
                                np.zeros(len(dense.bias), np.float32))
        for trees in config.get('prefix_checkpoints', [40, 80, 120, 160, 200]):
            model = prefix(entire, trees)
            bases[(depth, trees)] = model
            def raw(features):
                return model.raw_from_design(leaf_design(model.base.leaf_indices(features), len(model.base.coefficients)))
            variants = threshold_variants(raw(calibration_features), calibration_labels, raw(selection_features),
                selection_labels, dense.task, config['decision_grid'], config['decision_shrinkage'])
            for v in variants:
                rows.append(dict(candidate_id=f"d{depth}-t{trees}-s{v['threshold_fraction']}", kind='full',
                    depth=depth, trees=trees, leaves=len(model.base.coefficients), internal_nodes=len(model.base.nodes),
                    rank=model.base.coefficients.shape[1], nnz=0, parameters=model.parameter_count,
                    payload_bytes=model.payload_bytes, **v))
    return rows, bases


def prefix_budget_study(config, report, output):
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_prefix_budget_study.py',
                    'tests/test_parallel_leaf_core.py', 'tests/test_uniform_threshold_core.py'],
                   cwd=ROOT, check=True, timeout=300)
    manifest = expanded_manifest(config)
    path = Path(config['joint_development_manifest'])
    if sha256(path) != config['joint_development_manifest_sha256']:
        raise RuntimeError('prefix control parent changed')
    parent = json.loads(path.read_text())
    records = []
    with threadpool_limits(limits=1):
        for name in config['datasets']:
            x, y = load_expanded(name, manifest)
            for case in (c for c in manifest['cases'] if c['dataset'] == name):
                check_deadline(config, 2400)
                source = next(r for r in parent['records'] if r['dataset'] == name and r['seed'] == case['seed'])
                directory = output/f"{name}_{case['seed']}"
                directory.mkdir(parents=True, exist_ok=True)
                roles = load_roles(case, len(x))
                dense = {}
                for fit in source['fits']:
                    p = path.parent/fit['artifact']
                    if sha256(p) != fit['sha256']:
                        raise RuntimeError('frozen depth source changed')
                    dense[fit['depth']] = LeafEnsemble.load(p)
                rows, bases = prefix_grid(dense, x[roles['calibration']], y[roles['calibration']],
                    x[roles['selection']], y[roles['selection']], config)
                full_choice = min([r for r in rows if r['depth'] == 5 and r['trees'] == 200], key=choice_key)
                if any(abs(full_choice['validation'][k]-source['validation_reference'][k]) > 1e-7
                       for k in source['validation_reference']):
                    raise RuntimeError('prefix full200 reference differs from H-B2')
                ref_cost = Cost(**source['reference_cost'])
                choices, saved = [], {}
                for fraction in config['budget_fractions']:
                    budget = budget_for(ref_cost, 'joint', fraction)
                    for scope in ('depth5', 'all_depths'):
                        pool = [r for r in rows if scope == 'all_depths' or r['depth'] == 5]
                        choice = choose_joint(pool, source['validation_reference'], budget, 'leaf_only', 0, ref_cost.parameters)
                        row = dict(fraction=fraction, selection_scope=scope, family='leaf_only', cap_percent=0,
                                   budget=asdict(budget), **choice_record(choice))
                        if choice.candidate is not None:
                            chosen = choice.candidate
                            cid = chosen['candidate_id']
                            if cid not in saved:
                                model = replace(bases[(chosen['depth'], chosen['trees'])],
                                    decision_shifts=np.array(chosen['decision_shifts'], np.float32))
                                p = directory/(cid+'.npz')
                                model.save(p)
                                loaded = ThresholdModel.load(p)
                                design = leaf_design(loaded.base.leaf_indices(x[roles['selection']]), len(loaded.base.coefficients))
                                q = calibrated_quality(y[roles['selection']], loaded.raw_from_design(design),
                                                         case['task'], loaded.decision_shifts)
                                if any(abs(q[k]-chosen['validation'][k]) > 1e-7 for k in q):
                                    raise RuntimeError('prefix chosen quality replay failed')
                                saved[cid] = dict(artifact=p.relative_to(output).as_posix(), sha256=sha256(p),
                                    coo_bytes=loaded.payload_bytes, csr_bytes=loaded.prepare_inference().resident_bytes)
                            row.update(saved[cid])
                        choices.append(row)
                record = dict(dataset=name, seed=case['seed'], choices=choices, candidate_rows=rows,
                    reference_cost=source['reference_cost'], validation_reference=source['validation_reference'],
                    test_evaluated=False)
                write_report(directory/'selection.json', record)
                records.append({k:v for k,v in record.items() if k != 'candidate_rows'})
                write_report(output/'prefix_development_manifest.json', dict(records=records, test_evaluated=False))
                emit('prefix_case_sealed', dataset=name, seed=case['seed'], candidates=len(rows), models=len(saved),
                     test_evaluated=False)
                del rows, bases, dense
                gc.collect()
    report.update(status='development_complete', records=records, tree_fits=0, basis_fits=0,
                  test_evaluated=False, development_manifest_sha256=sha256(output/'prefix_development_manifest.json'),
                  elapsed_seconds=time.perf_counter()-started)
    emit('prefix_development_complete', cases=len(records), tree_fits=0, basis_fits=0, seconds=report['elapsed_seconds'])
