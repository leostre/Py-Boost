"""H-LR2 conditional geometry comparison, no tree fits and no test access."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.shared_leaf_basis import LeafEnsemble, reference_class_model
from experiments.leaf_geometry_core import tree_centered_leaf_basis
from experiments.sparse_leaf_core import (Cost, budget_for, leaf_design, contrast, training_weights,
    coordinate_scores, score_order, make_sparse_model)
from experiments.sparse_leaf_study import base_at_rank
from experiments.uniform_threshold_study import uniform_plan, materialize_uniform
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants
from experiments.joint_budget_core import choose_joint, choice_record
from experiments.output_calibration_core import calibrated_quality


def add_exact_anchor_rows(anchor, plan, dense, mean, basis, designs, y, roles, config):
    if anchor is None or not anchor['nnz']:
        return []
    r, count = anchor['rank'], anchor['nnz']
    base = base_at_rank(dense, mean, basis, r, plan['full'])
    plan['bases'][r] = base
    residual = (contrast(dense.coefficients, dense.task)-
                contrast(base.coefficients.astype(float)@base.basis.T, dense.task)).astype(np.float32)
    teacher = contrast(designs['train']@dense.coefficients+dense.bias, dense.task)
    error = teacher-contrast(designs['train']@base.coefficients@base.basis.T+base.bias, dense.task)
    new_rows = []
    for selector in config['tiny_selectors']:
        key = (r, selector)
        if key not in plan['orders']:
            weights = training_weights(y[roles['train']], teacher, dense.task, selector, config['margin_scale'])
            plan['orders'][key] = score_order(coordinate_scores(designs['train'], residual, error, selector, weights), residual)
        if count > len(plan['orders'][key]):
            continue
        if any(a['rank'] == r and a['selector'] == selector and a['nnz'] == count for a in plan['rows']):
            continue
        leaf = make_sparse_model(base, residual, plan['orders'][key], count, plan['frequency'])
        if len(leaf.values) != count:
            raise RuntimeError('exact geometry allocation lost a nonzero coordinate')
        variants = threshold_variants(leaf.raw_from_design(designs['calibration']), y[roles['calibration']],
            leaf.raw_from_design(designs['selection']), y[roles['selection']], dense.task,
            config['decision_grid'], config['decision_shrinkage'])
        for variant in variants:
            new_rows.append(dict(candidate_id=f"sparse-r{r}-{selector}-{count}-t{variant['threshold_fraction']}",
                kind='sparse', rank=r, selector=selector, nnz=count, parameters=leaf.parameter_count+len(dense.bias),
                payload_bytes=leaf.payload_bytes+4*len(dense.bias), **variant))
    plan['rows'].extend(new_rows)
    return new_rows


def prepare_geometry(case, source, root, config, output, x, y, roles):
    case_id = f"{case['dataset']}_{case['seed']}"
    directory = output/case_id
    directory.mkdir(parents=True, exist_ok=True)
    fit = next(f for f in source['fits'] if f['depth'] == 5)
    path = root/fit['artifact']
    if sha256(path) != fit['sha256']:
        raise RuntimeError('geometry source model changed')
    dense = LeafEnsemble.load(path)
    reference = reference_class_model(dense) if dense.task == 'multiclass' else dense
    ref_cost = Cost(**source['reference_cost'])
    budget = budget_for(ref_cost, 'joint', .75)
    leaves = {k: dense.leaf_indices(x[roles[k]]) for k in ('train', 'calibration', 'selection')}
    designs = {k: leaf_design(v, len(dense.coefficients)) for k, v in leaves.items()}
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    mean = np.asarray(designs['train']@dense.coefficients).mean(axis=0)
    anchors = {family: next(d for d in source['decisions'] if d['fraction'] == .75 and
                           d['family'] == family and d['cap_percent'] == cap)
               for family, cap in [('rank_only', 0), ('rank_residual', 5)]}
    cfg = {**config, 'budget_kinds': ['joint'], 'budget_fractions': [.75]}
    choices, grids = [], {}
    started = time.perf_counter()
    for geometry, weighted in [('leaf_centered', False), ('leaf_visitation', True)]:
        check_deadline(config, 2400)
        computed_mean, basis, singular = tree_centered_leaf_basis(dense.coefficients, dense.leaf_offsets,
                                                                 frequency, dense.task, visitation_weighted=weighted)
        np.testing.assert_allclose(computed_mean, mean, atol=1e-7, rtol=0)
        np.savez(directory/(geometry+'_basis.npz'), mean=mean, basis=basis, singular=singular)
        plan = uniform_plan(dense, reference, mean, basis, leaves['train'], y[roles['train']],
            leaves['calibration'], y[roles['calibration']], leaves['selection'], y[roles['selection']],
            ref_cost, cfg, lambda stage, rank, count: emit('geometry_progress', dataset=case['dataset'],
                seed=case['seed'], geometry=geometry, stage=stage, rank=rank, candidates=count), build_selections=False)
        for anchor in anchors.values():
            add_exact_anchor_rows(anchor['candidate'], plan, dense, mean, basis, designs, y, roles, cfg)
        for row in plan['rows']:
            row.update(candidate_id=geometry+'-'+row['candidate_id'], depth=5,
                       leaves=len(dense.coefficients), internal_nodes=len(dense.nodes), trees=dense.n_trees)
        grids[geometry] = plan['rows']
        saved = {}
        def save(chosen):
            cid = chosen['candidate_id']
            if cid in saved:
                return saved[cid]
            model = materialize_uniform(chosen, plan, dense, reference, mean, basis)
            model_path = directory/(cid+'.npz')
            model.save(model_path)
            loaded = ThresholdModel.load(model_path)
            q = calibrated_quality(y[roles['selection']], loaded.raw_from_design(designs['selection']), dense.task,
                                     loaded.decision_shifts)
            if any(abs(q[k]-chosen['validation'][k]) > 1e-7 for k in q):
                raise RuntimeError('geometry chosen quality replay mismatch')
            if (model.parameter_count, model.payload_bytes) != (chosen['parameters'], chosen['payload_bytes']):
                raise RuntimeError('geometry chosen actual cost mismatch')
            saved[cid] = dict(artifact=model_path.relative_to(output).as_posix(), sha256=sha256(model_path),
                             coo_bytes=model.payload_bytes, csr_bytes=model.prepare_inference().resident_bytes)
            return saved[cid]
        for family, percent in [('rank_only', 0), ('rank_residual', 5)]:
            for matching in (False, True):
                rows = plan['rows']
                anchor = anchors[family]['candidate']
                if matching:
                    rows = [] if anchor is None else [r for r in rows if r['parameters'] == anchor['parameters']
                                                      and r['payload_bytes'] == anchor['payload_bytes']
                                                      and r['rank'] == anchor['rank'] and r['nnz'] == anchor['nnz']]
                choice = choose_joint(rows, source['validation_reference'], budget, family, percent, ref_cost.parameters)
                record = dict(geometry=geometry, family=family, cap_percent=percent, fraction=.75,
                              exact_anchor_matching=matching, budget=asdict(budget), **choice_record(choice))
                if matching:
                    record['functional_anchor'] = anchors[family]
                    record['matching_scope'] = 'functional-selected allocation; not an independently optimized coefficient allocation'
                if choice.candidate is not None:
                    record.update(save(choice.candidate))
                choices.append(record)
        emit('geometry_case_variant_sealed', dataset=case['dataset'], seed=case['seed'], geometry=geometry,
             candidates=len(plan['rows']), models=len(saved))
    record = dict(dataset=case['dataset'], seed=case['seed'], task=case['task'], choices=choices,
        candidate_grids=grids, reference_cost=source['reference_cost'], validation_reference=source['validation_reference'],
        source_model_sha256=fit['sha256'], test_evaluated=False, seconds=time.perf_counter()-started)
    write_report(directory/'selection.json', record)
    return {k:v for k,v in record.items() if k != 'candidate_grids'}


def leaf_geometry_study(config, report, output):
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_leaf_geometry_core.py',
                    'tests/test_leaf_geometry_study.py', 'tests/test_uniform_threshold_core.py'],
                   cwd=ROOT, check=True, timeout=300)
    manifest = expanded_manifest(config)
    path = Path(config['joint_development_manifest'])
    if sha256(path) != config['joint_development_manifest_sha256']:
        raise RuntimeError('geometry parent manifest changed')
    parent = json.loads(path.read_text())
    records, complete, stop_reason = [], True, None
    for name in config['datasets']:
        x, y = load_expanded(name, manifest)
        for case in (c for c in manifest['cases'] if c['dataset'] == name):
            if time.perf_counter()-started > config.get('geometry_walltime_seconds', 3300)-300:
                complete, stop_reason = False, 'phase_time_budget'
                break
            try:
                check_deadline(config, 3600)
            except TimeoutError:
                complete, stop_reason = False, 'autonomous_deadline_reserve'
                break
            source = next(r for r in parent['records'] if r['dataset'] == name and r['seed'] == case['seed'])
            roles = load_roles(case, len(x))
            records.append(prepare_geometry(case, source, path.parent, config, output, x, y, roles))
            write_report(output/'geometry_development_manifest.json', dict(records=records, test_evaluated=False))
            gc.collect()
        if not complete:
            break
    report.update(status='development_complete' if complete else 'partial_budget', stop_reason=stop_reason, records=records,
                  tree_fits=0, basis_fits=2*len(records), test_evaluated=False,
                  development_manifest_sha256=sha256(output/'geometry_development_manifest.json'),
                  elapsed_seconds=time.perf_counter()-started)
    emit('geometry_development_complete', cases=len(records), tree_fits=0, basis_fits=report['basis_fits'],
         seconds=report['elapsed_seconds'])
