"""Synthetic end-to-end audit: all artifacts checked before sealed scoring."""
from dataclasses import asdict
import numpy as np
import pytest

from experiments.a100_runner import sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, functional_basis
from experiments.sparse_leaf_core import Cost, budget_for
from experiments.uniform_threshold_study import uniform_plan, materialize_uniform
from experiments.joint_budget_core import choose_joint, choice_record


@pytest.mark.parametrize('audit_only', [True, False])
@pytest.mark.parametrize('include_prefix', [True, False])
def test_independent_audit_pipeline_and_test_access_boundary(tmp_path, monkeypatch, audit_only, include_prefix):
    import experiments.joint_budget_audit as audit
    rng = np.random.default_rng(8004)
    c, repeats = 12, 15
    dense = LeafEnsemble(np.tile(np.array([[1, 0, -1, -2], [1, 0, -1, -1]], np.float32), (repeats, 1)),
        np.arange(2*repeats+1, dtype=np.int64),
        np.r_[0, np.cumsum([2, 1]*repeats)].astype(np.int64),
        rng.normal(scale=.1, size=(3*repeats, c)).astype(np.float32), np.empty((0, 0), np.float32),
        rng.normal(scale=.1, size=c).astype(np.float32), 2, 'multilabel')
    x = np.column_stack((np.tile([-1., 1.], 50), np.arange(100))).astype(np.float32)
    y = rng.integers(0, 2, (100, c))
    roles = dict(train=np.arange(40), calibration=np.arange(40, 60), control=np.arange(60, 70),
                 selection=np.arange(70, 80), test=np.arange(80, 100))
    leaves = {k:dense.leaf_indices(x[roles[k]]) for k in ('train', 'calibration', 'selection')}
    mean, basis, _ = functional_basis(dense.contributions(leaves['train']), 'multilabel')
    cfg = dict(budget_kinds=['joint'], budget_fractions=[.75], base_rank_fractions=[.5],
               residual_capacity_fractions=[1.], tiny_selectors=['balanced_mse'], margin_scale=.5,
               decision_grid=[0, -1, 1], decision_shrinkage=[.5, 1.])
    ref_cost = Cost(dense.parameter_count, dense.payload_bytes)
    plan = uniform_plan(dense, dense, mean, basis, leaves['train'], y[roles['train']],
        leaves['calibration'], y[roles['calibration']], leaves['selection'], y[roles['selection']],
        ref_cost, cfg, build_selections=False)
    all_rows = [*plan['full_candidates'], *plan['rows']]
    for row in all_rows:
        row.update(depth=5, leaves=len(dense.coefficients), internal_nodes=len(dense.nodes), trees=dense.n_trees)
    case_dir = tmp_path/'toy_1'
    case_dir.mkdir()
    def save(row):
        model = materialize_uniform(row, plan, dense, dense, mean, basis)
        path = case_dir/(row['candidate_id']+'.npz')
        model.save(path)
        return dict(candidate=row, artifact=path.relative_to(tmp_path).as_posix(), sha256=sha256(path),
                    coo_bytes=model.payload_bytes, csr_bytes=model.prepare_inference().resident_bytes)
    full = save(plan['full_choice'])
    decisions = []
    budget = budget_for(ref_cost, 'joint', .75)
    for family, cap in [('leaf_only', 0), ('rank_only', 0), ('joint_static', 5)]:
        choice = choose_joint(all_rows, plan['validation_reference'], budget, family, cap, ref_cost.parameters)
        d = dict(fraction=.75, family=family, cap_percent=cap, budget=asdict(budget), **choice_record(choice))
        if choice.candidate is not None:
            d.update(save(choice.candidate))
        decisions.append(d)
    record = dict(dataset='toy', seed=1, task='multilabel', full=full, decisions=decisions,
                  candidate_rows=all_rows, reference_cost=asdict(ref_cost), validation_reference=plan['validation_reference'])
    write_report(case_dir/'selection.json', record)
    manifest_path = tmp_path/'development_manifest.json'
    write_report(manifest_path, dict(records=[{k:v for k,v in record.items() if k != 'candidate_rows'}], test_evaluated=False))
    case = dict(dataset='toy', seed=1, outputs=c, task='multilabel')
    monkeypatch.setattr(audit, 'expanded_manifest', lambda config: dict(cases=[case], old_file_locks={}))
    monkeypatch.setattr(audit, 'load_expanded', lambda *args: (x, y))
    monkeypatch.setattr(audit, 'load_roles', lambda *args: roles)
    monkeypatch.setattr(audit, 'verify_sources', lambda *args: None)
    monkeypatch.setattr(audit, 'check_deadline', lambda *args: None)
    monkeypatch.setattr(audit.subprocess, 'run', lambda *args, **kwargs: None)
    original_raw = audit.raw
    def checked_raw(model, features):
        if audit_only:
            assert np.max(features[:, 1]) < 80, 'audit-only must never score a test row'
        return original_raw(model, features)
    monkeypatch.setattr(audit, 'raw', checked_raw)
    output = tmp_path/'audit'
    output.mkdir()
    config = {**cfg, 'joint_development_manifest': str(manifest_path),
              'joint_development_manifest_sha256': sha256(manifest_path), 'source_dir': str(tmp_path),
              'audit_only': audit_only}
    if include_prefix:
        from experiments.prefix_budget_study import prefix_grid
        p_rows, p_models = prefix_grid({5:dense}, x[roles['calibration']], y[roles['calibration']],
            x[roles['selection']], y[roles['selection']], {**cfg, 'prefix_checkpoints':[10, 20, 30]})
        p_dir = tmp_path/'prefix'/'toy_1'
        p_dir.mkdir(parents=True)
        p_choices = []
        for scope in ('depth5', 'all_depths'):
            choice = choose_joint(p_rows, plan['validation_reference'], budget, 'leaf_only', 0, ref_cost.parameters)
            selected = dict(fraction=.75, family='leaf_only', cap_percent=0, selection_scope=scope,
                            budget=asdict(budget), **choice_record(choice))
            if choice.candidate is not None:
                from dataclasses import replace
                r = choice.candidate
                model = replace(p_models[(5, r['trees'])], decision_shifts=np.array(r['decision_shifts'], np.float32))
                p = p_dir/(r['candidate_id']+'.npz')
                model.save(p)
                selected.update(artifact=p.relative_to(p_dir.parent).as_posix(), sha256=sha256(p),
                                csr_bytes=model.prepare_inference().resident_bytes)
            p_choices.append(selected)
        p_record = dict(dataset='toy', seed=1, choices=p_choices, candidate_rows=p_rows,
                        reference_cost=asdict(ref_cost), validation_reference=plan['validation_reference'], test_evaluated=False)
        write_report(p_dir/'selection.json', p_record)
        p_manifest = p_dir.parent/'prefix_development_manifest.json'
        write_report(p_manifest, dict(records=[{k:v for k,v in p_record.items() if k != 'candidate_rows'}], test_evaluated=False))
        config.update(prefix_development_manifest=str(p_manifest), prefix_development_manifest_sha256=sha256(p_manifest))
    report = {}
    audit.joint_budget_audit(config, report, output)
    assert report['status'] == ('audit_complete' if audit_only else 'complete')
    assert report['test_evaluated'] == (not audit_only)
    assert report['audit']['candidate_rows'] == len(all_rows)+(len(p_rows) if include_prefix else 0)
    assert report['audit']['models'] >= 2
    if not audit_only:
        assert len(report['records']) == (5 if include_prefix else 3) and (output/'per_label.csv').is_file()
