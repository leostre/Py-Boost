"""Independent artifact/cost/decision replay, then the single sealed test pass."""
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants
from experiments.sparse_leaf_core import leaf_design
from experiments.output_calibration_core import calibrated_quality, fit_decision_shifts, labels_from_raw
from experiments.label_budget_diagnostics import compare_labels, per_label, training_profile
from experiments.sparse_leaf_study import verify_sources


def guard(q, full):
    return bool(q['loss'] <= full['loss']*1.05 and q['f1_macro'] >= full['f1_macro']*.95
                and q['f1_weighted'] >= full['f1_weighted']*.95)


def array_cost(path):
    with np.load(path, allow_pickle=False) as f:
        fields = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias',
                  'rows', 'columns', 'values', 'decision_shifts')
        bytes_ = sum(f[k].nbytes for k in fields)
        coefficients = sum(f[k].size for k in ('coefficients', 'basis', 'bias', 'values', 'decision_shifts'))
        if str(f['representation']) == 'reference_class':
            coefficients -= 1
        return int(coefficients), int(bytes_)


def raw(model, features):
    return model.raw_from_design(leaf_design(model.base.leaf_indices(features), len(model.base.coefficients)))


def within(row, budget):
    return row['parameters'] <= budget['parameters'] and row['payload_bytes'] <= budget['payload_bytes']


def independent_static_choice(rows, reference, decision, reference_parameters):
    family, percent = decision['family'], decision['cap_percent']
    filtered = []
    for r in rows:
        if r['nnz']*100 > percent*reference_parameters or not within(r, decision['budget']):
            continue
        if family == 'leaf_only' and r['kind'] != 'full':
            continue
        if family == 'rank_only' and not (r['depth'] == 5 and r['kind'] == 'rank_only'):
            continue
        if family == 'rank_residual' and not (r['depth'] == 5 and r['kind'] in ('rank_only', 'sparse')):
            continue
        filtered.append(r)
    if not filtered:
        return 'no_memory_feasible', None
    eligible = [r for r in filtered if guard(r['validation'], reference)]
    chosen = min(eligible or filtered, key=lambda r: (-r['validation']['f1_macro'], r['validation']['loss'],
        r['payload_bytes'], r['parameters'], r['candidate_id']))
    return 'accepted' if eligible else 'no_quality_feasible', chosen


def verify_model(selection, root, x, y, roles, config, cache, stats):
    path = root/selection['artifact']
    if sha256(path) != selection['sha256']:
        raise RuntimeError('sealed candidate changed: '+str(path))
    key = str(path)
    if key in cache:
        return cache[key]
    model = ThresholdModel.load(path)
    p, b = array_cost(path)
    if (p, b) != (model.parameter_count, model.payload_bytes):
        raise RuntimeError('independent actual-array cost mismatch')
    candidate = selection['candidate']
    if (p, b) != (candidate['parameters'], candidate['payload_bytes']):
        raise RuntimeError('chosen cost differs from actual artifact')
    semantic = dict(trees=model.base.n_trees, leaves=len(model.base.coefficients),
                    internal_nodes=len(model.base.nodes), rank=model.base.coefficients.shape[1], nnz=len(model.values))
    if any(k in candidate and candidate[k] != v for k, v in semantic.items()):
        raise RuntimeError('chosen topology/rank/residual metadata differs from artifact')
    z_cal = raw(model, x[roles['calibration']])
    fitted = fit_decision_shifts(z_cal, y[roles['calibration']], model.base.task, config['decision_grid'])
    expected_shifts = (fitted*candidate['threshold_fraction']).astype(np.float32)
    np.testing.assert_array_equal(model.decision_shifts, expected_shifts)
    np.testing.assert_array_equal(model.decision_shifts, np.array(candidate['decision_shifts'], np.float32))
    selection_raw = raw(model, x[roles['selection']])
    q = calibrated_quality(y[roles['selection']], selection_raw, model.base.task, model.decision_shifts)
    error = max(abs(q[k]-candidate['validation'][k]) for k in q)
    if error > 1e-7:
        raise RuntimeError('independent selection metric replay failed')
    prepared = model.prepare_inference()
    if selection.get('csr_bytes') is not None and prepared.resident_bytes != selection['csr_bytes']:
        raise RuntimeError('prepared CSR bytes differ from sealed metadata')
    probes = x[roles['selection'][:min(13, len(roles['selection']))]]
    np.testing.assert_allclose(prepared.predict(probes), model.predict(probes), atol=2e-12, rtol=0)
    stats['models'] += 1
    stats['maximum_selection_replay_error'] = max(stats['maximum_selection_replay_error'], error)
    cache[key] = dict(model=model, csr_bytes=prepared.resident_bytes, parameters=p, payload_bytes=b)
    return cache[key]


def csv_table(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_budget_audit(config, report, output):
    started = time.perf_counter()
    expected_pretest_seal = None
    if not config.get('audit_only', True) and config.get('require_pretest_audit_parent', False):
        parent_path = Path(config['pretest_audit_report'])
        if sha256(parent_path) != config['pretest_audit_report_sha256']:
            raise RuntimeError('pretest audit parent report changed')
        checked = json.loads(parent_path.read_text())
        if checked['status'] != 'audit_complete' or checked['test_evaluated']:
            raise RuntimeError('all-case audit must answer before any test evaluation')
        expected_pretest_seal = checked['sealed_test_inputs_sha256']
        if sha256(parent_path.parent/'sealed_test_inputs.json') != expected_pretest_seal:
            raise RuntimeError('pretest parent input seal changed')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_joint_budget_audit.py',
                    'tests/test_joint_budget_audit_pipeline.py', 'tests/test_label_budget_diagnostics.py',
                    'tests/test_continuation_contract.py'],
                   cwd=ROOT, check=True, timeout=300)
    data_manifest = expanded_manifest(config)
    static_path = Path(config['joint_development_manifest'])
    if sha256(static_path) != config['joint_development_manifest_sha256']:
        raise RuntimeError('static manifest changed')
    static = json.loads(static_path.read_text())
    if static['test_evaluated']:
        raise RuntimeError('unexpected pre-evaluated static test')
    closed_inputs = list(config.get('closed_development_manifests', []))
    if config.get('closed_development_manifest'):
        closed_inputs.append(dict(path=config['closed_development_manifest'],
                                  sha256=config['closed_development_manifest_sha256']))
    if config.get('require_closed_loop_sources', False) and not closed_inputs:
        raise RuntimeError('mandatory closed-loop sources missing')
    closed = dict(records=[])
    closed_paths = []
    for entry in closed_inputs:
        source = Path(entry['path'])
        if sha256(source) != entry['sha256']:
            raise RuntimeError('closed manifest changed')
        part = json.loads(source.read_text())
        if part['test_evaluated']:
            raise RuntimeError('unexpected pre-evaluated closed test')
        if part['parent_manifest'] != str(static_path) or part['parent_manifest_sha256'] != sha256(static_path):
            raise RuntimeError('closed trajectory used a different static reference')
        closed_paths.append(source)
        closed['records'].extend({**r, '_artifact_root': str(source.parent)} for r in part['records'])
    keys = [(r['dataset'], r['seed'], r['method']) for r in closed['records']]
    if len(set(keys)) != len(keys):
        raise RuntimeError('duplicate scientific closed-loop case/method across manifests')
    geometry_path = Path(config['geometry_development_manifest']) if config.get('geometry_development_manifest') else None
    geometry = dict(records=[])
    if config.get('require_geometry_source', False) and geometry_path is None:
        raise RuntimeError('mandatory frozen geometry source missing')
    if geometry_path:
        if sha256(geometry_path) != config['geometry_development_manifest_sha256']:
            raise RuntimeError('geometry manifest changed')
        geometry = json.loads(geometry_path.read_text())
        if geometry['test_evaluated']:
            raise RuntimeError('unexpected pre-evaluated geometry test')
    prefix_path = Path(config['prefix_development_manifest']) if config.get('prefix_development_manifest') else None
    prefixes = dict(records=[])
    if config.get('require_prefix_source', False) and prefix_path is None:
        raise RuntimeError('mandatory early-stopping control missing')
    if prefix_path:
        if sha256(prefix_path) != config['prefix_development_manifest_sha256']:
            raise RuntimeError('prefix manifest changed')
        prefixes = json.loads(prefix_path.read_text())
        if prefixes['test_evaluated']:
            raise RuntimeError('unexpected pre-evaluated prefix test')
    # Compare original teachers to the historical sealed hashes, not just a
    # checksum sampled at the beginning of today's read.
    verify_sources(Path(config['source_dir']), json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text()))
    for path, expected in data_manifest['old_file_locks'].items():
        if sha256(path) != expected:
            raise RuntimeError('data preparation input lock changed')
    sealed_paths = {str(static_path): sha256(static_path)}
    for source in closed_paths:
        sealed_paths[str(source)] = sha256(source)
    if geometry_path:
        sealed_paths[str(geometry_path)] = sha256(geometry_path)
    if prefix_path:
        sealed_paths[str(prefix_path)] = sha256(prefix_path)
        for record in prefixes['records']:
            for selection in record['choices']:
                if selection['candidate'] is not None:
                    sealed_paths[str(prefix_path.parent/selection['artifact'])] = selection['sha256']
    for record in geometry['records']:
        for selection in record['choices']:
            if selection['candidate'] is not None:
                sealed_paths[str(geometry_path.parent/selection['artifact'])] = selection['sha256']
    for record in static['records']:
        for selected in [record['full'], *record['decisions']]:
            if selected.get('candidate') is not None:
                path = static_path.parent/selected['artifact']
                sealed_paths[str(path)] = selected['sha256']
    for record in closed['records']:
        record_root = Path(record['_artifact_root'])
        if record['choice'].get('candidate') is not None:
            selected = record['choice']
            sealed_paths[str(record_root/selected['artifact'])] = selected['sha256']
        for cycle in record['cycles']:
            if 'fit' in cycle:
                cycle_dir = record_root/f"{record['dataset']}_{record['seed']}"/record['method']/f"block{cycle['block']}"
                sealed_paths[str(cycle_dir/'initial.npz')] = cycle['initial_sha256']
                sealed_paths[str(cycle_dir/'increment.npz')] = cycle['fit']['sha256']
            if cycle['state'] == 'accepted':
                sealed_paths[str(record_root/cycle['artifact'])] = cycle['sha256']
    write_report(output/'sealed_test_inputs.json', sealed_paths)
    seal = sha256(output/'sealed_test_inputs.json')
    if expected_pretest_seal is not None and seal != expected_pretest_seal:
        raise RuntimeError('test inputs differ from the successful all-case audit')
    emit('joint_test_inputs_sealed', sha256=seal, paths=len(sealed_paths), test_started=False)
    rows, audits, label_rows, profiles = [], [], [], []
    stats = dict(models=0, candidate_rows=0, decisions=0, maximum_selection_replay_error=0.)
    # Each case is verified before its test rows are read. No decisions are
    # changed after this common seal; no subsequent training occurs in this run.
    for source_record in static['records']:
        check_deadline(config, 300)
        name, seed = source_record['dataset'], source_record['seed']
        case = next(c for c in data_manifest['cases'] if c['dataset'] == name and c['seed'] == seed)
        x, y = load_expanded(name, data_manifest)
        roles = load_roles(case, len(x))
        record_path = static_path.parent/f'{name}_{seed}'/'selection.json'
        record = json.loads(record_path.read_text())
        if {k:v for k,v in record.items() if k != 'candidate_rows'} != source_record:
            raise RuntimeError('static case manifest differs')
        reference_cost = record['reference_cost']
        for candidate in record['candidate_rows']:
            if candidate['kind'] != 'full':
                m, c, r, s = candidate['leaves'], case['outputs'], candidate['rank'], candidate['nnz']
                p = (m+c)*r+s+2*c
                b = 16*candidate['internal_nodes']+16*(candidate['trees']+1)+4*(m+c)*r+12*s+8*c
                if (p, b) != (candidate['parameters'], candidate['payload_bytes']):
                    raise RuntimeError('independent candidate-map cost failed')
            stats['candidate_rows'] += 1
        cache = {}
        full = verify_model(record['full'], static_path.parent, x, y, roles, config, cache, stats)
        c = case['outputs']
        if (full['parameters']-c, full['payload_bytes']-4*c) != (
                reference_cost['parameters'], reference_cost['payload_bytes']):
            raise RuntimeError('global uncalibrated exact-reference denominator changed')
        if record['validation_reference'] != record['full']['candidate']['validation']:
            raise RuntimeError('global full quality reference changed')
        profiles.append(dict(dataset=name, seed=seed,
            **training_profile(x[roles['train']], y[roles['train']], raw(full['model'], x[roles['train']]), case['task']),
            full_leaves=len(full['model'].base.coefficients), full_internal_nodes=len(full['model'].base.nodes),
            full_trees=full['model'].base.n_trees))
        decisions = []
        for d in record['decisions']:
            state, chosen = independent_static_choice(record['candidate_rows'], record['validation_reference'], d,
                                                       reference_cost['parameters'])
            if state != d['state'] or chosen != d['candidate']:
                raise RuntimeError('independent global static selection differs')
            stats['decisions'] += 1
            if d['candidate'] is not None:
                verified = verify_model(d, static_path.parent, x, y, roles, config, cache, stats)
                decisions.append((f"static_{d['family']}_cap{d['cap_percent']}", d['fraction'], d, verified))
            else:
                decisions.append((f"static_{d['family']}_cap{d['cap_percent']}", d['fraction'], d, None))
        for trajectory in (r for r in closed['records'] if r['dataset'] == name and r['seed'] == seed):
            trajectory_root = Path(trajectory['_artifact_root'])
            choice = trajectory['choice']
            worker_path = trajectory_root/f'{name}_{seed}'/trajectory['method']/'worker_data.npz'
            worker_sha = sha256(worker_path)
            with np.load(worker_path, allow_pickle=False) as worker:
                for key, source, role in [('train_x', x, 'train'), ('train_y', y, 'train'),
                                          ('validation_x', x, 'calibration'), ('validation_y', y, 'calibration')]:
                    np.testing.assert_array_equal(worker[key], source[roles[role]])
            pool = choice['candidates']
            feasible = [r for r in pool if within(r, trajectory['budget'])]
            eligible = [r for r in feasible if guard(r['validation'], record['validation_reference'])]
            if pool:
                expected = min(eligible or feasible, key=lambda r: (-r['validation']['f1_macro'], r['validation']['loss'],
                    r['payload_bytes'], r['parameters'], r['candidate_id']))
                if choice['candidate'] != expected or choice['state'] != ('accepted' if eligible else 'no_quality_feasible'):
                    raise RuntimeError('independent closed final selection differs')
            previous = None
            full_fit = next(f for f in record['fits'] if f['depth'] == 5)
            from experiments.shared_leaf_basis import LeafEnsemble
            prior = LeafEnsemble.load(static_path.parent/full_fit['artifact']).bias
            for cycle in trajectory['cycles']:
                if 'fit' not in cycle:
                    continue
                if cycle['fit']['data_sha256'] != worker_sha:
                    raise RuntimeError('native continuation data checksum changed')
                cycle_dir = trajectory_root/f'{name}_{seed}'/trajectory['method']/f"block{cycle['block']}"
                initial_path = cycle_dir/'initial.npz'
                if sha256(initial_path) != cycle['fit']['initial_sha256'] or sha256(initial_path) != cycle['initial_sha256']:
                    raise RuntimeError('continued initial-state checksum changed')
                with np.load(initial_path, allow_pickle=False) as initial:
                    expected_train = (np.broadcast_to(prior, (len(roles['train']), c)) if previous is None else
                                      raw(previous, x[roles['train']])).astype(np.float32)
                    expected_cal = (np.broadcast_to(prior, (len(roles['calibration']), c)) if previous is None else
                                    raw(previous, x[roles['calibration']]))
                    np.testing.assert_array_equal(initial['train'], expected_train)
                    np.testing.assert_array_equal(initial['validation'], expected_cal)
                from experiments.closed_budget_study import combine_actual
                increment = ThresholdModel.load(cycle_dir/'increment.npz')
                proposal = combine_actual(previous, increment, prior)
                full_variants = threshold_variants(raw(proposal, x[roles['calibration']]), y[roles['calibration']],
                    raw(proposal, x[roles['control']]), y[roles['control']], case['task'],
                    config['decision_grid'], config['decision_shrinkage'])
                local_full = min(full_variants, key=lambda r: (-r['validation']['f1_macro'], r['validation']['loss'],
                                                              r['threshold_fraction']))['validation']
                if any(abs(local_full[k]-cycle['projection']['proposal_control'][k]) > 1e-7 for k in local_full):
                    raise RuntimeError('local full-proposal quality replay differs')
                if cycle['state'] == 'accepted':
                    accepted_path = trajectory_root/cycle['artifact']
                    if sha256(accepted_path) != cycle['sha256']:
                        raise RuntimeError('closed accepted checkpoint changed')
                    previous = ThresholdModel.load(accepted_path)
                    p, b = array_cost(accepted_path)
                    if (p, b) != (cycle['parameters'], cycle['payload_bytes']) or not within(cycle, trajectory['budget']):
                        raise RuntimeError('accepted closed checkpoint broke resource cap')
                    local = calibrated_quality(y[roles['control']], raw(previous, x[roles['control']]),
                                               case['task'], previous.decision_shifts)
                    chosen_local = cycle['projection']['chosen']['validation']
                    if any(abs(local[k]-chosen_local[k]) > 1e-7 for k in local):
                        raise RuntimeError('accepted local-control quality replay differs')
                    enforced = trajectory['method'] != 'periodic_no_local_guard'
                    if cycle['projection'].get('local_quality_enforced', enforced) != enforced:
                        raise RuntimeError('local guard policy metadata differs')
                    if enforced and not guard(local, local_full):
                        raise RuntimeError('guarded trajectory accepted a locally rejected state')
                del proposal, increment
            verified = verify_model(choice, trajectory_root, x, y, roles, config, cache, stats) if choice['candidate'] else None
            decisions.append((trajectory['method'], .75, {**choice, 'budget': trajectory['budget']}, verified))
        for g_record in (r for r in geometry['records'] if r['dataset'] == name and r['seed'] == seed):
            g_path = geometry_path.parent/f'{name}_{seed}'/'selection.json'
            g_full = json.loads(g_path.read_text())
            if {k:v for k,v in g_full.items() if k != 'candidate_grids'} != g_record:
                raise RuntimeError('geometry case manifest differs')
            if g_record['reference_cost'] != reference_cost or g_record['validation_reference'] != record['validation_reference']:
                raise RuntimeError('geometry global reference differs')
            for g_rows in g_full['candidate_grids'].values():
                for r in g_rows:
                    m, rank, count = r['leaves'], r['rank'], r['nnz']
                    p = (m+c)*rank+count+2*c
                    b = 16*r['internal_nodes']+16*(r['trees']+1)+4*(m+c)*rank+12*count+8*c
                    if (p, b) != (r['parameters'], r['payload_bytes']):
                        raise RuntimeError('geometry grid actual cost oracle failed')
                    stats['candidate_rows'] += 1
            for selection in g_record['choices']:
                pool = g_full['candidate_grids'][selection['geometry']]
                if selection['exact_anchor_matching']:
                    expected_anchor = next(d for d in record['decisions'] if d['fraction'] == .75 and
                        d['family'] == selection['family'] and d['cap_percent'] == selection['cap_percent'])
                    if selection['functional_anchor'] != expected_anchor:
                        raise RuntimeError('geometry exact functional anchor changed')
                    anchor = expected_anchor['candidate']
                    pool = [] if anchor is None else [r for r in pool if all(
                        r[k] == anchor[k] for k in ('parameters', 'payload_bytes', 'rank', 'nnz'))]
                state, chosen = independent_static_choice(pool, record['validation_reference'], selection,
                                                           reference_cost['parameters'])
                if state != selection['state'] or chosen != selection['candidate']:
                    raise RuntimeError('independent geometry selection differs')
                stats['decisions'] += 1
                verified = verify_model(selection, geometry_path.parent, x, y, roles, config, cache, stats) if chosen else None
                suffix = 'exact' if selection['exact_anchor_matching'] else 'cap'
                decisions.append((f"geometry_{selection['geometry']}_{selection['family']}_{suffix}", .75,
                                  selection, verified))
        for p_record in (r for r in prefixes['records'] if r['dataset'] == name and r['seed'] == seed):
            p_full = json.loads((prefix_path.parent/f'{name}_{seed}'/'selection.json').read_text())
            if {k:v for k,v in p_full.items() if k != 'candidate_rows'} != p_record:
                raise RuntimeError('prefix case manifest differs')
            if p_record['reference_cost'] != reference_cost or p_record['validation_reference'] != record['validation_reference']:
                raise RuntimeError('prefix global reference differs')
            width = c-1 if case['task'] == 'multiclass' else c
            for r in p_full['candidate_rows']:
                p = r['leaves']*width+2*c-(case['task'] == 'multiclass')
                b = 16*r['internal_nodes']+16*(r['trees']+1)+4*r['leaves']*width+8*c
                if r['rank'] != width or (p, b) != (r['parameters'], r['payload_bytes']):
                    raise RuntimeError('prefix exact full model array-cost oracle failed')
                stats['candidate_rows'] += 1
            for selection in p_record['choices']:
                pool = [r for r in p_full['candidate_rows'] if selection['selection_scope'] == 'all_depths' or r['depth'] == 5]
                state, chosen = independent_static_choice(pool, record['validation_reference'], selection, reference_cost['parameters'])
                if state != selection['state'] or chosen != selection['candidate']:
                    raise RuntimeError('independent prefix selection differs')
                stats['decisions'] += 1
                verified = verify_model(selection, prefix_path.parent, x, y, roles, config, cache, stats) if chosen else None
                decisions.append(('native_prefix_'+selection['selection_scope'], selection['fraction'], selection, verified))
        if config.get('audit_only', True):
            audits.append(dict(dataset=name, seed=seed, models=len(cache), test_rows=0))
            emit('joint_artifact_audit_case', dataset=name, seed=seed, models=len(cache), test_started=False)
            del cache, x, y
            gc.collect()
            continue
        # The audit-only parent must first verify ALL cases. No fitting,
        # choice, or budget changes after that parent's successful seal.
        full_test_raw = raw(full['model'], x[roles['test']])
        reference_test = calibrated_quality(y[roles['test']], full_test_raw, case['task'], full['model'].decision_shifts)
        full_labels = labels_from_raw(full_test_raw, case['task'], full['model'].decision_shifts)
        full_counts = per_label(y[roles['test']], full_labels, case['task'], c)
        if max(abs(full_counts['macro']-reference_test['f1_macro']),
               abs(full_counts['weighted']-reference_test['f1_weighted'])) > 1e-12:
            raise RuntimeError('independent full F1 count oracle differs')
        test_cache = {}
        for method, fraction, decision, verified in decisions:
            candidate = decision.get('candidate')
            row = dict(dataset=name, seed=seed, method=method, fraction=fraction, state=decision['state'],
                       test_quality_pass=False, joint_success=False, parameters=None, bytes=None, csr_bytes=None,
                       memory_fraction=None, coefficient_fraction=None, trees=None, leaves=None, rank=None, nnz=None,
                       loss=None, macro_f1=None, weighted_f1=None, loss_ratio=None, macro_retained=None,
                       weighted_retained=None, reference_loss=reference_test['loss'],
                       reference_macro_f1=reference_test['f1_macro'], reference_weighted_f1=reference_test['f1_weighted'],
                       rare_labels=None, rare_full_macro_fixed_domain=None, rare_macro_fixed_domain=None,
                       rare_macro_retained=None, label_f1_losses=None, label_f1_gains=None,
                       full_macro_domain=None, macro_domain=None)
            if verified is not None:
                artifact = decision['artifact']
                if artifact not in test_cache:
                    model = verified['model']
                    test_raw = raw(model, x[roles['test']])
                    q = calibrated_quality(y[roles['test']], test_raw, case['task'], model.decision_shifts)
                    labels = labels_from_raw(test_raw, case['task'], model.decision_shifts)
                    details, diagnostic = compare_labels(y[roles['test']], full_labels, labels,
                                                         y[roles['train']], case['task'], c)
                    if max(abs(diagnostic['independent_macro']-q['f1_macro']),
                           abs(diagnostic['independent_weighted']-q['f1_weighted'])) > 1e-12:
                        raise RuntimeError('independent candidate F1 count oracle differs')
                    diagnostic.pop('independent_macro')
                    diagnostic.pop('independent_weighted')
                    test_cache[artifact] = (q, details, diagnostic)
                q, details, diagnostic = test_cache[artifact]
                label_rows.extend(dict(dataset=name, seed=seed, method=method, fraction=fraction, **d) for d in details)
                row.update(diagnostic)
                cost_pass = within(candidate, decision['budget'])
                if not cost_pass:
                    raise RuntimeError('test artifact violates declared joint budget')
                row.update(test_quality_pass=guard(q, reference_test),
                    joint_success=decision['state'] == 'accepted' and guard(q, reference_test),
                    parameters=verified['parameters'], bytes=verified['payload_bytes'], csr_bytes=verified['csr_bytes'],
                    memory_fraction=verified['payload_bytes']/reference_cost['payload_bytes'],
                    coefficient_fraction=verified['parameters']/reference_cost['parameters'],
                    trees=verified['model'].base.n_trees, leaves=len(verified['model'].base.coefficients),
                    rank=verified['model'].base.coefficients.shape[1], nnz=len(verified['model'].values),
                    loss=q['loss'], macro_f1=q['f1_macro'], weighted_f1=q['f1_weighted'],
                    loss_ratio=q['loss']/reference_test['loss'],
                    macro_retained=q['f1_macro']/reference_test['f1_macro'] if reference_test['f1_macro'] else None,
                    weighted_retained=q['f1_weighted']/reference_test['f1_weighted'] if reference_test['f1_weighted'] else None)
            rows.append(row)
        audits.append(dict(dataset=name, seed=seed, models=len(cache), test_rows=len(roles['test'])))
        emit('joint_audit_test_case', dataset=name, seed=seed, models=len(cache), decisions=len(decisions))
        csv_table(output/'comparisons.csv', rows)
        csv_table(output/'per_label.csv', label_rows)
        del cache, x, y
        gc.collect()
    for path, expected in sealed_paths.items():
        if sha256(path) != expected:
            raise RuntimeError('sealed test input changed')
    summaries = []
    for name, method, fraction in sorted({(r['dataset'], r['method'], r['fraction']) for r in rows}):
        block = [r for r in rows if (r['dataset'], r['method'], r['fraction']) == (name, method, fraction)]
        values = [r for r in block if r['bytes'] is not None]
        summaries.append(dict(dataset=name, method=method, fraction=fraction, splits=len(block),
            development_acceptances=sum(r['state']=='accepted' for r in block),
            test_passes=sum(r['test_quality_pass'] for r in block), joint_successes=sum(r['joint_success'] for r in block),
            missing=sum(r['bytes'] is None for r in block),
            median_memory_fraction=float(np.median([r['memory_fraction'] for r in values])) if values else None,
            median_macro_retained=float(np.median([r['macro_retained'] for r in values if r['macro_retained'] is not None])) if values else None))
    csv_table(output/'summaries.csv', summaries)
    csv_table(output/'training_profiles.csv', profiles)
    report.update(status='audit_complete' if config.get('audit_only', True) else 'complete',
                  records=rows, summaries=summaries, training_profiles=profiles, audit_cases=audits, audit=stats,
                  sealed_test_inputs_sha256=seal, input_paths_verified=len(sealed_paths), tree_fits=0,
                  test_evaluated=not config.get('audit_only', True), elapsed_seconds=time.perf_counter()-started)
    emit('joint_audit_complete', cases=len(audits), **stats, seconds=report['elapsed_seconds'])
