"""Approved bounded follow-up: exact causal quotas, null errors, split controls."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.rmt_followup_core import (aggregate_duplicate_sums, duplicate_split_control,
    empirical_checks, gains_from_sums, quota_actions, rare_split_control, split_sums)
from experiments.revision_forecast_replay import file_sha256, verified_saved_records


def features_for_models(rows):
    cheap = ('progress', 'log_energy', 'entropy', 'mass_25', 'mass_50', 'mass_75')
    strong = cheap + ('last_risk_25', 'last_risk_50', 'last_risk_75', 'risk_age')
    centered = tuple(sorted(k for k in rows[0] if k.startswith('centered_')))
    second = tuple(sorted(k for k in rows[0] if k.startswith('second_moment_')))
    if len(centered) != 7 or len(second) != 7:
        raise ValueError('frozen parent RMT feature schema differs')
    return {'iteration': ('progress',), 'current': cheap, 'last_risk': strong,
            'history': strong + tuple(f'{prefix}_{key}' for prefix in ('ema', 'change') for key in cheap[1:]),
            'spectrum': strong + ('spectral_rank', 'spectral_top4_mass', 'fixed_rank4_drift', 'rank90_fraction'),
            'rmt_second_moment': strong + second, 'rmt_centered': strong + centered,
            'rmt_combined': strong + second + centered}


def evaluate_actions(rows, actions, forced):
    risks = [0. if b == r['n_outputs'] else r[f'risk_{b}'] for r, b in zip(rows, actions)]
    return {'unsafe_count': sum(r > .1 for r in risks), 'unsafe_rate': float(np.mean(np.asarray(risks) > .1)),
            'mean_selected_risk': float(np.mean(risks)),
            'mean_budget_fraction': float(np.mean([b / r['n_outputs'] for b, r in zip(actions, rows)])),
            'forced_count': int(np.count_nonzero(forced)),
            'forced_unsafe_count': sum(r > .1 for r, f in zip(risks, forced) if f),
            'budgets': [int(b) for b in actions], 'forced': [bool(f) for f in forced]}


def forecasts_at_quotas(rows, protocol):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import average_precision_score, roc_auc_score
    sets = features_for_models(rows)
    result = []
    for horizon in ('next_tree', 'next_control'):
        selected = [r for r in rows if r['horizon'] == horizon]
        groups = sorted({(r['dataset'], r['seed']) for r in selected})
        for mode in ('trajectory', 'dataset'):
            folds = groups if mode == 'trajectory' else sorted({g[0] for g in groups})
            for held in folds:
                def is_held(r):
                    return (r['dataset'], r['seed']) == held if mode == 'trajectory' else r['dataset'] == held
                train = [r for r in selected if not is_held(r) and (mode != 'trajectory' or r['dataset'] == held[0])]
                test = sorted((r for r in selected if is_held(r)), key=lambda r: (r['dataset'], r['seed'], r['iteration']))
                train_groups, test_groups = sorted({(r['dataset'], r['seed']) for r in train}), sorted({(r['dataset'], r['seed']) for r in test})
                if set(train_groups) & set(test_groups):
                    raise ValueError('forecast train/test trajectory leakage')
                risks_train = np.array([[r[f'risk_{b}'] for b in (25, 50, 75)] for r in train])
                unsafe_train = (risks_train > .1).astype(float)
                targets = np.column_stack((risks_train, unsafe_train))
                risks_test = np.array([[r[f'risk_{b}'] for b in (25, 50, 75)] for r in test])
                unsafe_test = (risks_test > .1).astype(float)
                prior = unsafe_train.mean(axis=0)
                for name, keys in sets.items():
                    before = time.perf_counter()
                    model = RandomForestRegressor(n_estimators=protocol['forest_trees'],
                              min_samples_leaf=protocol['forest_min_leaf'], random_state=protocol['forest_seed'], n_jobs=1)
                    model.fit([[r[k] for k in keys] for r in train], targets)
                    predictions = model.predict([[r[k] for k in keys] for r in test])
                    scores = []
                    for i, budget in enumerate((25, 50, 75)):
                        actual, probability = unsafe_test[:, i], predictions[:, i + 3]
                        scores.append({'budget': budget, 'brier': float(np.mean((probability - actual) ** 2)),
                                       'mae_risk': float(np.mean(abs(predictions[:, i] - risks_test[:, i]))),
                                       'roc_auc': float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
                                       'average_precision': float(average_precision_score(actual, probability)) if actual.sum() else None,
                                       'unsafe_prevalence': float(actual.mean()),
                                       'prior_brier': float(np.mean((prior[i] - actual) ** 2))})
                    quotas = []
                    for dataset, seed in test_groups:
                        indices = [i for i, r in enumerate(test) if (r['dataset'], r['seed']) == (dataset, seed)]
                        block = [test[i] for i in indices]
                        expected = 84 if horizon == 'next_tree' else 79
                        if len(block) != expected or [r['iteration'] for r in block] != list(range(16, 16 + expected)):
                            raise ValueError('planned fixed-length causal row grid differs')
                        for scenario in protocol['quota_scenarios']:
                            low_count = int(np.floor(len(block) * scenario['share']))
                            decision = quota_actions(predictions[indices, 3:], prior, scenario['budget'], block[0]['n_outputs'], low_count)
                            if np.count_nonzero(decision.budgets == scenario['budget']) != low_count:
                                raise ValueError('exact budget quota violated')
                            rng = np.random.default_rng(protocol['seed'] + seed)
                            random_budgets = np.full(len(block), block[0]['n_outputs'])
                            random_budgets[rng.permutation(len(block))[:low_count]] = scenario['budget']
                            true_risk = np.array([r[f'risk_{scenario["budget"]}'] for r in block])
                            oracle = np.full(len(block), block[0]['n_outputs'])
                            oracle[np.argsort(true_risk, kind='stable')[:low_count]] = scenario['budget']
                            quotas.append({'dataset': dataset, 'seed': seed, **scenario, 'low_count': low_count,
                                           'iterations': [r['iteration'] for r in block],
                                           **evaluate_actions(block, decision.budgets, decision.forced),
                                           'random_reference': evaluate_actions(block, random_budgets, np.zeros(len(block), bool)),
                                           'future_oracle_bound': evaluate_actions(block, oracle, np.zeros(len(block), bool))})
                    result.append({'horizon': horizon, 'holdout': mode, 'held_out': held, 'model': name,
                                   'feature_keys': keys, 'train_groups': train_groups, 'test_groups': test_groups,
                                   'train_rows': len(train), 'test_keys': [[r['dataset'], r['seed'], r['iteration']] for r in test],
                                   'train_unsafe_prior': prior.tolist(), 'predictions': predictions.tolist(),
                                   'fixed_budget_scores': scores, 'quotas': quotas,
                                   'fit_predict_seconds': time.perf_counter() - before})
                print(json.dumps({'QUOTA_FORECAST_FOLD_DONE': {'horizon': horizon, 'holdout': mode, 'held_out': held}}), flush=True)
    return result


def quota_summary(evaluation, protocol):
    result = []
    for scenario in protocol['quota_scenarios']:
        for horizon in ('next_tree', 'next_control'):
            for model in sorted({e['model'] for e in evaluation}):
                items = [q for e in evaluation if e['horizon'] == horizon and e['model'] == model and e['holdout'] == 'trajectory'
                         for q in e['quotas'] if q['budget'] == scenario['budget'] and q['share'] == scenario['share']]
                result.append({'horizon': horizon, 'model': model, **scenario, 'trajectories': len(items),
                               **{f'mean_{k}': float(np.mean([q[k] for q in items])) for k in
                                  ('unsafe_rate', 'mean_budget_fraction', 'mean_selected_risk', 'forced_count')}})
    return result


def followup_screen(evaluation, protocol):
    checks = []
    for dataset in ('helena', 'mediamill'):
        selected = [e for e in evaluation if e['horizon'] == 'next_control' and e['holdout'] == 'trajectory' and e['held_out'][0] == dataset]
        by_name = {name: sorted([e for e in selected if e['model'] == name], key=lambda e: e['held_out'][1]) for name in {e['model'] for e in selected}}
        for name in ('rmt_centered', 'rmt_combined'):
            candidate, base = by_name[name], by_name['last_risk']
            brier_changes = [float(np.mean([s['brier'] for s in e['fixed_budget_scores']]) -
                                   np.mean([s['brier'] for s in b['fixed_budget_scores']])) for e, b in zip(candidate, base)]
            scenarios = []
            for scenario in [s for s in protocol['quota_scenarios'] if s['primary']]:
                def q(e):
                    return next(q for q in e['quotas'] if q['budget'] == scenario['budget'] and q['share'] == scenario['share'])
                candidate_q, baseline_q = [q(e) for e in candidate], [q(e) for e in base]
                if any(c['low_count'] != b['low_count'] or abs(c['mean_budget_fraction'] - b['mean_budget_fraction']) > 1e-12 for c, b in zip(candidate_q, baseline_q)):
                    raise ValueError('quota candidate and baseline budgets differ')
                changes = [c['unsafe_rate'] - b['unsafe_rate'] for c, b in zip(candidate_q, baseline_q)]
                mean = float(np.mean([c['unsafe_rate'] for c in candidate_q]))
                competitors = {n: float(np.mean([q(e)['unsafe_rate'] for e in by_name[n]])) for n in ('last_risk', 'history', 'spectrum')}
                scenarios.append({**scenario, 'candidate_unsafe_rate': mean, 'competitor_unsafe_rates': competitors,
                                  'paired_last_risk_changes': changes,
                                  'passes': bool(all(mean < v - 1e-12 for v in competitors.values()) and sum(c < -1e-12 for c in changes) >= 2)})
            checks.append({'dataset': dataset, 'model': name, 'paired_brier_changes': brier_changes,
                           'mean_brier_change': float(np.mean(brier_changes)), 'primary_scenarios': scenarios,
                           'passes_attribution_screen': bool(np.mean(brier_changes) < -1e-12 and
                              sum(c < -1e-12 for c in brier_changes) >= 2 and all(s['passes'] for s in scenarios))})
    return {'checks': checks, 'any_attribution_candidate': any(c['passes_attribution_screen'] for c in checks),
            'runtime_promotion': False, 'independent_confirmation': False}


def noise_controls(protocol):
    import cupy as cp
    from scipy.stats import beta
    from experiments.revision_rmt_validation import normalize_parts, null_spectra, observed_geometry
    cases = []
    for kind in ('iid', 'ar'):
        for replicate in range(protocol['noise_replicates_per_kind']):
            seed = protocol['seed'] + 10000 + replicate + (1000 if kind == 'ar' else 0)
            rng = np.random.default_rng(seed)
            t, n, m = protocol['noise_window'], protocol['noise_rows'], protocol['noise_outputs']
            raw = rng.normal(size=(t, n, m)).astype(np.float32)
            if kind == 'ar':
                rho = protocol['ar_rho']
                for epoch in range(1, t):
                    raw[epoch] = rho * raw[epoch - 1] + np.sqrt(1 - rho ** 2) * raw[epoch]
            for centered in (False, True):
                parts = normalize_parts(cp.asarray(raw), cp.ones_like(cp.asarray(raw)), centered, cp)
                observed, _ = observed_geometry(parts, t)
                nulls = null_spectra(parts, t, seed + 100000, protocol['permutations'], [t])[:, 0]
                checks = empirical_checks(observed[0], nulls, protocol['alpha'])
                cases.append({'kind': kind, 'replicate': replicate, 'seed': seed, 'centered': centered,
                              'rows': n, 'null_rows': n, 'observed': observed[0].tolist(), 'nulls': nulls.tolist(),
                              'checks': checks})
                del parts, observed, nulls
                cp.get_default_memory_pool().free_all_blocks()
            if (replicate + 1) % 6 == 0:
                print(json.dumps({'NULL_CALIBRATION_PROGRESS': {'kind': kind, 'replicates_done': replicate + 1}}), flush=True)
    summary = []
    for kind in ('iid', 'ar'):
        for centered in (False, True):
            block = [c for c in cases if c['kind'] == kind and c['centered'] == centered]
            for method in ('rankwise', 'holm', 'maxT'):
                counts = [len(c['checks'][method]) for c in block]
                k, n = sum(v > 0 for v in counts), len(counts)
                interval = [0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
                            1. if k == n else float(beta.ppf(.975, k + 1, n - k))]
                summary.append({'kind': kind, 'centered': centered, 'method': method, 'replicates': n,
                                'false_positive_cases': k, 'false_positive_rate': k / n,
                                'mean_false_components': float(np.mean(counts)), 'binomial_95_interval': interval})
    return {'cases': cases, 'summary': summary, 'scope': 'single window only; no formal real-data guarantee'}


def functional_controls(protocol):
    import cupy as cp
    from experiments.revision_rmt_validation import normalize_parts, null_spectra, observed_geometry
    G, H, masks = rare_split_control(protocol)
    end, window, k, l2 = protocol['functional_end'], protocol['functional_window'], protocol['structure_budget'], protocol['l2']
    parts = normalize_parts(cp.asarray(G[:end]), cp.asarray(H[:end]), True, cp)
    values, vectors = observed_geometry(parts, window)
    nulls = null_spectra(parts, window, protocol['seed'] + 200000, protocol['permutations'], [end])[:, 0]
    index = end - window
    checks = empirical_checks(values[index], nulls, protocol['alpha'])
    selected_basis = vectors[index][:, checks['maxT']]
    loading = np.sum(values[index, checks['maxT']][None, :] * selected_basis ** 2, axis=1)
    sums = split_sums(G[end - 1], H[end - 1], masks)
    profiles = gains_from_sums(sums, l2)
    full_gains = profiles.sum(axis=1)
    def top(values):
        return np.argsort(-np.asarray(values), kind='stable')[:k]
    energy = top((G[end - 1].astype(float) ** 2).sum(axis=0))
    subsets = {'full': np.arange(G.shape[2]), 'energy_topk': energy,
               'spectral_loading_topk': top(loading) if selected_basis.shape[1] else energy,
               'contribution_oracle_topk': top(profiles.max(axis=0))}
    methods = []
    for name, columns in subsets.items():
        scores = profiles[:, columns].sum(axis=1)
        # Scores below numerical epsilon are the explicit no-split action.
        chosen = int(np.argmax(scores)) if scores.max() > 1e-6 else None
        selected_gain = 0. if chosen is None else float(full_gains[chosen])
        full_best = float(full_gains.max())
        if chosen is None:
            leaves = [(-(sums['left_g'][0] + sums['right_g'][0]) /
                         (sums['left_h'][0] + sums['right_h'][0] + l2)).tolist()]
        else:
            leaves = [(-sums[f'{side}_g'][chosen] / (sums[f'{side}_h'][chosen] + l2)).tolist()
                      for side in ('left', 'right')]
        methods.append({'method': name, 'outputs': columns.tolist(), 'scores': scores.tolist(),
                        'chosen_candidate': chosen, 'selected_full_gain': selected_gain,
                        'full_gain_regret': max(0., (full_best - selected_gain) / (abs(full_best) + 1e-6)),
                        'retains_important_output': bool(0 in columns),
                        'all_original_output_leaf_values': leaves})
    rare_report = {'shape': list(G.shape), 'rare_objects': int(masks[0].sum()), 'important_output': 0,
                   'candidate_names': ['rare', 'common'], 'l2': l2,
                   'full_candidate_gains': full_gains.tolist(), 'sufficient_sums': {key: value.tolist() for key, value in sums.items()},
                   'spectrum': {'observed': values[index].tolist(), 'nulls': nulls.tolist(), 'checks': checks},
                   'important_axis_retention_in_maxT_basis': float(np.sum(selected_basis[0] ** 2)),
                   'spectral_empty_basis_fallback_to_energy': not bool(selected_basis.shape[1]),
                   'methods': methods, 'contribution_oracle_is_not_a_runtime_method': True}
    if np.argmax(full_gains) != 0:
        raise ValueError('rare control does not have the stipulated full winner')
    del G, H, parts, values, vectors, nulls
    cp.get_default_memory_pool().free_all_blocks()
    G, H, masks, groups = duplicate_split_control(protocol)
    sums = split_sums(G, H, masks)
    full_gains = gains_from_sums(sums, l2).sum(axis=1)
    merged = []
    for weighted in (False, True):
        aggregated, penalties = aggregate_duplicate_sums(sums, groups, l2, weighted)
        scores = gains_from_sums(aggregated, penalties).sum(axis=1)
        chosen = int(np.argmax(scores))
        leaf_difference = 0.
        for mask in masks:
            original = -G[mask, 0].sum() / (H[mask, 0].sum() + l2)
            factor = 16 if weighted else 1
            representative = -(factor * G[mask, 0].sum()) / (factor * H[mask, 0].sum() + factor * l2)
            leaf_difference = max(leaf_difference, abs(original - representative))
        merged.append({'weighted': weighted, 'candidate_gains': scores.tolist(), 'chosen_candidate': chosen,
                       'selected_full_gain': float(full_gains[chosen]),
                       'full_gain_regret': float((full_gains.max() - full_gains[chosen]) / (abs(full_gains.max()) + 1e-6)),
                       'max_gain_difference_from_full': float(np.max(abs(scores - full_gains))),
                       'representative_leaf_max_difference': float(leaf_difference)})
    if not np.allclose(merged[1]['candidate_gains'], full_gains, atol=1e-8, rtol=1e-10):
        raise ValueError('weighted exact-duplicate aggregation fails gain conservation')
    # Actual full leaf values always use every original output and object.
    leaves = [[(-G[mask].sum(axis=0) / (H[mask].sum(axis=0) + l2)).tolist(),
               (-G[~mask].sum(axis=0) / (H[~mask].sum(axis=0) + l2)).tolist()] for mask in masks]
    duplicate_report = {'shape': list(G.shape), 'candidate_names': ['rare', 'common'], 'l2': l2,
                        'duplicate_groups': groups, 'full_candidate_gains': full_gains.tolist(),
                        'full_winner': int(np.argmax(full_gains)), 'all_original_output_leaf_values': leaves,
                        'sufficient_sums': {key: value.tolist() for key, value in sums.items()}, 'merging': merged}
    return {'rare_important_split': rare_report, 'exact_duplicate_contributions': duplicate_report,
            'scope': 'controlled single-node objective, not an end-to-end boosting experiment'}


def run_followup(config):
    started = time.perf_counter()
    records, provenance = verified_saved_records(config['source_receipt'], config['source_receipt_sha256'])
    for key in ('forecast_report', 'rmt_report'):
        if file_sha256(config[key]) != config[key + '_sha256']:
            raise ValueError(f'{key} checksum mismatch')
    parent = json.loads(Path(config['rmt_report']).read_text(encoding='utf-8'))
    original = json.loads(Path(config['forecast_report']).read_text(encoding='utf-8'))
    rows = parent['rows']
    def key(r):
        return r['dataset'], r['seed'], r['iteration'], r['horizon']
    old = {key(r): r for r in original['rows']}
    if len(rows) != 978 or len({key(r) for r in rows}) != 978 or len(records) != 6:
        raise ValueError('source rows/trajectories mismatch')
    for row in rows:
        if {k: row[k] for k in old[key(row)]} != old[key(row)]:
            raise ValueError('parent causal fields/targets changed')
    protocol = config['followup_protocol']
    print(json.dumps({'FOLLOWUP_VERIFIED_INPUTS': provenance, 'rows': len(rows), 'new_boosting_trainings': 0}), flush=True)
    evaluation = forecasts_at_quotas(rows, protocol)
    summary = quota_summary(evaluation, protocol)
    gate = followup_screen(evaluation, protocol)
    print(json.dumps({'FOLLOWUP_BUDGET_DONE': {'evaluations': len(evaluation), 'gate': gate}}), flush=True)
    noise = noise_controls(protocol)
    print(json.dumps({'FOLLOWUP_NULL_DONE': noise['summary']}), flush=True)
    functional = functional_controls(protocol)
    signal_checks = []
    for source in parent['synthetic']:
        for mode in ('second_moment', 'centered'):
            for item in source['analysis'][mode]['sensitivity']:
                signal_checks.append({'kind': source['kind'], 'mode': mode, 'end': item['end'],
                                      'known_signal_rank': source['known_signal_rank'],
                                      'checks': empirical_checks(item['observed_eigenvalues'], item['null_eigenvalues'], protocol['alpha'])})
    directory = Path('results/revision')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('rmt_followup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    report = {'protocol': {**protocol, 'new_boosting_trainings': 0, 'runtime_claim': False,
                          'quota_selection': 'current unsafe probability minus training prior; fixed counts, no test suffix sorting',
                          'exact_action_histograms': True, 'unit': 'whole trajectory',
                          'forest_targets': ['risk25', 'risk50', 'risk75', 'unsafe25', 'unsafe50', 'unsafe75'],
                          'null_scope': 'single-window pilot; real-process exchangeability unproved',
                          'previous_test_trajectories_reused': True},
              'provenance': {**provenance, 'forecast_report_sha256': config['forecast_report_sha256'],
                             'rmt_report_sha256': config['rmt_report_sha256']},
              'rows': rows, 'evaluation': evaluation, 'quota_summary': summary, 'screen': gate,
              'null_calibration': noise, 'saved_synthetic_checks': signal_checks,
              'functional': functional, 'analysis_seconds': time.perf_counter() - started}
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'FINAL_SUMMARY': {'report': str(path.resolve()), 'new_boosting_trainings': 0,
                    'seconds': report['analysis_seconds'], 'screen': gate,
                    'primary_next_control_summary': [s for s in summary if s['primary'] and s['horizon'] == 'next_control'],
                    'noise_summary': noise['summary'], 'rare_split_methods': functional['rare_important_split']['methods'],
                    'duplicate_merging': functional['exact_duplicate_contributions']['merging']}}), flush=True)
