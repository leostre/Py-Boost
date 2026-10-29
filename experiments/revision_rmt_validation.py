"""Bounded matched-row empirical null validation. No new boosting fits."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.revision_forecast import evaluate_forecasts, load_trajectory
from experiments.revision_forecast_replay import file_sha256, verified_saved_records


def permutation_orders(rng, count, rows, outputs):
    """Independent row permutations per output; reuse for every epoch and G/H."""
    return np.stack([[rng.permutation(rows) for _ in range(outputs)]
                     for _ in range(count)]).transpose(0, 2, 1)


def normalize_parts(G, H, centered, xp=np):
    result = []
    for raw in (G, xp.broadcast_to(H, G.shape)):
        part = xp.asarray(raw, dtype=xp.float32)
        if centered:
            part = part - part.mean(axis=1, keepdims=True)
        scale = xp.sqrt(xp.mean(part ** 2, axis=(1, 2), keepdims=True))
        result.append(part / xp.maximum(scale, 1e-12))
    return result


def epoch_moments(parts, xp=np):
    return sum(xp.matmul(p.transpose(0, 2, 1), p) for p in parts) / (2 * parts[0].shape[1])


def rolling_moments(moments, window, xp=np):
    if window < 1 or window > moments.shape[-3]:
        raise ValueError('invalid causal window')
    # Time is third-last axis, including a possible leading permutation axis.
    zero = xp.zeros_like(moments[..., :1, :, :])
    cumulative = xp.concatenate((zero, xp.cumsum(moments, axis=-3)), axis=-3)
    return (cumulative[..., window:, :, :] - cumulative[..., :-window, :, :]) / window


def significant_indices(observed, null_values, protocol):
    threshold = np.quantile(null_values, protocol['null_quantile'], axis=0)
    tolerance = protocol['absolute_tolerance'] + protocol['relative_tolerance'] * np.maximum(np.abs(observed), np.abs(threshold))
    return np.flatnonzero(observed > threshold + tolerance), threshold


def basis_overlap(left, right):
    """Symmetric projector overlap with rank change reported separately."""
    if not left.shape[1] or not right.shape[1]:
        return None
    return float(np.clip(np.linalg.norm(left.T @ right) ** 2 / max(left.shape[1], right.shape[1]), 0, 1))


def null_spectra(parts, window, seed, count, ends=None, batch=4):
    """GPU covariance/eigh; null and observed always use exactly these rows."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    epochs, rows, outputs = parts[0].shape
    indices = None if ends is None else cp.asarray(np.asarray(ends) - window)
    result = []
    for start in range(0, count, batch):
        orders = cp.asarray(permutation_orders(rng, min(batch, count - start), rows, outputs))
        covariance = None
        for part in parts:
            permuted = part[:, orders, cp.arange(outputs)].transpose(1, 0, 2, 3)
            moment = cp.matmul(permuted.transpose(0, 1, 3, 2), permuted) / (2 * rows)
            covariance = moment if covariance is None else covariance + moment
            del permuted, moment
        windows = rolling_moments(covariance, window, cp)
        if indices is not None:
            windows = windows[:, indices]
        values = cp.maximum(cp.linalg.eigvalsh(windows)[..., ::-1], 0)
        result.append(cp.asnumpy(values))
        del covariance, windows, values, orders
    return np.concatenate(result)


def observed_geometry(parts, window):
    import cupy as cp
    moments = rolling_moments(epoch_moments(parts, cp), window, cp)
    values, vectors = cp.linalg.eigh(moments)
    return cp.asnumpy(cp.maximum(values[:, ::-1], 0)), cp.asnumpy(vectors[:, :, ::-1])


def analyze_arrays(G, H, protocol, seed, full_features=True, truth=None):
    import cupy as cp
    window = protocol['window']
    ends = protocol['control_ends']
    report, features = {}, {}
    for centered in (False, True):
        mode = 'centered' if centered else 'second_moment'
        parts = normalize_parts(cp.asarray(G), cp.asarray(H), centered, cp)
        values, vectors = observed_geometry(parts, window)
        if full_features:
            nulls = null_spectra(parts, window, seed, 20)
            # Generate the SAME nested permutation prefix. Only selected windows
            # use 100 nulls, so this is a bounded sensitivity study, not retuning.
            controls = null_spectra(parts, window, seed, 100, ends)
        else:
            controls = null_spectra(parts, window, seed, 100, ends)
            nulls = None
        sensitivity = []
        for e, end in enumerate(ends):
            item = {'end': end, 'observed_rows': int(G.shape[1]), 'null_rows': int(G.shape[1]),
                    'observed_eigenvalues': values[end - window].tolist(),
                    'null_eigenvalues': controls[:, e].tolist(),
                    'nested_counts': []}
            for count in protocol['sensitivity_permutations']:
                selected, threshold = significant_indices(values[end - window], controls[:count, e], protocol)
                result = {'permutations': count, 'indices': selected.tolist(),
                          'rank': len(selected), 'null_threshold': threshold.tolist()}
                if truth is not None:
                    basis = truth[end - 1]
                    selected_basis = vectors[end - window][:, selected]
                    result['true_basis_overlap'] = basis_overlap(basis, selected_basis)
                item['nested_counts'].append(result)
            sensitivity.append(item)
        if full_features:
            previous, previous_indices = None, None
            for i, end in enumerate(range(window, len(G) + 1)):
                selected, threshold = significant_indices(values[i], nulls[:, i], protocol)
                basis = vectors[i][:, selected]
                overlap = None if previous is None else basis_overlap(previous, basis)
                p = values[i] / max(float(values[i].sum()), 1e-12)
                feature = {f'{mode}_spike_fraction': len(selected) / len(p),
                           f'{mode}_spike_mass': float(p[selected].sum()),
                           f'{mode}_spectral_excess': float(np.maximum(values[i] - threshold, 0).sum() / max(float(values[i].sum()), 1e-12)),
                           f'{mode}_spike_drift': 0. if overlap is None else 1 - overlap,
                           f'{mode}_spike_basis_known': float(overlap is not None),
                           f'{mode}_rank_change': 0. if previous_indices is None else abs(len(selected) - len(previous_indices)) / len(p),
                           f'{mode}_fixed_rank4_drift': 0. if i == 0 else 1 - basis_overlap(vectors[i - 1, :, :4], vectors[i, :, :4])}
                features.setdefault(end, {}).update(feature)
                previous, previous_indices = basis, selected
        halves = [observed_geometry([p[:, split] for p in parts], window)[1]
                  for split in (slice(0, G.shape[1] // 2), slice(G.shape[1] // 2, None))]
        geometry = []
        for end in range(window, len(G) + 1, window):
            i = end - window
            for rank in protocol['fixed_ranks']:
                geometry.append({'end': end, 'rank': rank,
                                 'split_half_overlap': basis_overlap(halves[0][i, :, :rank], halves[1][i, :, :rank]),
                                 'previous_nonoverlap_window_overlap': None if end == window else
                                 basis_overlap(vectors[i - window, :, :rank], vectors[i, :, :rank])})
        report[mode] = {'sensitivity': sensitivity, 'geometry': geometry,
                        'constant_hessian_after_centering': bool(centered and np.max(np.abs(cp.asnumpy(parts[1]))) < 1e-7)}
        del parts, values, vectors, nulls, controls, halves
        cp.get_default_memory_pool().free_all_blocks()
    return report, features


def synthetic_control(kind, protocol):
    rng = np.random.default_rng(protocol['seed'])
    n, m, t = (protocol[key] for key in ('synthetic_rows', 'synthetic_outputs', 'synthetic_epochs'))
    orthogonal = np.linalg.qr(rng.normal(size=(m, 9)))[0]
    G = np.empty((t, n, m), dtype=np.float32)
    truth = []
    for epoch in range(t):
        loadings = orthogonal[:, 4:8] if kind == 'regime' and epoch >= 25 else orthogonal[:, :4].copy()
        if kind == 'duplicates':
            loadings[8:16] = loadings[:8]
        signal = 4 * rng.normal(size=(n, 4)) @ loadings.T
        basis = np.linalg.qr(loadings)[0][:, :4]
        if kind == 'rare':
            rare = np.zeros(n)
            rare[:max(1, int(.02 * n))] = rng.normal(0, 12, max(1, int(.02 * n)))
            signal += rare[:, None] * orthogonal[:, 8]
            basis = np.column_stack((basis, orthogonal[:, 8]))
        if kind == 'noise':
            signal = np.zeros_like(signal)
            basis = np.empty((m, 0))
        G[epoch] = signal + rng.normal(0, .2, size=(n, m))
        if kind == 'constant_hessian':
            G[epoch] = 0
            basis = np.empty((m, 0))
        truth.append(basis)
    return G, np.ones((t, n, m), dtype=np.float32), truth


def summarize_evaluation(evaluation):
    result = []
    for horizon in ('next_tree', 'next_control'):
        for model in sorted({e['model'] for e in evaluation}):
            items = [e for e in evaluation if e['horizon'] == horizon and e['model'] == model and e['holdout'] == 'trajectory']
            result.append({'horizon': horizon, 'model': model, 'trajectories': len(items),
                           **{f'mean_{key}': float(np.mean([e[key] for e in items])) for key in
                              ('unsafe_rate', 'mean_budget_fraction', 'accuracy')}})
    return result


def promotion_gate(evaluation):
    """Frozen conservative pilot screen; not a statistical/runtime guarantee."""
    checks = []
    for dataset in ('helena', 'mediamill'):
        items = [e for e in evaluation if e['horizon'] == 'next_control' and e['holdout'] == 'trajectory' and e['held_out'][0] == dataset]
        baselines = {name: [e for e in items if e['model'] == name] for name in ('last_risk', 'history', 'spectrum')}
        for model in ('rmt_second_moment', 'rmt_centered', 'rmt_combined'):
            selected = [e for e in items if e['model'] == model]
            unsafe = float(np.mean([e['unsafe_rate'] for e in selected]))
            budget = float(np.mean([e['mean_budget_fraction'] for e in selected]))
            paired = sum(e['unsafe_rate'] <= b['unsafe_rate'] + 1e-12 and
                         e['mean_budget_fraction'] <= b['mean_budget_fraction'] + 1e-12 and
                         (e['unsafe_rate'] < b['unsafe_rate'] - 1e-12 or e['mean_budget_fraction'] < b['mean_budget_fraction'] - 1e-12)
                         for e, b in zip(selected, baselines['last_risk']))
            dominates_all = all(unsafe <= np.mean([b['unsafe_rate'] for b in base]) + 1e-12 and
                                budget <= np.mean([b['mean_budget_fraction'] for b in base]) + 1e-12 and
                                (unsafe < np.mean([b['unsafe_rate'] for b in base]) - 1e-12 or
                                 budget < np.mean([b['mean_budget_fraction'] for b in base]) - 1e-12)
                                for base in baselines.values())
            checks.append({'dataset': dataset, 'model': model, 'mean_unsafe_rate': unsafe,
                           'mean_budget_fraction': budget, 'paired_last_risk_dominance': paired,
                           'passes_offline_screen': bool(dominates_all and paired >= 2)})
    return {'checks': checks, 'any_offline_candidate': any(c['passes_offline_screen'] for c in checks),
            'runtime_promotion': False, 'criterion': 'next-control risk/budget Pareto dominance of all three baselines and at least2/3 paired last-risk folds; descriptive only'}


def run_validation(config):
    started = time.perf_counter()
    records, provenance = verified_saved_records(config['source_receipt'], config['source_receipt_sha256'])
    if file_sha256(config['forecast_report']) != config['forecast_report_sha256']:
        raise ValueError('parent forecast report checksum mismatch')
    parent = json.loads(Path(config['forecast_report']).read_text(encoding='utf-8'))
    protocol = config['rmt_protocol']
    expected = {(d, s) for d in config['datasets'] for s in config['model_seeds']}
    if {(r['dataset'], r['model_seed']) for r in records} != expected or len(records) != 6:
        raise ValueError('source trajectories mismatch')
    print(json.dumps({'VERIFIED_INPUTS': provenance, 'new_boosting_trainings': 0}), flush=True)
    real, feature_map = [], {}
    for record in sorted(records, key=lambda r: (r['dataset'], r['model_seed'])):
        before = time.perf_counter()
        manifest, G, H = load_trajectory(record['trajectory_path'])
        if len(G) != 100 or len(manifest['train_row_indices']) != G.shape[1]:
            raise ValueError('invalid saved length/subset')
        result, features = analyze_arrays(G, H, protocol, protocol['seed'] + record['model_seed'])
        for t, item in features.items():
            feature_map[(record['dataset'], record['model_seed'], t)] = item
        real.append({'dataset': record['dataset'], 'seed': record['model_seed'], 'analysis': result,
                     'analysis_seconds': time.perf_counter() - before})
        print(json.dumps({'RMT_SOURCE_DONE': {k: v for k, v in real[-1].items() if k != 'analysis'}}), flush=True)
        del G, H
    rows = [{**r, **feature_map[(r['dataset'], r['seed'], r['iteration'])]} for r in parent['rows']]
    modes = {mode: tuple(sorted(k for k in rows[0] if k.startswith(mode + '_')))
             for mode in ('second_moment', 'centered')}
    sets = {'rmt_second_moment': modes['second_moment'], 'rmt_centered': modes['centered'],
            'rmt_combined': modes['second_moment'] + modes['centered']}
    evaluation = evaluate_forecasts(rows, sets, save_predictions=True)
    # Refit untouched baselines to verify exact parent aggregate reproduction.
    for old in parent['evaluation']:
        new = next(e for e in evaluation if all(e[k] == old[k] for k in ('horizon', 'holdout', 'model'))
                   and (list(e['held_out']) if isinstance(e['held_out'], tuple) else e['held_out']) == old['held_out'])
        for key in ('unsafe_rate', 'mean_budget_fraction', 'accuracy'):
            if abs(new[key] - old[key]) > 1e-12:
                raise ValueError('parent baseline not reproduced')
    print(json.dumps({'RMT_FORECAST_DONE': {'rows': len(rows), 'evaluations': len(evaluation)}}), flush=True)
    synthetic = []
    synthetic_protocol = {**protocol, 'control_ends': protocol['synthetic_control_ends']}
    for kind in ('rank4', 'regime', 'duplicates', 'rare', 'noise', 'constant_hessian'):
        G, H, truth = synthetic_control(kind, protocol)
        result, _ = analyze_arrays(G, H, synthetic_protocol, protocol['seed'], full_features=False, truth=truth)
        synthetic.append({'kind': kind, 'analysis': result, 'generative_seed': protocol['seed'],
                          'known_signal_rank': 5 if kind == 'rare' else 0 if kind in ('noise', 'constant_hessian') else 4})
        print(json.dumps({'RMT_SYNTHETIC_DONE': kind}), flush=True)
        del G, H, truth
    directory = Path('results/revision')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('rmt_validation_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    gate = promotion_gate(evaluation)
    report = {'protocol': {**protocol, 'null': 'independent row permutation per output, shared across time and G/H',
                          'same_observed_null_rows': True, 'empirical_rankwise_threshold_not_formal_significance': True,
                          'new_boosting_trainings': 0, 'runtime_claim': False,
                          'half_split': 'fixed first/second half of saved sorted train-only subset',
                          'forecast_folds_and_RF': 'unchanged from parent; predictions and test keys saved',
                          'rare_control': '2% rows, amplitude12; mathematical direction, not task-utility validation'},
              'provenance': {**provenance, 'forecast_report_sha256': config['forecast_report_sha256']},
              'real': real, 'synthetic': synthetic, 'rows': rows, 'evaluation': evaluation,
              'summary': summarize_evaluation(evaluation), 'gate': gate,
              'total_analysis_seconds': time.perf_counter() - started}
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'FINAL_SUMMARY': {'report': str(path.resolve()), 'summary': report['summary'],
                     'gate': gate, 'new_boosting_trainings': 0, 'seconds': report['total_analysis_seconds']}}), flush=True)
