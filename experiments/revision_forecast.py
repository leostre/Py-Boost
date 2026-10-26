"""Causal diagnostics; trajectories, not overlapping windows, are held out."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from py_boost.callbacks.optimization import OptimizationTrajectoryRecorder


def energy_features(energy):
    energy = np.asarray(energy, dtype=float)
    total = float(energy.sum())
    p = energy / total if total > 0 else np.zeros_like(energy)
    ordered = np.sort(p)[::-1]
    return {'log_energy': float(np.log1p(total)),
            'entropy': float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p))),
            **{f'mass_{b}': float(ordered[:b].sum()) for b in (25, 50, 75)}}


class SubsetTrajectoryRecorder(OptimizationTrajectoryRecorder):
    """Fixed train-only rows; cheap features use the actual full gradient."""
    def __init__(self, *args, subset_rows=1024, subset_seed=1000000, **kwargs):
        super().__init__(*args, **kwargs)
        if subset_rows < 1:
            raise ValueError('subset_rows must be positive')
        self.subset_rows, self.subset_seed = subset_rows, subset_seed
        self.row_indices = None
        self.cheap_trace = []

    def before_train(self, build_info):
        super().before_train(build_info)
        self.row_indices = None
        self.cheap_trace.clear()

    def before_iteration(self, build_info):
        train = build_info['data']['train']
        gradient, hessian = train['grad'], train['hess']
        if self.row_indices is None:
            rng = np.random.default_rng(self.subset_seed)
            self.row_indices = np.sort(rng.choice(gradient.shape[0],
                         min(self.subset_rows, gradient.shape[0]), replace=False))
        energy = self._to_host((gradient ** 2).sum(axis=0))
        self.cheap_trace.append({'iteration': int(build_info['num_iter']) + 1,
                                 **energy_features(energy)})
        context = {**build_info, 'data': {**build_info['data'], 'train': {
            **train, 'grad': gradient[self.row_indices], 'hess': hessian[self.row_indices]}}}
        super().before_iteration(context)

    def manifest(self):
        return {**super().manifest(), 'subset_seed': self.subset_seed,
                'train_row_indices': self.row_indices.tolist(), 'cheap_trace': self.cheap_trace}


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data['manifest']))
        G = np.stack([data[f'gradient_{t:04d}'] for t in manifest['iterations']])
        H = np.stack([data[f'hessian_{t:04d}'] for t in manifest['iterations']])
    return manifest, G, H


def window_moments(G, H, centered=False):
    """Separate per-epoch RMS; a constant H disappears only in centered mode."""
    parts = []
    for raw in (G, np.broadcast_to(H, G.shape)):
        value = raw.astype(np.float64)
        if centered:
            value = value - value.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(value ** 2, axis=(1, 2), keepdims=True))
        value = value / np.maximum(scale, 1e-12)
        parts.append(np.matmul(value.transpose(0, 2, 1), value) / value.shape[1])
    return (parts[0] + parts[1]) / 2


def spectral_features(moments, window=16):
    output, previous = {}, None
    for index in range(window - 1, len(moments)):
        covariance = moments[index - window + 1:index + 1].mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues[::-1], 0)
        vectors = vectors[:, ::-1]
        p = eigenvalues / max(eigenvalues.sum(), 1e-12)
        basis = vectors[:, :4]
        drift = 0. if previous is None else float(1 - np.linalg.norm(previous.T @ basis) ** 2 / 4)
        rank90 = int(np.searchsorted(np.cumsum(p), .90) + 1)
        output[index + 1] = {'spectral_rank': float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) / len(p)),
                             'spectral_top4_mass': float(p[:4].sum()),
                             'fixed_rank4_drift': max(0., drift), 'rank90_fraction': rank90 / len(p)}
        previous = basis
    return output


def future_risks(risk_trace, iteration, horizon, control_interval=10, first_control=5):
    end = iteration + 1 if horizon == 'next_tree' else (
        first_control + ((iteration - first_control) // control_interval + 1) * control_interval)
    if end > max(risk_trace):
        return None
    # Invalid/missing measurements remain unknown, never silently treated as safe.
    values = [risk_trace.get(t) for t in range(iteration + 1, end + 1)]
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return float(max(values))


def build_forecast_rows(cheap, spectra, traces, dataset, seed, outputs, planned_trees=100):
    from experiments.analyze_rmt_budget import choose_budget_label
    rows, history = [], None
    keys = ('log_energy', 'entropy', 'mass_25', 'mass_50', 'mass_75')
    for item in cheap:
        t = int(item['iteration'])
        current = np.asarray([item[key] for key in keys])
        previous = current if history is None else history
        history = .9 * previous + .1 * current
        if t not in spectra:
            continue
        last_control = 5 + ((t - 5) // 10) * 10
        for horizon in ('next_tree', 'next_control'):
            risks = {b: future_risks(trace, t, horizon) for b, trace in traces.items()}
            if any(value is None for value in risks.values()):
                continue
            _, label = choose_budget_label(risks, outputs, .10)
            row = {'dataset': dataset, 'seed': seed, 'iteration': t, 'horizon': horizon,
                   'progress': t / planned_trees, 'n_outputs': outputs, 'budget_label': label,
                   'safe_threshold': .10, 'risk_age': t - last_control,
                   **{key: float(item[key]) for key in keys}, **spectra[t],
                   **{f'ema_{key}': float(history[i]) for i, key in enumerate(keys)},
                   **{f'change_{key}': float(current[i] - previous[i]) for i, key in enumerate(keys)},
                   **{f'risk_{b}': risk for b, risk in risks.items()},
                   **{f'last_risk_{b}': traces[b][last_control] for b in traces}}
            rows.append(row)
    return rows


def evaluate_forecasts(rows, additional_sets=None, save_predictions=False):
    from sklearn.ensemble import RandomForestClassifier
    from experiments.analyze_rmt_budget import evaluate_budget_actions
    cheap = ('progress', 'log_energy', 'entropy', 'mass_25', 'mass_50', 'mass_75')
    strong = cheap + ('last_risk_25', 'last_risk_50', 'last_risk_75', 'risk_age')
    sets = {'iteration': ('progress',), 'current': cheap, 'last_risk': strong,
            'history': strong + tuple(f'{prefix}_{key}' for prefix in ('ema', 'change') for key in cheap[1:]),
            'spectrum': strong + ('spectral_rank', 'spectral_top4_mass', 'fixed_rank4_drift', 'rank90_fraction')}
    if additional_sets:
        sets.update({name: strong + tuple(keys) for name, keys in additional_sets.items()})
    output = []
    for horizon in ('next_tree', 'next_control'):
        selected = [r for r in rows if r['horizon'] == horizon]
        groups = sorted({(r['dataset'], r['seed']) for r in selected})
        for mode in ('trajectory', 'dataset'):
            folds = groups if mode == 'trajectory' else sorted({g[0] for g in groups})
            for held in folds:
                def held_out(r):
                    return (r['dataset'], r['seed']) == held if mode == 'trajectory' else r['dataset'] == held
                train = [r for r in selected if not held_out(r)
                         and (mode != 'trajectory' or r['dataset'] == held[0])]
                test = [r for r in selected if held_out(r)]
                for name, keys in sets.items():
                    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight='balanced_subsample', random_state=20260902, n_jobs=1)
                    model.fit([[r[k] for k in keys] for r in train], [r['budget_label'] for r in train])
                    predictions = model.predict([[r[k] for k in keys] for r in test])
                    levels = (25, 50, 75)
                    fractions = [(levels[int(p)] if p < 3 else r['n_outputs']) / r['n_outputs']
                                 for r, p in zip(test, predictions)]
                    result = {'horizon': horizon, 'holdout': mode, 'held_out': held, 'model': name,
                                   'train_rows': len(train), 'test_rows': len(test),
                                   'accuracy': float(np.mean(predictions == [r['budget_label'] for r in test])),
                                   'mean_budget_fraction': float(np.mean(fractions)),
                                   **evaluate_budget_actions(test, predictions, .10)}
                    if save_predictions:
                        result['feature_keys'] = keys
                        result['predictions'] = [int(p) for p in predictions]
                        result['test_keys'] = [[r['dataset'], r['seed'], r['iteration']] for r in test]
                    output.append(result)
    return output


def analyze_forecasts(records, path):
    from py_boost.multioutput.split_diagnostics import action_gain_regret, summarize_split_risks
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['dataset'], r['seed'])].append(r)
    rows, validation = [], []
    for (dataset, seed), group in sorted(grouped.items()):
        full = next(r for r in group if r['case_id'] == 'matched_trajectory')
        if len(group) != 1 or full['scheduler_mode'] != 'never' or full['history_decisions']:
            raise ValueError('risk and features must come from one unchanged full-statistics trajectory')
        traces = {}
        all_nodes = json.loads(Path(full['split_diagnostics_path']).read_text(encoding='utf-8'))
        for budget in (25, 50, 75):
            nodes = [node for node in all_nodes if int(node['diagnostic_output_budget']) == budget]
            by_iteration = defaultdict(list)
            for node in nodes:
                node['regret'] = action_gain_regret(float(node['full_best_gain']), float(node['selected_full_gain']),
                                       bool(node['diagnostic_split_available']), bool(node['full_split_available']))
                by_iteration[int(node['iteration'])].append(node)
            trace = {}
            for t, nodes_t in by_iteration.items():
                summary = summarize_split_risks(nodes_t)
                trace[t] = 1. if summary.spurious_split_count else summary.weighted_quantile
            traces[budget] = trace
        manifest, G, H = load_trajectory(full['trajectory_path'])
        spectra = spectral_features(window_moments(G, H))
        rows.extend(build_forecast_rows(manifest['cheap_trace'], spectra, traces, dataset, seed, full['n_outputs']))
        validation.append({'dataset': dataset, 'seed': seed, 'risk_features_share_same_tree': True,
                           'snapshot_rows': G.shape[1], 'iterations': len(G),
                           'risk_iterations_by_budget': {str(b): len(trace) for b, trace in traces.items()}})
        print(json.dumps({'FORECAST_SOURCE': validation[-1]}), flush=True)
    evaluation = evaluate_forecasts(rows)
    report = {'rows': rows, 'evaluation': evaluation, 'validation': validation,
              'protocol': {'unit': 'whole trajectory', 'within_trajectory_split': False,
                           'purge': 'not applicable: whole trajectory holdout', 'window': 16,
                           'causal_features': True, 'planned_trees': 100,
                           'spectral_features': 'matched-subset second moment, not RMT significance',
                           'runtime_claim': False}}
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = []
    for horizon in ('next_tree', 'next_control'):
        for name in ('iteration', 'current', 'last_risk', 'history', 'spectrum'):
            items = [r for r in evaluation if r['horizon'] == horizon and r['holdout'] == 'trajectory' and r['model'] == name]
            summary.append({'horizon': horizon, 'model': name, 'trajectories': len(items),
                            'mean_unsafe_rate': float(np.mean([r['unsafe_rate'] for r in items])),
                            'mean_budget_fraction': float(np.mean([r['mean_budget_fraction'] for r in items])),
                            'mean_accuracy': float(np.mean([r['accuracy'] for r in items]))})
    return {'forecast_report': str(path.resolve()), 'summary': summary,
            'interpretation': 'offline causal pilot; correlated windows do not establish controller speedup'}
