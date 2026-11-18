"""Evaluate whether spectral output probabilities predict split importance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.analyze_scheduler_diagnostics import weighted_quantile
from experiments.analyze_spectral_risk import safe_spearman


def aggregate_output_attribution(
    records: Iterable[Mapping[str, object]],
) -> Dict[int, Tuple[float, ...]]:
    """Return a gain-weighted p95 leave-one-output regret vector per tree."""

    grouped: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        values = record.get('leave_one_output_regret')
        if values is not None and bool(record.get('full_split_available', True)):
            grouped[int(record['iteration'])].append(record)

    result: Dict[int, Tuple[float, ...]] = {}
    for iteration, items in grouped.items():
        output_count = len(items[0]['leave_one_output_regret'])
        if any(len(item['leave_one_output_regret']) != output_count for item in items):
            raise ValueError('output attribution vectors must have a stable size')
        weights = [max(0.0, float(item['full_best_gain'])) for item in items]
        result[iteration] = tuple(
            weighted_quantile(
                [float(item['leave_one_output_regret'][index]) for item in items],
                weights,
                0.95,
            )
            for index in range(output_count)
        )
    return result


def budget_capture(
    probabilities: Sequence[float],
    importance: Sequence[float],
    budget: int,
) -> Tuple[float, float, float]:
    """Return spectral, optimal, and uniform-expected captured importance."""

    probabilities = np.asarray(probabilities, dtype=float)
    importance = np.asarray(importance, dtype=float)
    if probabilities.shape != importance.shape or probabilities.ndim != 1:
        raise ValueError('probabilities and importance must be equal-length vectors')
    if not 1 <= budget <= probabilities.size:
        raise ValueError('budget must be within the output dimension')
    total = float(np.maximum(importance, 0).sum())
    if total <= 0:
        return 0.0, 0.0, budget / probabilities.size
    selected = np.argsort(probabilities)[-budget:]
    optimal = np.argsort(importance)[-budget:]
    return (
        float(importance[selected].sum() / total),
        float(importance[optimal].sum() / total),
        budget / probabilities.size,
    )


def evaluate_output_probabilities(
    spectral_rows: Iterable[Mapping[str, object]],
    attributions: Mapping[int, Sequence[float]],
    budgets: Sequence[int],
) -> List[Dict[str, object]]:
    """Create per-window H-RMT-2 ranking and budget-capture diagnostics."""

    rows: List[Dict[str, object]] = []
    for spectral in spectral_rows:
        iteration = int(spectral['end_iteration'])
        importance = attributions.get(iteration)
        if importance is None:
            continue
        probabilities = tuple(float(value) for value in spectral['output_probabilities'])
        if len(probabilities) != len(importance):
            raise ValueError('spectral policy and attribution dimensions differ')
        row: Dict[str, object] = {
            'representation': spectral['representation'],
            'window': int(spectral['window']),
            'end_iteration': iteration,
            'output_spearman': safe_spearman(probabilities, importance),
        }
        for budget in budgets:
            if budget > len(probabilities):
                continue
            captured, optimal, uniform = budget_capture(
                probabilities,
                importance,
                budget,
            )
            row[f'capture_{budget}'] = captured
            row[f'optimal_capture_{budget}'] = optimal
            row[f'uniform_expected_{budget}'] = uniform
            row[f'capture_ratio_to_optimal_{budget}'] = (
                captured / optimal if optimal > 0 else None
            )
        rows.append(row)
    return rows


def summarize_evaluation(
    rows: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate H-RMT-2 evidence by representation and window."""

    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row['representation']), int(row['window']))].append(row)
    summaries: List[Dict[str, object]] = []
    for (representation, window), group in sorted(grouped.items()):
        correlations = [
            float(row['output_spearman'])
            for row in group
            if row.get('output_spearman') is not None
        ]
        summary: Dict[str, object] = {
            'representation': representation,
            'window': window,
            'windows': len(group),
            'output_spearman_mean': mean(correlations) if correlations else None,
            'positive_spearman_fraction': (
                mean(float(value > 0) for value in correlations)
                if correlations
                else None
            ),
        }
        ratio_keys = sorted(
            key
            for key in group[0]
            if key.startswith('capture_ratio_to_optimal_')
        )
        for key in ratio_keys:
            values = [float(row[key]) for row in group if row.get(key) is not None]
            summary[f'{key}_mean'] = mean(values) if values else None
        capture_keys = sorted(
            key
            for key in group[0]
            if key.startswith('capture_')
            and not key.startswith('capture_ratio_to_optimal_')
        )
        for key in capture_keys:
            budget = key.rsplit('_', 1)[1]
            values = [float(row[key]) for row in group if row.get(key) is not None]
            uniform_values = [
                float(row[f'uniform_expected_{budget}'])
                for row in group
                if row.get(f'uniform_expected_{budget}') is not None
            ]
            captured_mean = mean(values) if values else None
            uniform_mean = mean(uniform_values) if uniform_values else None
            summary[f'{key}_mean'] = captured_mean
            summary[f'capture_lift_over_uniform_{budget}'] = (
                captured_mean / uniform_mean
                if captured_mean is not None and uniform_mean
                else None
            )
        summaries.append(summary)
    return summaries


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('spectral_json', type=Path)
    parser.add_argument('split_diagnostics_json', type=Path)
    parser.add_argument('--budgets', nargs='+', type=int, default=[10, 25, 50])
    parser.add_argument('--output-dir', type=Path, default=Path('results/output_attribution'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.spectral_json.open(encoding='utf-8') as stream:
        spectral_rows = json.load(stream)
    with args.split_diagnostics_json.open(encoding='utf-8') as stream:
        split_records = json.load(stream)
    attributions = aggregate_output_attribution(split_records)
    evaluation = evaluate_output_probabilities(
        spectral_rows,
        attributions,
        args.budgets,
    )
    summary = summarize_evaluation(evaluation)
    detail_path = args.output_dir / 'output_attribution_windows.csv'
    summary_path = args.output_dir / 'output_attribution_summary.csv'
    _write_csv(evaluation, detail_path)
    _write_csv(summary, summary_path)
    print(
        json.dumps(
            {
                'details': str(detail_path.resolve()),
                'summary': str(summary_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
