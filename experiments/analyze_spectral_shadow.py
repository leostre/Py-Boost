"""Summarize post-warmup risk and stability of online spectral output policies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import exp, log, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence

from experiments.analyze_scheduler_diagnostics import aggregate_tree_risks
from py_boost.multioutput.split_diagnostics import summarize_split_risks


def probability_effective_rank(probabilities: Sequence[float]) -> float:
    """Return entropy perplexity of a normalized output distribution."""

    positive = tuple(float(value) for value in probabilities if value > 0)
    total = sum(positive)
    if total <= 0:
        return 0.0
    normalized = tuple(value / total for value in positive)
    return exp(-sum(value * log(value) for value in normalized if value > 0))


def consecutive_topk_jaccard(trace: Sequence[Mapping[str, object]]) -> float | None:
    """Return mean overlap between consecutive selected output sets."""

    if len(trace) < 2:
        return None
    similarities = []
    for left, right in zip(trace, trace[1:]):
        left_set = set(int(value) for value in left['top_outputs'])
        right_set = set(int(value) for value in right['top_outputs'])
        union = left_set | right_set
        similarities.append(len(left_set & right_set) / len(union) if union else 1.0)
    return mean(similarities)


def selected_loading_redundancy(item: Mapping[str, object]) -> float | None:
    """Mean absolute cosine between selected spectral loading profiles."""

    profiles = item.get('loading_profiles')
    selected = item.get('top_outputs')
    if not profiles or not selected or len(selected) < 2:
        return None
    rows = [tuple(float(value) for value in profiles[int(index)]) for index in selected]
    norms = [sqrt(sum(value * value for value in row)) for row in rows]
    similarities = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            denominator = norms[left] * norms[right]
            if denominator <= 0:
                continue
            similarities.append(
                abs(sum(a * b for a, b in zip(rows[left], rows[right])))
                / denominator
            )
    return mean(similarities) if similarities else None


def summarize_run(
    records: Iterable[Mapping[str, object]],
    spectral_trace: Iterable[Mapping[str, object]],
    min_iteration: int,
) -> Dict[str, object]:
    """Aggregate one shadow run after the spectral window becomes available."""

    selected = [
        record for record in records if int(record['iteration']) >= min_iteration
    ]
    trace = [
        item for item in spectral_trace if int(item['iteration']) >= min_iteration
    ]
    risks = summarize_split_risks(selected)
    tree_risks = aggregate_tree_risks(selected)
    return {
        'min_iteration': min_iteration,
        'iterations': len(tree_risks),
        'nodes': len(selected),
        'risk_nodes': risks.node_count,
        'split_match_rate': (
            mean(float(record['same_split']) for record in selected)
            if selected
            else None
        ),
        'spurious_split_rate': (
            mean(float(record.get('spurious_split', False)) for record in selected)
            if selected
            else None
        ),
        'split_regret_mean': risks.mean,
        'split_regret_max': risks.maximum,
        'split_regret_weighted_p95': risks.weighted_quantile,
        'tree_risk_mean': mean(tree_risks.values()) if tree_risks else None,
        'spectral_updates': len(trace),
        'spike_count_mean': (
            mean(float(item['spike_count']) for item in trace) if trace else None
        ),
        'spectral_effective_rank_mean': (
            mean(float(item['effective_rank']) for item in trace) if trace else None
        ),
        'spike_energy_fraction_mean': (
            mean(float(item['spike_energy_fraction']) for item in trace)
            if trace and 'spike_energy_fraction' in trace[0]
            else None
        ),
        'relative_spike_margin_mean': (
            mean(float(item['relative_spike_margin']) for item in trace)
            if trace and 'relative_spike_margin' in trace[0]
            else None
        ),
        'subspace_drift_mean': (
            mean(
                float(item['subspace_drift'])
                for item in trace
                if item.get('subspace_drift') is not None
            )
            if any(item.get('subspace_drift') is not None for item in trace)
            else None
        ),
        'selected_loading_redundancy_mean': (
            mean(values)
            if (
                values := [
                    value
                    for item in trace
                    for value in (selected_loading_redundancy(item),)
                    if value is not None
                ]
            )
            else None
        ),
        'probability_effective_rank_mean': (
            mean(probability_effective_rank(item['probabilities']) for item in trace)
            if trace
            else None
        ),
        'consecutive_topk_jaccard': consecutive_topk_jaccard(trace),
    }


def analyze_report(report_path: Path, min_iteration: int) -> List[Dict[str, object]]:
    """Load benchmark artifacts and create one post-warmup row per run."""

    with report_path.open(encoding='utf-8') as stream:
        report = json.load(stream)
    rows: List[Dict[str, object]] = []
    for run in report['results']:
        diagnostics_path = run.get('split_diagnostics_path')
        trace_path = run.get('spectral_trace_path')
        if not diagnostics_path:
            continue
        with Path(diagnostics_path).open(encoding='utf-8') as stream:
            records = json.load(stream)
        trace = []
        if trace_path:
            with Path(trace_path).open(encoding='utf-8') as stream:
                trace = json.load(stream)
        rows.append(
            {
                'dataset': run['dataset'],
                'case_id': run['case_id'],
                'seed': int(run['seed']),
                'sketch_outputs': int(run['sketch_outputs']),
                'train_seconds': run.get('train_seconds'),
                'get_weights_total_ms': run.get('get_weights_total_ms'),
                'get_indexers_total_ms': run.get('get_indexers_total_ms'),
                **summarize_run(records, trace, min_iteration),
            }
        )
    return rows


def aggregate_rows(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate post-warmup evidence across seeds for each policy budget."""

    groups: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row['dataset']), str(row['case_id']))].append(row)
    metrics = (
        'train_seconds',
        'get_weights_total_ms',
        'get_indexers_total_ms',
        'split_match_rate',
        'spurious_split_rate',
        'split_regret_mean',
        'split_regret_max',
        'split_regret_weighted_p95',
        'tree_risk_mean',
        'spike_count_mean',
        'spectral_effective_rank_mean',
        'spike_energy_fraction_mean',
        'relative_spike_margin_mean',
        'subspace_drift_mean',
        'selected_loading_redundancy_mean',
        'probability_effective_rank_mean',
        'consecutive_topk_jaccard',
    )
    output = []
    for (dataset, case_id), group in sorted(groups.items()):
        row: Dict[str, object] = {
            'dataset': dataset,
            'case_id': case_id,
            'sketch_outputs': group[0]['sketch_outputs'],
            'runs': len(group),
            'seeds': ','.join(str(item['seed']) for item in sorted(group, key=lambda x: int(x['seed']))),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[f'{metric}_mean'] = mean(values) if values else None
            row[f'{metric}_std'] = stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(row)
    return output


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write heterogeneous mapping rows with a stable header."""

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
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--min-iteration', type=int, default=16)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/diagnostics/spectral_shadow'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for report in args.reports
        for row in analyze_report(report, args.min_iteration)
    ]
    summary = aggregate_rows(rows)
    details_path = args.output_dir / 'spectral_shadow_post_warmup_runs.csv'
    summary_path = args.output_dir / 'spectral_shadow_post_warmup_summary.csv'
    write_csv(rows, details_path)
    write_csv(summary, summary_path)
    print(json.dumps({'details': str(details_path.resolve()), 'summary': str(summary_path.resolve())}, indent=2))


if __name__ == '__main__':
    main()
