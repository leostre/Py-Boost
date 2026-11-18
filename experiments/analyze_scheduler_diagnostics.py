"""Evaluate H-SCHED-1 and scheduler false activations from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence

from scipy.stats import mannwhitneyu


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Return the left-continuous weighted quantile used by the live scheduler."""

    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))][0])
    target = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def aggregate_tree_risks(records: Iterable[Mapping[str, object]]) -> Dict[int, float]:
    """Aggregate node regrets into one gain-weighted p95 value per tree."""

    grouped = defaultdict(list)
    for record in records:
        if record.get('regret') is not None:
            grouped[int(record['iteration'])].append(record)
    return {
        iteration: weighted_quantile(
            [float(item['regret']) for item in items],
            [max(0.0, float(item['full_best_gain'])) for item in items],
            0.95,
        )
        for iteration, items in grouped.items()
    }


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the signed probability-of-superiority effect size."""

    if not left or not right:
        raise ValueError('both samples must be non-empty')
    greater = sum(x > y for x in left for y in right)
    lower = sum(x < y for x in left for y in right)
    return (greater - lower) / (len(left) * len(right))


def analyze_report(report_path: Path) -> List[Dict[str, object]]:
    """Create one hypothesis-oriented row for every run with split artifacts."""

    with report_path.open(encoding='utf-8') as stream:
        report = json.load(stream)
    rows: List[Dict[str, object]] = []
    for run in report['results']:
        diagnostics_path = run.get('split_diagnostics_path')
        if not diagnostics_path:
            continue
        with Path(diagnostics_path).open(encoding='utf-8') as stream:
            records = json.load(stream)
        tree_risks = aggregate_tree_risks(records)
        early = [risk for iteration, risk in tree_risks.items() if iteration <= 5]
        late = [risk for iteration, risk in tree_risks.items() if iteration > 10]
        applied_iterations = {
            int(item['iteration'])
            for item in records
            if bool(item.get('applied_to_structure'))
        }
        applied_risks = [
            risk for iteration, risk in tree_risks.items() if iteration in applied_iterations
        ]
        risk_threshold = float(run.get('gain_risk_threshold', 0.05))
        row: Dict[str, object] = {
            'dataset': run['dataset'],
            'case_id': run['case_id'],
            'seed': run['seed'],
            'trees_measured': len(tree_risks),
            'early_trees': len(early),
            'late_trees': len(late),
            'early_risk_mean': mean(early) if early else None,
            'late_risk_mean': mean(late) if late else None,
            'early_minus_late': (
                mean(early) - mean(late) if early and late else None
            ),
            'cliffs_delta_early_vs_late': (
                cliffs_delta(early, late) if early and late else None
            ),
            'mannwhitney_p_early_greater': (
                float(mannwhitneyu(early, late, alternative='greater').pvalue)
                if early and late
                else None
            ),
            'applied_trees_measured': len(applied_risks),
            'false_activation_rate': (
                mean(float(risk > risk_threshold) for risk in applied_risks)
                if applied_risks
                else None
            ),
            'gain_risk_threshold': risk_threshold,
            'history_decision_rate': run.get('history_decision_rate'),
            'accuracy_delta_vs_full': run.get('accuracy_delta_vs_full'),
            'train_speedup_vs_full': run.get('train_speedup_vs_full'),
        }
        rows.append(row)
    return rows


def save_rows(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/diagnostics/scheduler_hypotheses.csv'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_report(args.report)
    print(save_rows(rows, args.output).resolve())


if __name__ == '__main__':
    main()
