"""Relate offline spectral state to current and next-tree split risk."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from experiments.analyze_scheduler_diagnostics import aggregate_tree_risks


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Return Spearman's rho when at least three non-constant pairs exist."""

    if len(left) != len(right):
        raise ValueError('paired samples must have the same length')
    finite = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if np.isfinite(x) and np.isfinite(y)
    ]
    if len(finite) < 3:
        return None
    x_values, y_values = zip(*finite)
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None
    return float(spearmanr(x_values, y_values).statistic)


def _tree_volatilities(
    split_records: Iterable[Mapping[str, object]],
) -> Dict[int, float]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for record in split_records:
        value = record.get('gradient_norm_volatility')
        if value is not None:
            grouped[int(record['iteration'])].append(float(value))
    return {
        iteration: float(np.mean(values))
        for iteration, values in grouped.items()
    }


def join_spectral_risk(
    spectral_rows: Iterable[Mapping[str, object]],
    split_records: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Join spectral windows with measured risk at t and t+1."""

    split_records = tuple(split_records)
    tree_risks = aggregate_tree_risks(split_records)
    volatilities = _tree_volatilities(split_records)
    joined: List[Dict[str, object]] = []
    for row in spectral_rows:
        iteration = int(row['end_iteration'])
        angle = row.get('mean_principal_angle_deg')
        joined.append(
            {
                **row,
                'end_iteration': iteration,
                'mean_principal_angle_deg': (
                    None if angle in (None, '') else float(angle)
                ),
                'tree_risk': tree_risks.get(iteration),
                'next_tree_risk': tree_risks.get(iteration + 1),
                'gradient_norm_volatility': volatilities.get(iteration),
            }
        )
    return joined


def summarize_joined(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compute H-RMT-3 correlation diagnostics per representation and window."""

    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row['representation']), int(row['window']))].append(row)

    summaries: List[Dict[str, object]] = []
    for (representation, window), group in sorted(grouped.items()):
        angle_pairs = [
            (float(row['mean_principal_angle_deg']), float(row['next_tree_risk']))
            for row in group
            if row.get('mean_principal_angle_deg') is not None
            and row.get('next_tree_risk') is not None
        ]
        baseline_rows = [
            row for row in group if row.get('next_tree_risk') is not None
        ]
        volatility_pairs = [
            (float(row['gradient_norm_volatility']), float(row['next_tree_risk']))
            for row in baseline_rows
            if row.get('gradient_norm_volatility') is not None
        ]
        angle_rho = safe_spearman(
            [pair[0] for pair in angle_pairs],
            [pair[1] for pair in angle_pairs],
        )
        iteration_rho = safe_spearman(
            [float(row['end_iteration']) for row in baseline_rows],
            [float(row['next_tree_risk']) for row in baseline_rows],
        )
        volatility_rho = safe_spearman(
            [pair[0] for pair in volatility_pairs],
            [pair[1] for pair in volatility_pairs],
        )
        summaries.append(
            {
                'representation': representation,
                'window': window,
                'spectral_windows': len(group),
                'angle_next_risk_pairs': len(angle_pairs),
                'angle_next_risk_spearman': angle_rho,
                'iteration_next_risk_spearman': iteration_rho,
                'volatility_next_risk_spearman': volatility_rho,
                'angle_beats_available_baselines': (
                    None
                    if angle_rho is None
                    else abs(angle_rho) > max(
                        [
                            abs(value)
                            for value in (iteration_rho, volatility_rho)
                            if value is not None
                        ]
                        or [-1.0]
                    )
                ),
            }
        )
    return summaries


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('spectral_csv', type=Path)
    parser.add_argument('split_diagnostics_json', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results/spectral_risk'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.split_diagnostics_json.open(encoding='utf-8') as stream:
        split_records = json.load(stream)
    joined = join_spectral_risk(_read_csv(args.spectral_csv), split_records)
    summaries = summarize_joined(joined)
    joined_path = args.output_dir / 'spectral_risk_joined.csv'
    summary_path = args.output_dir / 'spectral_risk_summary.csv'
    _write_csv(joined, joined_path)
    _write_csv(summaries, summary_path)
    print(
        json.dumps(
            {
                'joined': str(joined_path.resolve()),
                'summary': str(summary_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
