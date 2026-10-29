"""Paired bootstrap confirmation for repeated HASBoost benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_METRICS = (
    "f1_weighted",
    "f1_micro",
    "log_loss",
    "train_seconds",
    "gpu_peak_fit_delta_mib",
    "mean_output_budget",
    "budget_transitions",
)


def paired_bootstrap_interval(
    deltas: Sequence[float],
    repetitions: int = 20_000,
    seed: int = 20260902,
) -> tuple[float, float, float]:
    """Return mean and a deterministic percentile interval for paired deltas."""

    values = np.asarray(deltas, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired deltas must be finite and non-empty")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(lower), float(upper)


def analyze_records(
    records: Iterable[Mapping[str, object]],
    baseline_case: str,
    candidate_cases: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Pair cases by dataset and seed, rejecting duplicates and reporting gaps."""

    rows = list(records)
    counts = Counter(
        (str(row.get("dataset")), str(row.get("case_id")), int(row.get("seed", -1)))
        for row in rows
    )
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate run keys: {duplicates}")
    indexed = {
        (str(row["dataset"]), str(row["case_id"]), int(row["seed"])): row
        for row in rows
    }
    datasets = sorted({str(row["dataset"]) for row in rows})
    output = []
    missing_pairs = []
    for dataset in datasets:
        seeds = sorted(
            int(row["seed"])
            for row in rows
            if row["dataset"] == dataset and row["case_id"] == baseline_case
        )
        for candidate in candidate_cases:
            pairs = []
            for seed in seeds:
                baseline = indexed.get((dataset, baseline_case, seed))
                treatment = indexed.get((dataset, candidate, seed))
                if baseline is None or treatment is None:
                    missing_pairs.append((dataset, candidate, seed))
                    continue
                pairs.append((seed, baseline, treatment))
            for metric_index, metric in enumerate(metrics):
                deltas = [
                    float(treatment[metric]) - float(baseline[metric])
                    for _, baseline, treatment in pairs
                    if baseline.get(metric) is not None
                    and treatment.get(metric) is not None
                ]
                if not deltas:
                    continue
                mean_delta, lower, upper = paired_bootstrap_interval(
                    deltas,
                    seed=20260902 + metric_index,
                )
                output.append(
                    {
                        "dataset": dataset,
                        "baseline_case": baseline_case,
                        "candidate_case": candidate,
                        "metric": metric,
                        "pairs": len(deltas),
                        "mean_delta_candidate_minus_baseline": mean_delta,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "all_deltas": ";".join(f"{value:.12g}" for value in deltas),
                    }
                )
    validation = {
        "records": len(rows),
        "datasets": datasets,
        "duplicate_keys": len(duplicates),
        "missing_pairs": [list(item) for item in missing_pairs],
        "complete": not duplicates and not missing_pairs,
    }
    return output, validation


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline-case", default="full")
    parser.add_argument("--candidate-cases", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rows, validation = analyze_records(
        payload.get("results", payload),
        args.baseline_case,
        args.candidate_cases,
    )
    summary_path = args.output_dir / "paired_summary.csv"
    validation_path = args.output_dir / "validation.json"
    _write_csv(summary_path, rows)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)
    print(json.dumps({"summary": str(summary_path.resolve()), "validation": str(validation_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
