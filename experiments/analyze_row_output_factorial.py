"""Estimate row/output interaction contrasts from factorial HASBoost runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def factorial_contrast(full: float, rows: float, outputs: float, joint: float) -> float:
    """Return the two-factor interaction on an additive metric scale."""

    return float(joint - rows - outputs + full)


def _read_report(path: Path):
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload.get("results", payload)


def _case_map(records, dataset, seed):
    return {
        record["case_id"]: record
        for record in records
        if record.get("dataset") == dataset and int(record.get("seed")) == seed
    }


def analyze_records(records):
    details = []
    datasets_and_seeds = sorted(
        {(record["dataset"], int(record["seed"])) for record in records}
    )
    for dataset, seed in datasets_and_seeds:
        cases = _case_map(records, dataset, seed)
        full = cases.get("full")
        if full is None:
            continue
        for policy in ("current", "history"):
            rows = cases.get(f"{policy}_rows_only")
            outputs = next(
                (
                    record
                    for case_id, record in cases.items()
                    if case_id.startswith(f"{policy}_outputs_only_")
                ),
                None,
            )
            joint = next(
                (
                    record
                    for case_id, record in cases.items()
                    if case_id.startswith(f"{policy}_joint_")
                ),
                None,
            )
            if rows is None or outputs is None or joint is None:
                continue
            result = {"dataset": dataset, "seed": seed, "policy": policy}
            for metric in ("f1_weighted", "f1_micro", "log_loss", "train_seconds"):
                values = tuple(float(item[metric]) for item in (full, rows, outputs, joint))
                result[f"{metric}_row_delta"] = values[1] - values[0]
                result[f"{metric}_output_delta"] = values[2] - values[0]
                result[f"{metric}_joint_delta"] = values[3] - values[0]
                result[f"{metric}_interaction"] = factorial_contrast(*values)
            details.append(result)
    return details


def aggregate_details(details):
    summaries = []
    for dataset, policy in sorted({(row["dataset"], row["policy"]) for row in details}):
        rows = [
            row for row in details
            if row["dataset"] == dataset and row["policy"] == policy
        ]
        summary = {
            "dataset": dataset,
            "policy": policy,
            "runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for key in rows[0]:
            if key in {"dataset", "seed", "policy"}:
                continue
            values = [float(row[key]) for row in rows]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/row_output_factorial_analysis"),
    )
    args = parser.parse_args()
    records = [record for path in args.reports for record in _read_report(path)]
    details = analyze_records(records)
    summaries = aggregate_details(details)
    details_path = args.output_dir / "factorial_details.csv"
    summary_path = args.output_dir / "factorial_summary.csv"
    _write_csv(details_path, details)
    _write_csv(summary_path, summaries)
    print(
        json.dumps(
            {
                "details": str(details_path.resolve()),
                "summary": str(summary_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
