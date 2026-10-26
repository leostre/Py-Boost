"""Summarize per-iteration output budgets from adaptive HASBoost reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def summarize_budget_trace(trace):
    """Return stable, JSON-serializable diagnostics for one scheduler trace."""

    records = [item for item in trace if item.get("output_budget") is not None]
    budgets = [int(item["output_budget"]) for item in records]
    if not budgets:
        return {
            "mean_output_budget": None,
            "min_output_budget": None,
            "max_output_budget": None,
            "budget_transitions": 0,
            "first_reduction_iteration": None,
            "budget_histogram": "",
        }
    full_budget = max(budgets)
    first_reduction = next(
        (
            int(item["iteration"])
            for item in records
            if int(item["output_budget"]) < full_budget
        ),
        None,
    )
    histogram = Counter(budgets)
    return {
        "mean_output_budget": mean(budgets),
        "min_output_budget": min(budgets),
        "max_output_budget": full_budget,
        "budget_transitions": sum(
            left != right for left, right in zip(budgets, budgets[1:])
        ),
        "first_reduction_iteration": first_reduction,
        "budget_histogram": ";".join(
            f"{budget}:{histogram[budget]}" for budget in sorted(histogram)
        ),
    }


def _read_report(path: Path):
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload.get("results", payload)


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_reports(paths):
    rows = []
    for report_path in paths:
        for record in _read_report(report_path):
            trace_path = record.get("scheduler_trace_path")
            if not trace_path:
                continue
            with Path(trace_path).open(encoding="utf-8") as stream:
                trace = json.load(stream)
            rows.append(
                {
                    "dataset": record.get("dataset"),
                    "case_id": record.get("case_id"),
                    "seed": record.get("seed"),
                    "train_seconds": record.get("train_seconds"),
                    "accuracy": record.get("accuracy"),
                    "f1_weighted": record.get("f1_weighted"),
                    "f1_micro": record.get("f1_micro"),
                    "log_loss": record.get("log_loss"),
                    **summarize_budget_trace(trace),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diagnostics/adaptive_budget_trace_summary.csv"),
    )
    args = parser.parse_args()
    rows = analyze_reports(args.reports)
    _write_csv(args.output, rows)
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
