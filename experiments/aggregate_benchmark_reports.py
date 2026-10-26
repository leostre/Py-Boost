"""Combine independent HASBoost benchmark reports into one result bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from experiments.diagnostic_results import (
    add_grouped_baseline_ratios,
    aggregate_records,
)


def load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Load per-run rows from benchmark JSON files in deterministic order."""
    records: List[Dict[str, Any]] = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if "results" not in payload or "metadata" not in payload:
            continue
        for raw_record in payload["results"]:
            record = dict(raw_record)
            record["source_report"] = path.name
            records.append(record)
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_combined_bundle(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Recompute matching baselines and persist combined raw/summary tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = add_grouped_baseline_ratios(records)
    summary = aggregate_records(normalized)
    json_path = output_dir / "combined_results.json"
    runs_path = output_dir / "combined_runs.csv"
    summary_path = output_dir / "combined_summary.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"results": normalized, "summary": summary},
            stream,
            indent=2,
        )
    _write_csv(runs_path, normalized)
    _write_csv(summary_path, summary)
    return {"json": json_path, "runs_csv": runs_path, "summary_csv": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/diagnostics/final"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    input_paths = [
        path
        for path in args.input_dir.glob("*.json")
        if path.name != "combined_results.json"
    ]
    records = load_records(input_paths)
    if not records:
        raise RuntimeError(f"No benchmark records found in {args.input_dir}")
    paths = save_combined_bundle(records, output_dir)
    print(json.dumps({key: str(path.resolve()) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
