"""Test whether spectral state predicts the smallest safe output budget."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from math import log1p
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from py_boost.multioutput.split_diagnostics import action_gain_regret, summarize_split_risks


def evaluate_budget_actions(rows, predicted_labels, safe_threshold=0.10):
    """Evaluate measured risks, never infer safety from ordinal budget ordering."""
    if len(rows) != len(predicted_labels):
        raise ValueError('predictions must align with rows')
    risks = []
    unknown = 0
    for row, label in zip(rows, predicted_labels):
        levels = sorted({int(key[5:]) for key in row
                         if re.fullmatch(r'risk_[1-9][0-9]*', key)}
                        | {int(row['n_outputs'])})
        if not 0 <= int(label) < len(levels):
            raise ValueError('predicted budget label outside measured levels')
        budget = levels[int(label)]
        risk = 0.0 if budget == int(row['n_outputs']) else row.get(f'risk_{budget}')
        if risk is None or not np.isfinite(float(risk)):
            unknown += 1
        else:
            risks.append(float(risk))
    return {
        'unsafe_rate': float(np.mean(np.asarray(risks) > safe_threshold)) if risks else None,
        'mean_selected_risk': float(np.mean(risks)) if risks else None,
        'evaluated_actions': len(risks),
        'unknown_actions': unknown,
        'unsafe_count': sum(risk > safe_threshold for risk in risks),
        'safe_threshold': safe_threshold,
    }


def choose_budget_label(
    risks_by_budget: Mapping[int, float | None],
    full_budget: int,
    safe_threshold: float,
) -> tuple[int, int]:
    """Return the smallest safe budget and its ordinal index."""

    if safe_threshold < 0:
        raise ValueError("safe threshold must be non-negative")
    levels = tuple(sorted(set(risks_by_budget) | {int(full_budget)}))
    for index, budget in enumerate(levels[:-1]):
        risk = risks_by_budget.get(budget)
        if risk is not None and np.isfinite(risk) and risk <= safe_threshold:
            return budget, index
    return full_budget, len(levels) - 1


def _iteration_risks(records: Iterable[Mapping[str, object]]) -> dict[int, float | None]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        corrected = dict(record)
        if 'diagnostic_split_available' in record:
            corrected['regret'] = action_gain_regret(
                float(record['full_best_gain']), float(record['selected_full_gain']),
                bool(record['diagnostic_split_available']), bool(record['full_split_available']))
        grouped[int(record["iteration"])].append(corrected)
    return {
        iteration: summarize_split_risks(items).weighted_quantile
        for iteration, items in grouped.items()
    }


def build_budget_dataset(
    report: Mapping[str, object],
    safe_threshold: float = 0.10,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Join spectral traces to per-iteration risk curves without leakage."""

    runs = list(report.get("results", ()))
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["dataset"]), int(run["seed"]))].append(run)
    rows = []
    missing = []
    for (dataset, seed), group in sorted(grouped.items()):
        full = next((run for run in group if run["case_id"] == "full"), None)
        spectral = next(
            (run for run in group if str(run["case_id"]).startswith("spectral_probe_")),
            None,
        )
        current_runs = [
            run for run in group if str(run["case_id"]).startswith("current_topk_")
        ]
        if full is None or spectral is None or len(current_runs) < 1:
            missing.append([dataset, seed, "required_case"])
            continue
        full_budget = int(full["n_outputs"])
        traces_by_budget = {}
        for run in current_runs:
            path = run.get("split_diagnostics_path")
            if not path:
                missing.append([dataset, seed, run["case_id"]])
                continue
            with Path(str(path)).open(encoding="utf-8") as stream:
                traces_by_budget[int(run["sketch_outputs"])] = _iteration_risks(
                    json.load(stream)
                )
        spectral_path = spectral.get("spectral_trace_path")
        if not spectral_path:
            missing.append([dataset, seed, "spectral_trace"])
            continue
        with Path(str(spectral_path)).open(encoding="utf-8") as stream:
            spectral_trace = json.load(stream)
        max_iteration = max(int(item["iteration"]) for item in spectral_trace)
        for item in spectral_trace:
            iteration = int(item["iteration"])
            risks = {
                budget: trace.get(iteration)
                for budget, trace in traces_by_budget.items()
            }
            budget, label = choose_budget_label(risks, full_budget, safe_threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "iteration": iteration,
                    "progress": iteration / max_iteration,
                    "n_outputs": full_budget,
                    "safe_budget": budget,
                    "safe_budget_fraction": budget / full_budget,
                    "budget_label": label,
                    "safe_threshold": safe_threshold,
                    "spike_count_fraction": float(item["spike_count"]) / full_budget,
                    "effective_rank_fraction": float(item["effective_rank"]) / full_budget,
                    "spike_energy_fraction": float(item["spike_energy_fraction"]),
                    "log_relative_spike_margin": log1p(float(item["relative_spike_margin"])),
                    "subspace_drift": (
                        0.0 if item.get("subspace_drift") is None
                        else float(item["subspace_drift"])
                    ),
                    **{f"risk_{budget}": risk for budget, risk in sorted(risks.items())},
                }
            )
    keys = [(row["dataset"], row["seed"], row["iteration"]) for row in rows]
    validation = {
        "rows": len(rows),
        "groups": len(grouped),
        "duplicate_keys": len(keys) - len(set(keys)),
        "missing": missing,
        "complete": bool(rows) and len(keys) == len(set(keys)) and not missing,
    }
    return rows, validation


def cross_dataset_evaluation(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compare progress-only and spectral models under dataset holdout."""

    feature_sets = {
        "progress_only": ("progress",),
        "spectral_state": (
            "progress",
            "spike_count_fraction",
            "effective_rank_fraction",
            "spike_energy_fraction",
            "log_relative_spike_margin",
            "subspace_drift",
        ),
    }
    datasets = sorted({str(row["dataset"]) for row in rows})
    if len(datasets) < 2:
        raise ValueError("cross-dataset evaluation requires at least two datasets")
    output = []
    for held_out in datasets:
        train = [row for row in rows if row["dataset"] != held_out]
        test = [row for row in rows if row["dataset"] == held_out]
        for name, features in feature_sets.items():
            X_train = np.asarray([[float(row[key]) for key in features] for row in train])
            y_train = np.asarray([int(row["budget_label"]) for row in train])
            X_test = np.asarray([[float(row[key]) for key in features] for row in test])
            y_test = np.asarray([int(row["budget_label"]) for row in test])
            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=20260902,
                n_jobs=1,
            )
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            output.append(
                {
                    "held_out_dataset": held_out,
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "accuracy": float(np.mean(predicted == y_test)),
                    "ordinal_underbudget_rate": float(np.mean(predicted < y_test)),
                    **evaluate_budget_actions(test, predicted, float(test[0].get('safe_threshold', 0.10))),
                    "mean_absolute_level_error": float(np.mean(np.abs(predicted - y_test))),
                    "mean_level_delta": float(np.mean(predicted - y_test)),
                }
            )
    return output


def seed_holdout_evaluation(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Evaluate spectral features within a dataset without mixing iterations."""

    feature_sets = {
        "progress_only": ("progress",),
        "spectral_state": (
            "progress",
            "spike_count_fraction",
            "effective_rank_fraction",
            "spike_energy_fraction",
            "log_relative_spike_margin",
            "subspace_drift",
        ),
    }
    output = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        seeds = sorted({int(row["seed"]) for row in dataset_rows})
        if len(seeds) < 2:
            continue
        for held_out_seed in seeds:
            train = [row for row in dataset_rows if int(row["seed"]) != held_out_seed]
            test = [row for row in dataset_rows if int(row["seed"]) == held_out_seed]
            for name, features in feature_sets.items():
                X_train = np.asarray([[float(row[key]) for key in features] for row in train])
                y_train = np.asarray([int(row["budget_label"]) for row in train])
                X_test = np.asarray([[float(row[key]) for key in features] for row in test])
                y_test = np.asarray([int(row["budget_label"]) for row in test])
                model = RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=20260902,
                    n_jobs=1,
                )
                model.fit(X_train, y_train)
                predicted = model.predict(X_test)
                output.append(
                    {
                        "dataset": dataset,
                        "held_out_seed": held_out_seed,
                        "model": name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "accuracy": float(np.mean(predicted == y_test)),
                        "ordinal_underbudget_rate": float(np.mean(predicted < y_test)),
                        **evaluate_budget_actions(test, predicted, float(test[0].get('safe_threshold', 0.10))),
                        "mean_absolute_level_error": float(
                            np.mean(np.abs(predicted - y_test))
                        ),
                        "mean_level_delta": float(np.mean(predicted - y_test)),
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--safe-threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as stream:
        report = json.load(stream)
    rows, validation = build_budget_dataset(report, args.safe_threshold)
    evaluation = cross_dataset_evaluation(rows)
    seed_evaluation = seed_holdout_evaluation(rows)
    dataset_path = args.output_dir / "rmt_budget_dataset.csv"
    evaluation_path = args.output_dir / "cross_dataset_evaluation.csv"
    seed_evaluation_path = args.output_dir / "seed_holdout_evaluation.csv"
    validation_path = args.output_dir / "validation.json"
    _write_csv(dataset_path, rows)
    _write_csv(evaluation_path, evaluation)
    _write_csv(seed_evaluation_path, seed_evaluation)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)
    print(json.dumps({"dataset": str(dataset_path.resolve()), "cross_dataset_evaluation": str(evaluation_path.resolve()), "seed_holdout_evaluation": str(seed_evaluation_path.resolve()), "validation": str(validation_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
