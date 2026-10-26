"""Replay the H5-2 audit schedule over checksum-frozen full trajectories."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.h5_2_event_scheduler_core import (
    AuditProtocol,
    empirical_quantile,
    replay_schedule,
    volatility_changes,
)
from experiments.rmt_subspace_reproducibility import _risk_trace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_receipts(report: dict) -> list[dict]:
    provenance = report["provenance"]
    receipts = [*provenance["reused_source"]["verified_artifacts"],
                *provenance["new_artifacts"]]
    expected = {(name, int(seed))
                for name in report["protocol"]["datasets"]
                for seed in report["protocol"]["model_seeds"]}
    actual = {(item["dataset"], int(item["model_seed"])) for item in receipts}
    if len(receipts) != len(expected) or actual != expected:
        raise ValueError("source receipts do not cover the frozen trajectory grid")
    return receipts


def _trace(receipt: dict, budget: int, trees: int) -> tuple[dict[int, float], dict[int, float | None]]:
    artifact = receipt["artifacts"]["split_diagnostics_path"]
    path = Path(artifact["path"])
    if _sha256(path) != artifact["sha256"].lower():
        raise ValueError(f"split diagnostic checksum mismatch: {path}")
    risks = _risk_trace(str(path), budget)
    if set(risks) != set(range(1, trees + 1)):
        raise ValueError(f"missing per-tree risk at budget {budget}: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = [item for item in records
                if int(item["diagnostic_output_budget"]) == budget]
    volatility = {}
    spurious = set()
    for item in selected:
        iteration = int(item["iteration"])
        value = item.get("gradient_norm_volatility")
        if iteration in volatility and volatility[iteration] != value:
            raise ValueError("volatility differs within one tree")
        volatility[iteration] = float(value) if value is not None else None
        if item.get("spurious_split"):
            spurious.add(iteration)
    # Preserve the existing scheduler's conservative treatment of spurious splits.
    for iteration in spurious:
        risks[iteration] = max(1.0, risks[iteration])
    return risks, volatility


def _summary(rows: list[dict], dataset: str, stage: str) -> dict:
    selected = [row for row in rows
                if row["dataset"] == dataset and row["stage"] == stage]
    out = {"dataset": dataset, "stage": stage, "trajectories": len(selected)}
    for mode in ("periodic", "event"):
        for key in ("audits", "reduced_trees", "safe_reduced_trees",
                    "unsafe_applied_trees", "excess_risk_sum"):
            out[f"{mode}_{key}"] = sum(row[mode][key] for row in selected)
    out["unsafe_reduction_fraction"] = (
        (out["periodic_unsafe_applied_trees"] - out["event_unsafe_applied_trees"])
        / out["periodic_unsafe_applied_trees"]
        if out["periodic_unsafe_applied_trees"] else None
    )
    out["safe_exposure_ratio"] = (
        out["event_safe_reduced_trees"] / out["periodic_safe_reduced_trees"]
        if out["periodic_safe_reduced_trees"] else None
    )
    return out


def run_h5_2_event_scheduler(config: dict) -> dict:
    specification = dict(config["h5_2_event_scheduler"])
    source_path = Path(specification["source_report"])
    source_sha = _sha256(source_path)
    if source_sha != specification["source_report_sha256"].lower():
        raise ValueError("frozen source report checksum mismatch")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if report["decision"]["verdict"] != specification["source_verdict"]:
        raise ValueError("source report verdict differs from protocol")
    receipts = _source_receipts(report)
    protocol = AuditProtocol(
        trees=int(specification["trees"]),
        warmup_audits=tuple(specification["warmup_audits"]),
        block_size=int(specification["block_size"]),
        safe_threshold=float(specification["safe_threshold"]),
        margin_fraction=float(specification["margin_fraction"]),
        margin_delay=int(specification["margin_delay"]),
    )
    seeds = tuple(int(seed) for seed in report["protocol"]["model_seeds"])
    development = set(int(seed) for seed in specification["development_seeds"])
    holdout = set(int(seed) for seed in specification["holdout_seeds"])
    if development & holdout or development | holdout != set(seeds):
        raise ValueError("development and holdout seeds must partition the sources")
    budgets = {name: int(value["risk"])
               for name, value in report["protocol"]["budgets"].items()}

    loaded = []
    development_changes = defaultdict(list)
    for receipt in receipts:
        dataset = receipt["dataset"]
        seed = int(receipt["model_seed"])
        risks, volatility = _trace(receipt, budgets[dataset], protocol.trees)
        loaded.append((dataset, seed, risks, volatility))
        if seed in development:
            development_changes[dataset].extend(
                volatility_changes(volatility, protocol.warmup_audits[-1] + 1,
                                   protocol.trees)
            )
    quantile = float(specification["alarm_quantile"])
    alarm_thresholds = {dataset: empirical_quantile(tuple(changes), quantile)
                        for dataset, changes in development_changes.items()}
    results = []
    for dataset, seed, risks, volatility in loaded:
        periodic = replay_schedule(risks, volatility, protocol, "periodic")
        event = replay_schedule(risks, volatility, protocol, "event",
                                alarm_thresholds[dataset])
        if periodic["audits"] != event["audits"]:
            raise AssertionError("audit budgets do not match")
        row = {
            "dataset": dataset,
            "model_seed": seed,
            "stage": "development" if seed in development else "holdout",
            "budget": budgets[dataset],
            "alarm_threshold": alarm_thresholds[dataset],
            "periodic": periodic,
            "event": event,
        }
        results.append(row)
        print(json.dumps({"TRAJECTORY_RESULT": {
            key: value for key, value in row.items() if key not in ("periodic", "event")
        } | {"periodic_unsafe": periodic["unsafe_applied_trees"],
             "event_unsafe": event["unsafe_applied_trees"],
             "periodic_safe": periodic["safe_reduced_trees"],
             "event_safe": event["safe_reduced_trees"]}}), flush=True)

    summaries = [_summary(results, dataset, stage)
                 for dataset in specification["datasets"]
                 for stage in ("development", "holdout")]
    primary = next(item for item in summaries
                   if item["dataset"] == specification["primary_dataset"]
                   and item["stage"] == "holdout")
    minimum_danger = int(specification["minimum_periodic_unsafe"])
    if primary["periodic_unsafe_applied_trees"] < minimum_danger:
        verdict = "inconclusive_insufficient_unsafe_exposure"
    elif (primary["unsafe_reduction_fraction"] is not None
          and primary["unsafe_reduction_fraction"] >= float(specification["minimum_unsafe_reduction"])
          and primary["safe_exposure_ratio"] is not None
          and primary["safe_exposure_ratio"] >= float(specification["minimum_safe_exposure_ratio"])
          and primary["event_excess_risk_sum"] <= primary["periodic_excess_risk_sum"]):
        verdict = "pilot_support"
    else:
        verdict = "rejected_or_tradeoff"
    output = {
        "hypothesis": "H5-2 event-driven audit scheduling",
        "verdict": verdict,
        "source_report": str(source_path),
        "source_report_sha256": source_sha,
        "protocol": specification,
        "alarm_thresholds": alarm_thresholds,
        "summaries": summaries,
        "primary": primary,
        "trajectories": results,
        "new_boosting_trainings": 0,
        "new_spectral_decompositions": 0,
        "interpretation_scope": "causal offline replay on previously collected full trajectories",
    }
    output_path = Path(specification["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": {
        "hypothesis": output["hypothesis"],
        "verdict": verdict,
        "source_report_sha256": source_sha,
        "alarm_thresholds": alarm_thresholds,
        "primary": primary,
        "summaries": summaries,
        "output_path": str(output_path.resolve()),
        "new_boosting_trainings": 0,
        "new_spectral_decompositions": 0,
    }}, ensure_ascii=False), flush=True)
    return output
