"""Run checksum-verified H5-3A/H5-3B analyses without retraining."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.revision_forecast_replay import file_sha256
from experiments.rmt_residual_expanded_core import validate_scalar_grid
from experiments.rmt_residual_signal_core import (
    cross_trajectory_residual_rows,
    local_linear_innovation_rows,
    summarize_residual_verdict,
)


def _verified_source(protocol):
    path = Path(protocol["source_report"]).resolve()
    actual_sha256 = file_sha256(path)
    if actual_sha256 != protocol["source_report_sha256"]:
        raise ValueError("source H5-3 report checksum mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    source_hypothesis = protocol.get(
        "source_hypothesis", "H5-3 robust scalar spectral state"
    )
    if report.get("hypothesis") != source_hypothesis:
        raise ValueError("unexpected source hypothesis")
    if report.get("decision", {}).get("verdict") != protocol["source_verdict"]:
        raise ValueError("source verdict differs from the frozen protocol")
    source_protocol = report.get("protocol", {})
    checks = {
        "datasets": protocol["datasets"],
        "model_seeds": protocol["model_seeds"],
        "split_seeds": protocol["split_seeds"],
        "endpoints": protocol["source_endpoints"],
        "partition_repeats": protocol["partition_repeats"],
        "primary_feature": protocol["primary_feature"],
        "secondary_features": protocol["secondary_features"],
    }
    for name, expected in checks.items():
        if source_protocol.get(name) != expected:
            raise ValueError(f"source protocol mismatch: {name}")
    features = [protocol["primary_feature"], *protocol["secondary_features"]]
    expected_keys = {
        (dataset, seed, end, feature)
        for dataset in protocol["datasets"]
        for seed in protocol["model_seeds"]
        for end in protocol["source_endpoints"]
        for feature in features
    }
    source_rows_key = protocol.get("source_rows_key", "rows")
    source_rows = report.get(source_rows_key, [])
    actual_keys = [
        (str(row["dataset"]), int(row["model_seed"]), int(row["end"]),
         str(row["feature"]))
        for row in source_rows
    ]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise ValueError("source rows differ from the frozen Cartesian protocol")
    split_by_model_seed = dict(zip(protocol["model_seeds"], protocol["split_seeds"]))
    risks = {}
    for row in source_rows:
        model_seed = int(row["model_seed"])
        if int(row["split_seed"]) != split_by_model_seed[model_seed]:
            raise ValueError("source split seed differs from the frozen protocol")
        repeats = int(row["partition_repeats"])
        if repeats != protocol["partition_repeats"]:
            raise ValueError("source partition count differs from the frozen protocol")
        if not all(len(row[name]) == repeats for name in (
            "partition_a_values", "partition_b_values"
        )):
            raise ValueError("source partition vectors have an unexpected length")
        risk_key = (str(row["dataset"]), model_seed, int(row["end"]))
        risk = float(row["current_risk"])
        if risk_key in risks and risks[risk_key] != risk:
            raise ValueError("source risk differs across spectral features")
        risks[risk_key] = risk
    return report, {
        "source_report": str(path),
        "source_report_sha256": actual_sha256,
        "source_experiment_id": source_protocol.get("experiment_id"),
        "source_verdict": report["decision"]["verdict"],
        "source_rows_key": source_rows_key,
        "verified_rows": len(actual_keys),
    }


def run_residual_signal(config):
    protocol = dict(config["rmt_residual_signal"])
    source, provenance = _verified_source(protocol)
    source_rows = source[protocol.get("source_rows_key", "rows")]
    mode = protocol["mode"]
    if mode == "cross_trajectory":
        rows = cross_trajectory_residual_rows(source_rows)
        hypothesis = protocol.get(
            "hypothesis", "H5-3A cross-trajectory spectral residuals"
        )
    elif mode == "local_linear":
        rows = local_linear_innovation_rows(source_rows)
        hypothesis = protocol.get(
            "hypothesis", "H5-3B causal local spectral innovations"
        )
    else:
        raise ValueError(f"unknown residual signal mode: {mode}")
    expected_endpoints = (
        protocol["source_endpoints"]
        if mode == "cross_trajectory"
        else protocol["source_endpoints"][2:]
    )
    actual_endpoints = sorted({int(row["end"]) for row in rows})
    if actual_endpoints != expected_endpoints:
        raise ValueError("transformed endpoints differ from the frozen protocol")
    features = [protocol["primary_feature"], *protocol["secondary_features"]]
    grid = validate_scalar_grid(
        rows,
        protocol["datasets"],
        protocol["model_seeds"],
        protocol["split_seeds"],
        expected_endpoints,
        features,
    )
    expected_rows = protocol.get("expected_transformed_rows")
    if expected_rows is not None and grid["rows"] != int(expected_rows):
        raise ValueError("transformed row count differs from the frozen protocol")
    decision = summarize_residual_verdict(
        rows,
        primary_feature=protocol["primary_feature"],
        secondary_features=protocol["secondary_features"],
        minimum_ccc=protocol["minimum_ccc"],
        minimum_sign_agreement=protocol["minimum_sign_agreement"],
        minimum_risk_rho=protocol["minimum_risk_rho"],
    )
    report = {
        "hypothesis": hypothesis,
        "protocol": protocol,
        "provenance": provenance,
        "grid": grid,
        "rows": rows,
        "decision": decision,
        "new_boosting_trainings": 0,
        "new_spectral_decompositions": 0,
        "interpretation_scope": (
            "Offline reproducibility and same-endpoint association after a frozen "
            "detrending transform; not a causal controller claim."
        ),
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_sha256 = file_sha256(output_path)
    print(json.dumps({
        "VERIFIED_INPUTS": provenance,
        "new_boosting_trainings": 0,
        "new_spectral_decompositions": 0,
    }), flush=True)
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": config["phase"],
        "mode": mode,
        "report": str(output_path.resolve()),
        "report_sha256": report_sha256,
        "rows": len(rows),
        "grid": grid,
        "new_boosting_trainings": 0,
        "new_spectral_decompositions": 0,
        "decision": decision,
    }}), flush=True)
