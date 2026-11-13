"""Run the preregistered H5-3 scalar spectral-state experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.revision_forecast import load_trajectory
from experiments.revision_forecast_replay import verified_saved_records
from experiments.rmt_scalar_reproducibility_core import (
    FEATURE_NAMES,
    relative_pair_errors,
    spectral_summaries,
    summarize_scalar_verdict,
)
from experiments.rmt_subspace_reproducibility import _risk_trace, _validate_sources
from experiments.rmt_subspace_reproducibility_core import balanced_partitions


def _window_eigenvalues(G, H, end, window, row_groups, xp, batch_size):
    """Compute descending spectra for many row groups with bounded device memory."""

    start = end - window
    if start < 0 or end > len(G):
        raise ValueError("window lies outside the trajectory")
    gradient = G[start:end]
    hessian = H[start:end]
    outputs = gradient.shape[2]
    spectra = []
    for batch_start in range(0, len(row_groups), batch_size):
        groups = xp.asarray(row_groups[batch_start:batch_start + batch_size])
        covariance = xp.zeros((len(groups), outputs, outputs), dtype=xp.float32)
        for raw in (gradient, hessian):
            selected = raw[:, groups, :].transpose(1, 0, 2, 3)
            if selected.shape[3] != outputs:
                selected = xp.broadcast_to(
                    selected,
                    (selected.shape[0], selected.shape[1], selected.shape[2], outputs),
                )
            selected = selected.astype(xp.float32, copy=False)
            selected = selected - selected.mean(axis=2, keepdims=True)
            scale = xp.sqrt(xp.mean(selected * selected, axis=(2, 3), keepdims=True))
            selected = selected / xp.maximum(scale, xp.float32(1e-12))
            flat = selected.reshape(selected.shape[0], -1, outputs)
            covariance += xp.matmul(flat.transpose(0, 2, 1), flat)
            del selected, flat
        covariance /= xp.float32(2 * window * row_groups.shape[1])
        values = xp.maximum(xp.linalg.eigvalsh(covariance)[:, ::-1], 0.0)
        spectra.append(values)
        del covariance, values, groups
    return xp.concatenate(spectra, axis=0)


def _feature_matrix(spectra, top_k):
    summaries = [spectral_summaries(values, top_k) for values in spectra]
    return {
        feature: np.asarray([summary[feature] for summary in summaries], dtype=float)
        for feature in FEATURE_NAMES
    }


def _analyze_trajectory(record, protocol, xp):
    manifest, gradient_host, hessian_host = load_trajectory(record["trajectory_path"])
    G = xp.asarray(gradient_host)
    H = xp.asarray(hessian_host)
    del gradient_host, hessian_host
    if G.ndim != 3 or H.ndim != 3 or G.shape[:2] != H.shape[:2]:
        raise ValueError("unexpected trajectory tensor shape")
    if len(G) != protocol["expected_snapshots"]:
        raise ValueError("trajectory snapshot count differs from the protocol")
    rows = int(G.shape[1])
    if rows != protocol["expected_rows"]:
        raise ValueError("saved row count differs from the protocol")
    dataset_offset = sum(ord(char) for char in str(record["dataset"]))
    partition_seed = (
        int(protocol["partition_seed"])
        + 1009 * int(record["model_seed"])
        + dataset_offset
    )
    partitions = balanced_partitions(rows, protocol["partition_repeats"], partition_seed)
    row_groups = partitions.reshape(-1, rows // 2)
    full_rows = np.arange(rows, dtype=np.int64)[None, :]
    risk_trace = _risk_trace(record["split_diagnostics_path"], protocol["risk_budget"])
    output = []
    for end in protocol["endpoints"]:
        partition_spectra = _window_eigenvalues(
            G, H, end, protocol["window"], row_groups, xp,
            protocol["device_batch_size"],
        )
        full_spectrum = _window_eigenvalues(
            G, H, end, protocol["window"], full_rows, xp, 1,
        )
        partition_features = _feature_matrix(
            xp.asnumpy(partition_spectra), protocol["top_k"]
        )
        full_features = spectral_summaries(
            xp.asnumpy(full_spectrum[0]), protocol["top_k"]
        )
        risk = risk_trace.get(end)
        if risk is None:
            raise ValueError(f"missing B={protocol['risk_budget']} risk at iteration {end}")
        for feature in FEATURE_NAMES:
            values = partition_features[feature]
            left, right = values[0::2], values[1::2]
            errors = relative_pair_errors(left, right)
            output.append({
                "dataset": str(record["dataset"]),
                "model_seed": int(record["model_seed"]),
                "split_seed": int(record["split_seed"]),
                "end": int(end),
                "feature": feature,
                "partition_seed": int(partition_seed),
                "partition_repeats": int(protocol["partition_repeats"]),
                "partition_a_values": left.tolist(),
                "partition_b_values": right.tolist(),
                "median_relative_error": float(np.median(errors)),
                "relative_error_q95": float(np.quantile(errors, 0.95)),
                "full_value": float(full_features[feature]),
                "current_risk": float(risk),
            })
        print(json.dumps({"WINDOW_SUMMARY": {
            "dataset": record["dataset"],
            "model_seed": record["model_seed"],
            "end": end,
            "current_risk": float(risk),
            **full_features,
        }}), flush=True)
        del partition_spectra, full_spectrum
        xp.get_default_memory_pool().free_all_blocks()
    del G, H
    xp.get_default_memory_pool().free_all_blocks()
    return {
        "dataset": str(record["dataset"]),
        "model_seed": int(record["model_seed"]),
        "split_seed": int(record["split_seed"]),
        "snapshot_rows": rows,
        "snapshots": len(manifest["iterations"]),
        "partition_seed": partition_seed,
    }, output


def run_scalar_reproducibility(config):
    import cupy as cp

    protocol = dict(config["rmt_scalar_reproducibility"])
    records, provenance = verified_saved_records(
        protocol["source_receipt"], protocol["source_receipt_sha256"]
    )
    _validate_sources(records, protocol)
    print(json.dumps({"VERIFIED_INPUTS": provenance, "new_boosting_trainings": 0}),
          flush=True)
    sources = []
    rows = []
    for record in sorted(records, key=lambda item: (item["dataset"], item["model_seed"])):
        source, trajectory_rows = _analyze_trajectory(record, protocol, cp)
        sources.append(source)
        rows.extend(trajectory_rows)
    decision = summarize_scalar_verdict(
        rows,
        primary_feature=protocol["primary_feature"],
        secondary_features=protocol["secondary_features"],
        maximum_relative_error=protocol["maximum_relative_error"],
        minimum_stable_fraction=protocol["minimum_stable_fraction"],
        minimum_split_rank_rho=protocol["minimum_split_rank_rho"],
        minimum_risk_rho=protocol["minimum_risk_rho"],
    )
    report = {
        "hypothesis": "H5-3 robust scalar spectral state",
        "protocol": protocol,
        "provenance": provenance,
        "sources": sources,
        "rows": rows,
        "decision": decision,
        "new_boosting_trainings": 0,
        "interpretation_scope": (
            "Offline reproducibility and same-iteration association only; "
            "not a causal forecast or a runtime controller claim."
        ),
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": config["phase"],
        "report": str(output_path.resolve()),
        "verified_artifacts": provenance["verified_artifacts"],
        "trajectories": len(sources),
        "rows": len(rows),
        "new_boosting_trainings": 0,
        "decision": decision,
    }}), flush=True)
