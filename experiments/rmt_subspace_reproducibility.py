"""Run the preregistered H5-1 subspace reproducibility experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.analyze_rmt_budget import _iteration_risks
from experiments.revision_forecast import load_trajectory
from experiments.revision_forecast_replay import verified_saved_records
from experiments.rmt_subspace_reproducibility_core import (
    balanced_partitions,
    bootstrap_median_interval,
    paired_distance_samples,
    subspace_distance,
    summarize_verdict,
)


def _window_bases(G, H, end, window, row_groups, maximum_rank, xp, batch_size):
    """Compute leading bases for many row groups with bounded device memory."""

    start = end - window
    if start < 0 or end > len(G):
        raise ValueError("window lies outside the trajectory")
    gradient = G[start:end]
    hessian = H[start:end]
    outputs = gradient.shape[2]
    bases = []
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
        _, vectors = xp.linalg.eigh(covariance)
        bases.append(vectors[:, :, ::-1][:, :, :maximum_rank])
        del covariance, vectors, groups
    return xp.concatenate(bases, axis=0)


def _risk_trace(path: str, budget: int) -> dict[int, float]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = [record for record in records
                if int(record["diagnostic_output_budget"]) == budget]
    trace = _iteration_risks(selected)
    return {int(iteration): float(risk) for iteration, risk in trace.items()
            if risk is not None and np.isfinite(risk)}


def _validate_sources(records, protocol):
    expected = {(dataset, seed) for dataset in protocol["datasets"]
                for seed in protocol["model_seeds"]}
    actual = {(str(record["dataset"]), int(record["model_seed"]))
              for record in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("saved trajectories do not match the frozen dataset/seed blocks")
    split_seeds = dict(zip(protocol["model_seeds"], protocol["split_seeds"]))
    for record in records:
        if int(record["split_seed"]) != split_seeds[int(record["model_seed"])]:
            raise ValueError("saved split seed differs from the frozen protocol")
        if int(record["trees_built"]) != protocol["expected_snapshots"]:
            raise ValueError("saved trajectory length differs from the frozen protocol")


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
    maximum_rank = max(protocol["fixed_ranks"])
    required_ends = sorted(set(protocol["endpoints"]) | {
        end - protocol["lag"] for end in protocol["endpoints"]
    })
    bases_by_end = {}
    for basis_end in required_ends:
        partition_bases = _window_bases(
            G, H, basis_end, protocol["window"], row_groups, maximum_rank, xp,
            protocol["device_batch_size"],
        )
        full_basis = _window_bases(
            G, H, basis_end, protocol["window"], full_rows, maximum_rank, xp, 1,
        )
        bases_by_end[basis_end] = (
            xp.asnumpy(partition_bases),
            xp.asnumpy(full_basis[0]),
        )
        del partition_bases, full_basis
        xp.get_default_memory_pool().free_all_blocks()
    output = []
    for end in protocol["endpoints"]:
        previous_end = end - protocol["lag"]
        current, current_full = bases_by_end[end]
        previous, previous_full = bases_by_end[previous_end]
        for rank in protocol["fixed_ranks"]:
            samples = paired_distance_samples(
                current[0::2, :, :rank], current[1::2, :, :rank],
                previous[0::2, :, :rank], previous[1::2, :, :rank],
            )
            interval_seed = (
                int(protocol["bootstrap_seed"])
                + 1000003 * int(record["model_seed"])
                + 1009 * end
                + rank
                + dataset_offset
            )
            interval = bootstrap_median_interval(
                samples["delta"], protocol["bootstrap_samples"], interval_seed,
                protocol["bootstrap_confidence"],
            )
            risk = risk_trace.get(end)
            if risk is None:
                raise ValueError(f"missing B={protocol['risk_budget']} risk at iteration {end}")
            row = {
                "dataset": str(record["dataset"]),
                "model_seed": int(record["model_seed"]),
                "split_seed": int(record["split_seed"]),
                "end": int(end),
                "previous_end": int(previous_end),
                "rank": int(rank),
                "partition_seed": int(partition_seed),
                "partition_repeats": int(protocol["partition_repeats"]),
                "median_noise": float(np.median(samples["noise"])),
                "noise_q05": float(np.quantile(samples["noise"], 0.05)),
                "noise_q95": float(np.quantile(samples["noise"], 0.95)),
                "median_overlap": float(1.0 - np.median(samples["noise"])),
                "median_temporal": float(np.median(samples["temporal"])),
                "median_delta": interval["median"],
                "delta_ci_low": interval["low"],
                "delta_ci_high": interval["high"],
                "full_temporal_distance": subspace_distance(
                    previous_full[:, :rank], current_full[:, :rank]
                ),
                "current_risk": float(risk),
            }
            output.append(row)
        primary = next(row for row in output
                       if row["end"] == end and row["rank"] == protocol["primary_rank"])
        print(json.dumps({"WINDOW_SUMMARY": {
            "dataset": record["dataset"],
            "model_seed": record["model_seed"],
            "end": end,
            "rank": protocol["primary_rank"],
            "median_overlap": primary["median_overlap"],
            "median_delta": primary["median_delta"],
            "delta_ci_low": primary["delta_ci_low"],
            "current_risk": primary["current_risk"],
        }}), flush=True)
    del G, H, bases_by_end
    xp.get_default_memory_pool().free_all_blocks()
    return {
        "dataset": str(record["dataset"]),
        "model_seed": int(record["model_seed"]),
        "split_seed": int(record["split_seed"]),
        "snapshot_rows": rows,
        "snapshots": len(manifest["iterations"]),
        "partition_seed": partition_seed,
    }, output


def run_reproducibility(config):
    import cupy as cp

    protocol = dict(config["rmt_subspace_reproducibility"])
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
    verdict = summarize_verdict(
        rows,
        primary_rank=protocol["primary_rank"],
        stable_overlap=protocol["stable_overlap"],
        minimum_stable_fraction=protocol["minimum_stable_fraction"],
        minimum_identifiable_fraction=protocol["minimum_identifiable_fraction"],
        minimum_risk_rho=protocol["minimum_risk_rho"],
    )
    report = {
        "hypothesis": "H5-1 reproducible spectral state",
        "protocol": protocol,
        "provenance": provenance,
        "sources": sources,
        "rows": rows,
        "decision": verdict,
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
        "windows": len(rows),
        "new_boosting_trainings": 0,
        "decision": verdict,
    }}), flush=True)
