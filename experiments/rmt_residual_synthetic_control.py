"""Run the preregistered H5-3C end-to-end synthetic instrumentation control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.rmt_residual_signal_core import (
    cross_trajectory_residual_rows,
    local_linear_innovation_rows,
    summarize_residual_verdict,
)
from experiments.rmt_residual_synthetic_control_core import (
    direction_reversed_risk,
    latent_complexity,
    output_spectra,
    summarize_control_gate,
    window_risk,
)
from experiments.rmt_scalar_reproducibility import _window_eigenvalues
from experiments.rmt_scalar_reproducibility_core import (
    FEATURE_NAMES,
    relative_pair_errors,
    spectral_summaries,
)
from experiments.rmt_subspace_reproducibility_core import balanced_partitions


def _statistics(protocol, model_seed, trajectory_index):
    complexity = latent_complexity(
        protocol["steps"],
        trajectory_index,
        len(protocol["model_seeds"]),
        protocol["latent_intercept"],
        protocol["latent_slope"],
        protocol["latent_amplitude"],
        protocol["latent_period"],
    )
    spectra = output_spectra(
        complexity, protocol["outputs"], protocol["spectral_concentration"]
    )
    scales = np.sqrt(spectra).astype(np.float32)[:, None, :]
    rng = np.random.default_rng(model_seed)
    shape = (protocol["steps"], protocol["rows"], protocol["outputs"])
    gradient = rng.normal(size=shape).astype(np.float32) * scales
    hessian = np.abs(rng.normal(size=shape)).astype(np.float32) * scales
    return complexity, gradient, hessian


def _feature_matrix(spectra, top_k):
    summaries = [spectral_summaries(values, top_k) for values in spectra]
    return {
        feature: np.asarray([summary[feature] for summary in summaries])
        for feature in FEATURE_NAMES
    }


def _trajectory_rows(protocol, model_seed, split_seed, trajectory_index):
    complexity, gradient, hessian = _statistics(
        protocol, model_seed, trajectory_index
    )
    partition_seed = protocol["partition_seed"] + 1009 * model_seed
    partitions = balanced_partitions(
        protocol["rows"], protocol["partition_repeats"], partition_seed
    )
    row_groups = partitions.reshape(-1, protocol["rows"] // 2)
    full_rows = np.arange(protocol["rows"], dtype=np.int64)[None, :]
    rows = []
    for end in protocol["endpoints"]:
        partition_spectra = _window_eigenvalues(
            gradient,
            hessian,
            end,
            protocol["window"],
            row_groups,
            np,
            protocol["batch_size"],
        )
        full_spectrum = _window_eigenvalues(
            gradient, hessian, end, protocol["window"], full_rows, np, 1
        )[0]
        partition_features = _feature_matrix(partition_spectra, protocol["top_k"])
        full_features = spectral_summaries(full_spectrum, protocol["top_k"])
        coupled_risk = window_risk(complexity, end, protocol["window"])
        reversed_risk = direction_reversed_risk(coupled_risk)
        for feature in FEATURE_NAMES:
            values = partition_features[feature]
            left, right = values[0::2], values[1::2]
            errors = relative_pair_errors(left, right)
            rows.append({
                "dataset": protocol["dataset"],
                "model_seed": model_seed,
                "split_seed": split_seed,
                "end": end,
                "feature": feature,
                "partition_seed": partition_seed,
                "partition_repeats": protocol["partition_repeats"],
                "partition_a_values": left.tolist(),
                "partition_b_values": right.tolist(),
                "median_relative_error": float(np.median(errors)),
                "relative_error_q95": float(np.quantile(errors, 0.95)),
                "full_value": float(full_features[feature]),
                "current_risk": coupled_risk,
                "direction_reversed_risk": reversed_risk,
            })
    return rows


def _analyze_arm(source_rows, protocol, risk_field):
    rows = []
    for source in source_rows:
        row = dict(source)
        row["current_risk"] = float(source[risk_field])
        rows.append(row)
    transformed = {
        "cross_trajectory": cross_trajectory_residual_rows(rows),
        "local_linear": local_linear_innovation_rows(rows),
    }
    output = {}
    for name, signal_rows in transformed.items():
        output[name] = {
            "rows": signal_rows,
            "decision": summarize_residual_verdict(
                signal_rows,
                primary_feature=protocol["primary_feature"],
                secondary_features=protocol["secondary_features"],
                minimum_ccc=protocol["minimum_ccc"],
                minimum_sign_agreement=protocol["minimum_sign_agreement"],
                minimum_risk_rho=protocol["minimum_risk_rho"],
            ),
        }
    return output


def run_synthetic_control(config):
    protocol = dict(config["rmt_residual_synthetic_control"])
    if tuple(protocol["features"]) != FEATURE_NAMES:
        raise ValueError("configured synthetic features differ from H5-3")
    expected_endpoints = list(range(
        protocol["first_endpoint"], protocol["steps"] + 1, protocol["stride"]
    ))
    if protocol["endpoints"] != expected_endpoints:
        raise ValueError("synthetic endpoints differ from the frozen protocol")
    source_rows = []
    for index, (model_seed, split_seed) in enumerate(zip(
        protocol["model_seeds"], protocol["split_seeds"]
    )):
        trajectory_rows = _trajectory_rows(
            protocol, model_seed, split_seed, index
        )
        source_rows.extend(trajectory_rows)
        print(json.dumps({"TRAJECTORY_SUMMARY": {
            "dataset": protocol["dataset"],
            "model_seed": model_seed,
            "split_seed": split_seed,
            "rows": len(trajectory_rows),
        }}), flush=True)
    analyses = {
        "coupled": _analyze_arm(source_rows, protocol, "current_risk"),
        "direction_reversed": _analyze_arm(
            source_rows, protocol, "direction_reversed_risk"
        ),
    }
    decision = summarize_control_gate(
        analyses, protocol["maximum_reversed_rho"]
    )
    spectral_decompositions = (
        len(protocol["model_seeds"])
        * len(protocol["endpoints"])
        * (2 * protocol["partition_repeats"] + 1)
    )
    if spectral_decompositions != protocol["synthetic_spectral_decompositions"]:
        raise ValueError("configured spectral decomposition count is inconsistent")
    report = {
        "hypothesis": "H5-3C synthetic residual-signal positive control",
        "protocol": protocol,
        "source_rows": source_rows,
        "analyses": analyses,
        "decision": decision,
        "new_boosting_fits": 0,
        "synthetic_spectral_decompositions": spectral_decompositions,
        "interpretation_scope": (
            "Instrumentation sensitivity and direction-specificity only; "
            "not evidence that real optimization trajectories contain the signal."
        ),
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": config["phase"],
        "report": str(output_path.resolve()),
        "source_rows": len(source_rows),
        "new_boosting_fits": 0,
        "synthetic_spectral_decompositions": spectral_decompositions,
        "decision": decision,
        "arm_verdicts": {
            arm: {
                transform: result["decision"]["verdict"]
                for transform, result in transforms.items()
            }
            for arm, transforms in analyses.items()
        },
    }}), flush=True)
