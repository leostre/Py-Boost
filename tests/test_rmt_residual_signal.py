"""Source-verification tests for residual-signal analyses."""

import json

from experiments.revision_forecast_replay import file_sha256
from experiments.rmt_residual_signal import _verified_source


def test_verified_source_accepts_configurable_expanded_report(tmp_path):
    datasets = ["helena"]
    model_seeds = [204, 214]
    split_seeds = [132, 142]
    endpoints = [20, 24, 28]
    primary = "effective_rank_fraction"
    secondary = ["tail_energy_after_top4", "inverse_relative_gap4"]
    rows = []
    for model_seed, split_seed in zip(model_seeds, split_seeds):
        for end in endpoints:
            for feature in [primary, *secondary]:
                rows.append({
                    "dataset": "helena",
                    "model_seed": model_seed,
                    "split_seed": split_seed,
                    "end": end,
                    "feature": feature,
                    "partition_repeats": 2,
                    "partition_a_values": [0.1, 0.2],
                    "partition_b_values": [0.1, 0.2],
                    "current_risk": 0.3,
                })
    report = {
        "hypothesis": "expanded source",
        "protocol": {
            "experiment_id": "source-id",
            "datasets": datasets,
            "model_seeds": model_seeds,
            "split_seeds": split_seeds,
            "endpoints": endpoints,
            "partition_repeats": 2,
            "primary_feature": primary,
            "secondary_features": secondary,
        },
        "decision": {"verdict": "rejected"},
        "scalar_rows": rows,
    }
    source_path = tmp_path / "report.json"
    source_path.write_text(json.dumps(report), encoding="utf-8")
    protocol = {
        "source_report": str(source_path),
        "source_report_sha256": file_sha256(source_path),
        "source_hypothesis": "expanded source",
        "source_rows_key": "scalar_rows",
        "source_verdict": "rejected",
        "datasets": datasets,
        "model_seeds": model_seeds,
        "split_seeds": split_seeds,
        "source_endpoints": endpoints,
        "partition_repeats": 2,
        "primary_feature": primary,
        "secondary_features": secondary,
    }

    verified, provenance = _verified_source(protocol)

    assert verified == report
    assert provenance["source_experiment_id"] == "source-id"
    assert provenance["source_rows_key"] == "scalar_rows"
    assert provenance["verified_rows"] == 18
