import numpy as np

from experiments.rmt_residual_signal_core import (
    cross_trajectory_residual_rows,
    lin_concordance,
    local_linear_innovation_rows,
    summarize_residual_verdict,
)


FEATURES = (
    "effective_rank_fraction",
    "tail_energy_after_top4",
    "inverse_relative_gap4",
)


def _source_row(seed, end, value, risk, feature="effective_rank_fraction"):
    return {
        "dataset": "helena",
        "model_seed": seed,
        "split_seed": seed + 1,
        "end": end,
        "feature": feature,
        "partition_repeats": 2,
        "partition_a_values": [value + 0.5, value - 0.5],
        "partition_b_values": [value + 0.25, value - 0.25],
        "full_value": value,
        "current_risk": risk,
    }


def test_lin_concordance_distinguishes_agreement_shift_and_reversal():
    values = np.array([-2.0, -1.0, 1.0, 3.0])

    assert np.isclose(lin_concordance(values, values), 1.0)
    assert lin_concordance(values, values + 2.0) < 1.0
    assert lin_concordance(values, -values) < 0.0


def test_cross_trajectory_residual_uses_only_other_trajectories():
    rows = [
        _source_row(1, 20, 10.0, 1.0),
        _source_row(2, 20, 12.0, 2.0),
        _source_row(3, 20, 14.0, 4.0),
    ]

    transformed = cross_trajectory_residual_rows(rows)
    first = next(row for row in transformed if row["model_seed"] == 1)

    assert first["expected_feature"] == 13.0
    assert first["feature_signal"] == -3.0
    assert first["expected_risk"] == 3.0
    assert first["risk_signal"] == -2.0
    np.testing.assert_allclose(first["partition_a_signal"], [-2.5, -3.5])


def test_local_linear_innovation_removes_linear_course_and_keeps_impulse():
    linear = [
        _source_row(1, end, 2.0 * index + 1.0, 3.0 * index)
        for index, end in enumerate((20, 24, 28, 32))
    ]
    linear_result = local_linear_innovation_rows(linear)

    assert [row["end"] for row in linear_result] == [28, 32]
    assert all(np.isclose(row["feature_signal"], 0.0) for row in linear_result)
    assert all(np.isclose(row["risk_signal"], 0.0) for row in linear_result)

    impulse = list(linear)
    impulse[-1] = _source_row(1, 32, 10.0, 9.0)
    impulse_result = local_linear_innovation_rows(impulse)
    assert impulse_result[-1]["feature_signal"] == 3.0


def _verdict_rows(primary_reversed=False, secondary_actionable=False, unstable=False):
    rows = []
    for dataset in ("helena", "mediamill"):
        for seed in (1, 2, 3):
            for index, end in enumerate((28, 32, 36, 40, 44)):
                risk = float(index)
                for feature in FEATURES:
                    reversed_signal = feature == FEATURES[0] and primary_reversed
                    if feature != FEATURES[0] and not secondary_actionable:
                        reversed_signal = True
                    value = float(4 - index if reversed_signal else index)
                    left = [value + repeat * 0.01 for repeat in range(4)]
                    right = [-value - repeat * 0.01 for repeat in range(4)] if unstable else left
                    rows.append({
                        "dataset": dataset,
                        "model_seed": seed,
                        "end": end,
                        "feature": feature,
                        "partition_repeats": 4,
                        "partition_a_signal": left,
                        "partition_b_signal": right,
                        "feature_signal": value,
                        "risk_signal": risk,
                    })
    return rows


def _decision(rows):
    return summarize_residual_verdict(
        rows,
        primary_feature=FEATURES[0],
        secondary_features=FEATURES[1:],
        minimum_ccc=0.80,
        minimum_sign_agreement=0.80,
        minimum_risk_rho=0.20,
    )


def test_verdict_preserves_primary_secondary_and_failure_states():
    assert _decision(_verdict_rows())["verdict"] == "confirmed_actionable"

    secondary = _decision(_verdict_rows(
        primary_reversed=True, secondary_actionable=True
    ))
    assert secondary["verdict"] == "secondary_candidate"
    assert secondary["primary_actionable"] is False
    assert secondary["secondary_actionable"] == list(FEATURES[1:])

    descriptive = _decision(_verdict_rows(primary_reversed=True))
    rejected = _decision(_verdict_rows(primary_reversed=True, unstable=True))
    assert descriptive["verdict"] == "descriptive_only"
    assert rejected["verdict"] == "rejected"
