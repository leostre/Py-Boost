"""Contract tests for equal-cost, causal H5-2 audit scheduling."""

from hypothesis import given, strategies as st

from experiments.h5_2_event_scheduler_core import AuditProtocol, replay_schedule, scheduled_blocks


def _series(values):
    return {index: value for index, value in enumerate(values, 1)}


@given(
    risks=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False),
                   min_size=31, max_size=31),
    volatilities=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False),
                          min_size=31, max_size=31),
)
def test_every_policy_spends_exactly_one_audit_per_block(risks, volatilities):
    protocol = AuditProtocol(trees=31, warmup_audits=(2, 4), block_size=5)
    for mode in ("periodic", "event"):
        result = replay_schedule(_series(risks), _series(volatilities),
                                 protocol, mode, alarm_threshold=0.2)
        assert result["audit_iterations"][:2] == [2, 4]
        for start, end in scheduled_blocks(protocol):
            assert sum(start <= time <= end for time in result["audit_iterations"]) == 1
        assert all(not row["use_budget"] for row in result["rows"][:4])
        assert all(not row["use_budget"] for row in result["rows"] if row["audit"])


def test_event_alarm_uses_only_prior_volatility_and_avoids_detected_tree():
    protocol = AuditProtocol(trees=30)
    risks = {iteration: 0.05 for iteration in range(1, 31)}
    risks[13] = 0.9
    volatility = {iteration: 0.01 for iteration in range(1, 31)}
    volatility[12] = 0.5
    event = replay_schedule(risks, volatility, protocol, "event", 0.2)
    periodic = replay_schedule(risks, volatility, protocol, "periodic")
    assert event["audit_iterations"][:3] == [5, 10, 13]
    assert event["rows"][12]["reason"] == "volatility_alarm"
    assert event["unsafe_applied_trees"] == 0
    assert periodic["unsafe_applied_trees"] == 1
    assert event["audits"] == periodic["audits"]


def test_one_safe_initial_audit_does_not_confirm_budget():
    protocol = AuditProtocol(trees=30)
    risks = {iteration: 0.05 for iteration in range(1, 31)}
    risks[5] = 0.2
    volatility = {iteration: 0.01 for iteration in range(1, 31)}
    result = replay_schedule(risks, volatility, protocol, "periodic")
    assert result["first_budget_iteration"] == 21
    assert all(not row["use_budget"] for row in result["rows"][:20])


def test_future_risk_cannot_change_earlier_schedule_or_actions():
    protocol = AuditProtocol(trees=30)
    risks = {iteration: 0.05 for iteration in range(1, 31)}
    volatility = {iteration: 0.01 for iteration in range(1, 31)}
    original = replay_schedule(risks, volatility, protocol, "event", 0.2)
    altered = replay_schedule({**risks, 19: 0.9}, volatility, protocol, "event", 0.2)
    for left, right in zip(original["rows"][:18], altered["rows"][:18]):
        assert left == right
