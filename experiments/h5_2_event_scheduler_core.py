"""Causal, fixed-audit-budget replay for the H5-2 scheduling hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class AuditProtocol:
    trees: int = 100
    warmup_audits: tuple[int, int] = (5, 10)
    block_size: int = 10
    safe_threshold: float = 0.10
    margin_fraction: float = 0.80
    margin_delay: int = 4

    def __post_init__(self) -> None:
        first, second = self.warmup_audits
        if not 1 <= first < second < self.trees:
            raise ValueError("warmup audits must be distinct and precede evaluation")
        if self.block_size < 2 or self.margin_delay < 0 or self.margin_delay >= self.block_size:
            raise ValueError("invalid audit block or margin delay")
        if not 0 < self.safe_threshold or not 0 <= self.margin_fraction <= 1:
            raise ValueError("invalid risk thresholds")


def scheduled_blocks(protocol: AuditProtocol) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + protocol.block_size - 1, protocol.trees))
        for start in range(protocol.warmup_audits[-1] + 1,
                           protocol.trees + 1, protocol.block_size)
    )


def volatility_changes(volatility: Mapping[int, float | None],
                       start: int, end: int) -> tuple[float, ...]:
    changes = []
    for iteration in range(max(3, start), end + 1):
        current = volatility.get(iteration - 1)
        previous = volatility.get(iteration - 2)
        if current is not None and previous is not None:
            if not isfinite(current) or not isfinite(previous):
                raise ValueError("volatility must be finite")
            changes.append(abs(current - previous))
    return tuple(changes)


def empirical_quantile(values: tuple[float, ...], level: float) -> float:
    if not values or not 0 < level < 1:
        raise ValueError("nonempty values and a quantile in (0, 1) are required")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(level * len(ordered)))
    return float(ordered[index])


def replay_schedule(
    risks: Mapping[int, float],
    volatility: Mapping[int, float | None],
    protocol: AuditProtocol,
    mode: str,
    alarm_threshold: float | None = None,
) -> dict:
    """Replay one policy. Risk is read only on audits or after choosing an action."""
    if mode not in {"periodic", "event"}:
        raise ValueError("unknown audit mode")
    if mode == "event" and (alarm_threshold is None or alarm_threshold < 0):
        raise ValueError("event scheduling requires a nonnegative alarm threshold")
    if set(risks) != set(range(1, protocol.trees + 1)):
        raise ValueError("every tree must have an oracle risk for offline scoring")
    if any(not isfinite(risk) or risk < 0 for risk in risks.values()):
        raise ValueError("oracle risks must be finite and nonnegative")

    blocks = scheduled_blocks(protocol)
    audit_iterations = []
    rows = []
    last_observed_risk = None
    consecutive_safe = 0
    confirmed = False
    for iteration in range(1, protocol.trees + 1):
        audit = iteration in protocol.warmup_audits
        reason = "warmup" if iteration <= protocol.warmup_audits[-1] else "retain"
        if iteration > protocol.warmup_audits[-1]:
            block_start, block_end = next(
                block for block in blocks if block[0] <= iteration <= block[1]
            )
            already_audited = any(block_start <= past <= block_end
                                  for past in audit_iterations)
            if not already_audited:
                if iteration == block_end:
                    audit, reason = True, "deadline"
                elif mode == "event":
                    previous = volatility.get(iteration - 1)
                    earlier = volatility.get(iteration - 2)
                    change = (abs(previous - earlier)
                              if previous is not None and earlier is not None
                              else None)
                    if change is not None and not isfinite(change):
                        raise ValueError("volatility change must be finite")
                    if change is not None and change >= alarm_threshold:
                        audit, reason = True, "volatility_alarm"
                    elif (last_observed_risk is not None
                          and last_observed_risk >= protocol.safe_threshold
                          * protocol.margin_fraction
                          and iteration >= block_start + protocol.margin_delay):
                        audit, reason = True, "risk_margin"

        # An audit tree uses full statistics; its candidate-budget risk is shadow-only.
        use_budget = confirmed and not audit and iteration > protocol.warmup_audits[-1]
        oracle_risk = float(risks[iteration])
        if audit:
            audit_iterations.append(iteration)
            last_observed_risk = oracle_risk
            consecutive_safe = consecutive_safe + 1 if oracle_risk <= protocol.safe_threshold else 0
            confirmed = consecutive_safe >= 2
        rows.append({
            "iteration": iteration,
            "audit": audit,
            "reason": reason,
            "use_budget": use_budget,
            "observed_risk": oracle_risk if audit else None,
            "oracle_risk": oracle_risk,
            "unsafe_application": use_budget and oracle_risk > protocol.safe_threshold,
        })

    expected_audits = len(protocol.warmup_audits) + len(blocks)
    if len(audit_iterations) != expected_audits:
        raise AssertionError("audit count differs from the fixed block budget")
    unsafe = [row for row in rows if row["unsafe_application"]]
    reduced = [row for row in rows if row["use_budget"]]
    safe = [row for row in reduced if not row["unsafe_application"]]
    return {
        "mode": mode,
        "audits": len(audit_iterations),
        "audit_iterations": audit_iterations,
        "first_budget_iteration": reduced[0]["iteration"] if reduced else None,
        "reduced_trees": len(reduced),
        "safe_reduced_trees": len(safe),
        "unsafe_applied_trees": len(unsafe),
        "excess_risk_sum": sum(row["oracle_risk"] - protocol.safe_threshold
                               for row in unsafe),
        "rows": rows,
    }
