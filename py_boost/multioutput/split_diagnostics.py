"""Pure split-risk metrics used by HASBoost diagnostic schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SplitRiskSummary:
    """Tree-level summary of node-wise normalized gain regret."""

    node_count: int
    maximum: Optional[float]
    mean: Optional[float]
    weighted_quantile: Optional[float]
    spurious_split_count: int = 0


def action_gain_regret(full_best_gain: float, candidate_full_gain: float,
                       diagnostic_split_available: bool,
                       full_split_available: bool) -> Optional[float]:
    """Score the action actually taken, including the zero-gain no-split action."""
    if not full_split_available:
        return 1.0 if diagnostic_split_available else None
    action_gain = candidate_full_gain if diagnostic_split_available else 0.0
    return normalized_gain_regret(full_best_gain, action_gain)


def normalized_gain_regret(
    full_best_gain: float,
    selected_gain_on_full_statistics: float,
    eps: float = 1e-6,
) -> float:
    """Return max(0, (best - selected) / (abs(best) + eps))."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if not isfinite(full_best_gain) or not isfinite(selected_gain_on_full_statistics):
        raise ValueError("split gains must be finite")
    return max(
        0.0,
        (full_best_gain - selected_gain_on_full_statistics)
        / (abs(full_best_gain) + eps),
    )


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> Optional[float]:
    """Return a deterministic left-continuous weighted quantile."""

    if not 0 <= quantile <= 1:
        raise ValueError("quantile must satisfy 0 <= quantile <= 1")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        return None
    if any(not isfinite(value) for value in values):
        raise ValueError("values must be finite")
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be finite and non-negative")

    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0:
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        return float(ordered[index][0])

    target = quantile * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def summarize_split_risks(
    records: Iterable[Mapping[str, object]],
    quantile: float = 0.95,
) -> SplitRiskSummary:
    """Aggregate node diagnostics with full gains used as importance weights."""

    materialized: Tuple[Mapping[str, object], ...] = tuple(
        record for record in records if record.get("regret") is not None
    )
    if not materialized:
        return SplitRiskSummary(0, None, None, None)

    regrets = tuple(float(record["regret"]) for record in materialized)
    weights = tuple(max(0.0, float(record["full_best_gain"])) for record in materialized)
    return SplitRiskSummary(
        node_count=len(regrets),
        maximum=max(regrets),
        mean=sum(regrets) / len(regrets),
        weighted_quantile=weighted_quantile(regrets, weights, quantile),
        spurious_split_count=sum(bool(record.get("spurious_split")) for record in materialized),
    )
