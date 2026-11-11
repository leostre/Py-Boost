"""Pure update and scheduling rules for HASBoost optimization history."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SchedulerDecision:
    """Result of evaluating the stable-regime scheduler."""

    volatility: Optional[float]
    stable_regime: bool


def update_scale_history(previous_history: Any, current_scale: Any, alpha: float) -> Any:
    """Update an EMA of per-sample gradient or Hessian magnitudes."""

    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    if previous_history is None:
        previous_history = current_scale * 0

    return alpha * previous_history + (1 - alpha) * current_scale


def relative_volatility(
    current_scale: Any,
    previous_scale: Any,
    previous_history: Any,
    eps: float,
) -> float:
    """Compute ||s_t - s_(t-1)||_1 / (||h_(t-1)||_1 + eps)."""

    if eps <= 0:
        raise ValueError("eps must be positive")

    numerator = abs(current_scale - previous_scale).sum()
    denominator = abs(previous_history).sum() + eps
    return float(numerator / denominator)


def decide_stable_regime(
    iteration: int,
    volatility: Optional[float],
    was_stable: bool,
    enable_threshold: float,
    disable_threshold: float,
    min_warmup_rounds: int,
) -> SchedulerDecision:
    """Apply warm-up and optional hysteresis to one volatility observation."""

    if min_warmup_rounds < 1:
        raise ValueError("min_warmup_rounds must be positive")
    if enable_threshold < 0:
        raise ValueError("enable_threshold must be non-negative")
    if disable_threshold < enable_threshold:
        raise ValueError("disable_threshold must be at least enable_threshold")

    if iteration < min_warmup_rounds or volatility is None:
        return SchedulerDecision(volatility=volatility, stable_regime=False)

    threshold = disable_threshold if was_stable else enable_threshold
    return SchedulerDecision(
        volatility=volatility,
        stable_regime=volatility < threshold,
    )
