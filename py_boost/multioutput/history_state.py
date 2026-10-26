"""Pure update and scheduling rules for HASBoost optimization history."""

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class SchedulerDecision:
    """Result of evaluating the stable-regime scheduler."""

    volatility: Optional[float]
    stable_regime: bool
    score: Optional[float] = None
    observations: int = 0
    reason: str = "current_threshold"
    budget_transitioned: bool = False
    budget_dwell_remaining: int = 0
    budget_cooldown_remaining: int = 0


@dataclass(frozen=True)
class RobustSchedulerConfig:
    """Configuration for a robust-window scheduler with patience and cooldown."""

    enable_threshold: float
    disable_threshold: float
    min_warmup_rounds: int = 5
    window_size: int = 8
    min_observations: int = 4
    patience_on: int = 2
    patience_off: int = 1
    robust_k: float = 1.0
    cooldown_rounds: int = 2

    def __post_init__(self) -> None:
        if self.enable_threshold < 0:
            raise ValueError("enable_threshold must be non-negative")
        if self.disable_threshold < self.enable_threshold:
            raise ValueError("disable_threshold must be at least enable_threshold")
        if self.min_warmup_rounds < 1:
            raise ValueError("min_warmup_rounds must be positive")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= self.min_observations <= self.window_size:
            raise ValueError("min_observations must be within the scheduler window")
        if self.patience_on < 1 or self.patience_off < 1:
            raise ValueError("scheduler patience values must be positive")
        if self.robust_k < 0:
            raise ValueError("robust_k must be non-negative")
        if self.cooldown_rounds < 0:
            raise ValueError("cooldown_rounds must be non-negative")


@dataclass(frozen=True)
class RobustSchedulerState:
    """Immutable state consumed and produced by the robust scheduler reducer."""

    stable_regime: bool = False
    observations: Tuple[float, ...] = ()
    consecutive_safe: int = 0
    consecutive_unsafe: int = 0
    cooldown_remaining: int = 0


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Validated levels and risk thresholds for output-budget control."""

    budgets: Tuple[int, ...]
    safe_threshold: float = 0.10
    unsafe_threshold: float = 0.15
    min_warmup_rounds: int = 5
    patience_down: int = 2
    min_dwell_rounds: int = 5
    cooldown_after_increase_rounds: int = 10

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("adaptive budgets must not be empty")
        if any(budget < 1 for budget in self.budgets):
            raise ValueError("adaptive budgets must be positive")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("adaptive budgets must be unique and increasing")
        if self.safe_threshold < 0:
            raise ValueError("safe_threshold must be non-negative")
        if self.unsafe_threshold < self.safe_threshold:
            raise ValueError("unsafe_threshold must be at least safe_threshold")
        if self.min_warmup_rounds < 1:
            raise ValueError("min_warmup_rounds must be positive")
        if self.patience_down < 1:
            raise ValueError("patience_down must be positive")
        if self.min_dwell_rounds < 0:
            raise ValueError("min_dwell_rounds must be non-negative")
        if self.cooldown_after_increase_rounds < 0:
            raise ValueError("cooldown_after_increase_rounds must be non-negative")


@dataclass(frozen=True)
class AdaptiveBudgetState:
    """Immutable output-budget state retained between boosting iterations."""

    current_budget: int
    consecutive_safe: int = 0
    observations: int = 0
    last_transition_iteration: int = 0
    cooldown_until_iteration: int = 0


@dataclass(frozen=True)
class AdaptiveBudgetDecision:
    """Budget to apply and budget to inspect at the current iteration."""

    output_budget: int
    probe_budget: int
    sketching_enabled: bool
    observation: Optional[float]
    observations: int
    reason: str
    transitioned: bool = False
    dwell_remaining: int = 0
    cooldown_remaining: int = 0


def resolve_budget_levels(levels: Tuple[int, ...], n_outputs: int) -> Tuple[int, ...]:
    """Clip requested levels to a dataset and always include the full budget."""

    if n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    if not levels or any(level < 1 for level in levels):
        raise ValueError("requested budget levels must be positive")
    return tuple(sorted({min(int(level), n_outputs) for level in levels} | {n_outputs}))


def advance_adaptive_budget(
    iteration: int,
    observation: Optional[float],
    state: AdaptiveBudgetState,
    config: AdaptiveBudgetConfig,
    probe_observation: Optional[float] = None,
) -> Tuple[AdaptiveBudgetState, AdaptiveBudgetDecision]:
    """Advance output-budget control without reading model or callback state."""

    if iteration < 1:
        raise ValueError("iteration must be positive")
    if state.current_budget not in config.budgets:
        raise ValueError("current budget must belong to configured budgets")
    if state.consecutive_safe < 0 or state.observations < 0:
        raise ValueError("adaptive budget counters must be non-negative")
    if state.last_transition_iteration < 0 or state.cooldown_until_iteration < 0:
        raise ValueError("adaptive budget iteration markers must be non-negative")
    if observation is not None and not 0 <= observation < float("inf"):
        raise ValueError("budget observations must be finite and non-negative")
    if probe_observation is not None and not 0 <= probe_observation < float("inf"):
        raise ValueError("probe observations must be finite and non-negative")

    full_budget = config.budgets[-1]
    current_index = config.budgets.index(state.current_budget)
    safe_count = state.consecutive_safe
    observations = state.observations
    reason = "retain_without_observation"

    if iteration <= config.min_warmup_rounds:
        next_state = AdaptiveBudgetState(
            current_budget=full_budget,
            consecutive_safe=0,
            observations=observations,
            last_transition_iteration=state.last_transition_iteration,
            cooldown_until_iteration=state.cooldown_until_iteration,
        )
        return next_state, AdaptiveBudgetDecision(
            output_budget=full_budget,
            probe_budget=config.budgets[-2] if len(config.budgets) > 1 else full_budget,
            sketching_enabled=False,
            observation=observation,
            observations=observations,
            reason="warmup",
            dwell_remaining=max(
                0,
                config.min_dwell_rounds
                - (iteration - state.last_transition_iteration),
            ),
            cooldown_remaining=max(
                0,
                state.cooldown_until_iteration - iteration,
            ),
        )

    next_budget = state.current_budget
    last_transition = state.last_transition_iteration
    cooldown_until = state.cooldown_until_iteration
    transitioned = False
    if observation is not None or probe_observation is not None:
        observations += 1
        if observation is not None and observation >= config.unsafe_threshold:
            next_budget = config.budgets[min(current_index + 1, len(config.budgets) - 1)]
            safe_count = 0
            transitioned = next_budget != state.current_budget
            if transitioned:
                last_transition = iteration
                cooldown_until = iteration + config.cooldown_after_increase_rounds
                reason = "risk_above_unsafe_threshold"
            else:
                reason = "unsafe_at_full_budget"
        elif probe_observation is not None and probe_observation <= config.safe_threshold:
            safe_count += 1
            if safe_count >= config.patience_down and current_index > 0:
                dwell_ready = (
                    iteration - state.last_transition_iteration
                    >= config.min_dwell_rounds
                )
                cooldown_ready = iteration >= state.cooldown_until_iteration
                if not cooldown_ready:
                    safe_count = min(safe_count, config.patience_down)
                    reason = "waiting_for_increase_cooldown"
                elif not dwell_ready:
                    safe_count = min(safe_count, config.patience_down)
                    reason = "waiting_for_minimum_dwell"
                else:
                    next_budget = config.budgets[current_index - 1]
                    safe_count = 0
                    last_transition = iteration
                    transitioned = True
                    reason = "safe_patience_satisfied"
            elif current_index == 0:
                safe_count = 0
                reason = "safe_at_minimum_budget"
            else:
                reason = "waiting_for_safe_patience"
        else:
            safe_count = 0
            reason = ("waiting_for_probe_observation" if probe_observation is None
                      else "probe_not_safe")

    next_state = AdaptiveBudgetState(
        current_budget=next_budget,
        consecutive_safe=safe_count,
        observations=observations,
        last_transition_iteration=last_transition,
        cooldown_until_iteration=cooldown_until,
    )
    next_index = config.budgets.index(next_budget)
    probe_budget = (
        config.budgets[next_index - 1]
        if next_index > 0
        else next_budget
    )
    return next_state, AdaptiveBudgetDecision(
        output_budget=next_budget,
        probe_budget=probe_budget,
        sketching_enabled=next_budget < full_budget,
        observation=observation,
        observations=observations,
        reason=reason,
        transitioned=transitioned,
        dwell_remaining=max(
            0,
            config.min_dwell_rounds - (iteration - last_transition),
        ),
        cooldown_remaining=max(0, cooldown_until - iteration),
    )


def update_scale_history(previous_history: Any, current_scale: Any, alpha: float) -> Any:
    """Update an EMA of per-sample gradient or Hessian magnitudes."""

    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    if previous_history is None:
        previous_history = current_scale * 0

    if alpha == 0:
        return current_scale

    # The delta form preserves exact equality when both operands match and
    # reduces round-off excursions outside the convex hull of the inputs.
    return previous_history + (1 - alpha) * (current_scale - previous_history)


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
        return SchedulerDecision(
            volatility=volatility,
            stable_regime=False,
            score=volatility,
            observations=int(volatility is not None),
            reason="warmup_or_missing_observation",
        )

    threshold = disable_threshold if was_stable else enable_threshold
    return SchedulerDecision(
        volatility=volatility,
        stable_regime=volatility < threshold,
        score=volatility,
        observations=1,
        reason="threshold_evaluated",
    )


def decide_fixed_warmup(iteration: int, full_rounds: int) -> SchedulerDecision:
    """Keep the first ``full_rounds`` trees exact, then always enable sketching."""

    if full_rounds < 1:
        raise ValueError("full_rounds must be positive")
    enabled = iteration > full_rounds
    return SchedulerDecision(
        volatility=None,
        stable_regime=enabled,
        score=None,
        observations=0,
        reason="warmup_complete" if enabled else "warmup",
    )


def robust_window_score(observations: Tuple[float, ...], robust_k: float) -> Optional[float]:
    """Compute median + k * MAD / sqrt(W) for one finite observation window."""

    if robust_k < 0:
        raise ValueError("robust_k must be non-negative")
    if not observations:
        return None
    center = median(observations)
    mad = median(tuple(abs(value - center) for value in observations))
    return float(center + robust_k * mad / sqrt(len(observations)))


def advance_robust_scheduler(
    iteration: int,
    observation: Optional[float],
    state: RobustSchedulerState,
    config: RobustSchedulerConfig,
) -> Tuple[RobustSchedulerState, SchedulerDecision]:
    """Advance a scheduler from one optional observation without side effects."""

    observations = state.observations
    if observation is not None:
        if not 0 <= observation < float("inf"):
            raise ValueError("scheduler observations must be finite and non-negative")
        observations = (*observations, float(observation))[-config.window_size :]

    score = robust_window_score(observations, config.robust_k)
    ready = iteration >= config.min_warmup_rounds and len(observations) >= config.min_observations
    cooldown = max(0, state.cooldown_remaining - 1)
    stable = state.stable_regime
    safe_count = state.consecutive_safe
    unsafe_count = state.consecutive_unsafe
    reason = "retain"

    if not ready:
        stable = False
        safe_count = 0
        unsafe_count = 0
        reason = "warmup_or_insufficient_history"
    elif stable and observation is None:
        safe_count = 0
        unsafe_count = 0
        reason = "retain_without_observation"
    elif stable:
        unsafe_count = unsafe_count + 1 if score is not None and score >= config.disable_threshold else 0
        safe_count = 0
        if unsafe_count >= config.patience_off:
            stable = False
            cooldown = config.cooldown_rounds
            unsafe_count = 0
            reason = "risk_above_disable_threshold"
        else:
            reason = "stable_hysteresis"
    elif cooldown > 0:
        safe_count = 0
        unsafe_count = 0
        reason = "cooldown"
    elif observation is None:
        safe_count = 0
        unsafe_count = 0
        reason = "waiting_for_observation"
    else:
        safe_count = safe_count + 1 if score is not None and score < config.enable_threshold else 0
        unsafe_count = 0
        if safe_count >= config.patience_on:
            stable = True
            safe_count = 0
            reason = "patience_satisfied"
        else:
            reason = "waiting_for_safe_patience"

    next_state = RobustSchedulerState(
        stable_regime=stable,
        observations=observations,
        consecutive_safe=safe_count,
        consecutive_unsafe=unsafe_count,
        cooldown_remaining=cooldown,
    )
    decision = SchedulerDecision(
        volatility=observation,
        stable_regime=stable,
        score=score,
        observations=len(observations),
        reason=reason,
    )
    return next_state, decision
