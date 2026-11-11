"""Pure, typed plans for HASBoost benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple


class Scenario(str, Enum):
    """Closed set of supported benchmark scenarios."""

    MAIN = "main"
    ABLATIONS = "ablations"
    SCHEDULER_SWEEP = "scheduler_sweep"
    BUDGET_SWEEP = "budget_sweep"
    OBSERVABILITY = "observability"
    SCHEDULER_DIAGNOSTICS = "scheduler_diagnostics"
    RISK_BUDGET_SWEEP = "risk_budget_sweep"
    RMT_OFFLINE = "rmt_offline"
    SPECTRAL_SHADOW = "spectral_shadow"
    ADAPTIVE_BUDGET_DIAGNOSTICS = "adaptive_budget_diagnostics"
    ADAPTIVE_BUDGET = "adaptive_budget"
    ADAPTIVE_BUDGET_STABILITY = "adaptive_budget_stability"
    ROW_OUTPUT_FACTORIAL = "row_output_factorial"
    ROW_BUDGET_SWEEP = "row_budget_sweep"
    RMT_BUDGET = "rmt_budget"
    SPECTRAL_DIVERSITY = "spectral_diversity"


class Variant(str, Enum):
    """Model families used by experiment cases."""

    FULL = "full"
    SKETCHBOOST_TOPK = "sketchboost_topk"
    HASBOOST_SIGMOID_RAND = "hasboost_sigmoid_rand"
    HASBOOST_HYPERBOLIC_RAND = "hasboost_hyperbolic_rand"
    HASBOOST_SIGMOID_TOPK = "hasboost_sigmoid_topk"
    HASBOOST_SIGMOID_UNIFORM = "hasboost_sigmoid_uniform"
    HASBOOST_CURRENT_TOPK = "hasboost_current_topk"
    HASBOOST_ENERGY_BUDGET = "hasboost_energy_budget"
    HASBOOST_EMA_ENERGY = "hasboost_ema_energy"
    HASBOOST_SPECTRAL_TOPK = "hasboost_spectral_topk"
    HASBOOST_SPECTRAL_DIVERSE = "hasboost_spectral_diverse"


class SchedulerMode(str, Enum):
    """How history-based sketching is gated."""

    CURRENT = "current"
    GATED = "current"
    NEVER = "never"
    FIXED_WARMUP = "fixed_warmup"
    AFTER_WARMUP = "fixed_warmup"
    ROBUST_WINDOW = "robust_window"
    SHADOW_RISK = "shadow_risk"
    ADAPTIVE_BUDGET = "adaptive_budget"


@dataclass(frozen=True)
class ExperimentCase:
    """One immutable model configuration inside a scenario."""

    case_id: str
    variant: Variant
    scheduler_mode: SchedulerMode = SchedulerMode.CURRENT
    row_subsample: Optional[float] = None
    sketch_outputs: Optional[int] = None
    use_all_outputs: bool = False
    full_leaf_values: bool = True
    smoothing_alpha: Optional[float] = None
    stabilization_threshold: Optional[float] = None
    destabilization_threshold: Optional[float] = None
    min_warmup_rounds: Optional[int] = None
    scheduler_window: Optional[int] = None
    scheduler_min_observations: Optional[int] = None
    scheduler_patience_on: Optional[int] = None
    scheduler_patience_off: Optional[int] = None
    scheduler_robust_k: Optional[float] = None
    scheduler_cooldown_rounds: Optional[int] = None
    scheduler_control_interval: Optional[int] = None
    adaptive_budgets: Optional[Tuple[int, ...]] = None
    adaptive_safe_threshold: Optional[float] = None
    adaptive_unsafe_threshold: Optional[float] = None
    adaptive_patience_down: Optional[int] = None
    adaptive_min_dwell_rounds: Optional[int] = None
    adaptive_cooldown_after_increase_rounds: Optional[int] = None
    spectral_diversity_weight: Optional[float] = None
    collect_split_diagnostics: bool = False

    def resolve(self, defaults: "PlanDefaults", n_outputs: int) -> "ResolvedCase":
        """Resolve dataset-dependent budgets and scheduler thresholds."""
        row_subsample = (
            defaults.row_subsample
            if self.row_subsample is None
            else self.row_subsample
        )
        requested_outputs = (
            n_outputs
            if self.use_all_outputs
            else (
                defaults.sketch_outputs
                if self.sketch_outputs is None
                else self.sketch_outputs
            )
        )
        sketch_outputs = min(n_outputs, max(1, requested_outputs))
        smoothing_alpha = (
            defaults.smoothing_alpha
            if self.smoothing_alpha is None
            else self.smoothing_alpha
        )
        threshold = (
            defaults.stabilization_threshold
            if self.stabilization_threshold is None
            else self.stabilization_threshold
        )
        disable_threshold = (
            defaults.destabilization_threshold
            if self.destabilization_threshold is None
            else self.destabilization_threshold
        )
        min_warmup_rounds = (
            defaults.min_warmup_rounds
            if self.min_warmup_rounds is None
            else self.min_warmup_rounds
        )
        if self.scheduler_mode is SchedulerMode.NEVER:
            threshold = 0.0
        requested_adaptive_budgets = (
            defaults.adaptive_budgets
            if self.adaptive_budgets is None
            else self.adaptive_budgets
        )
        if not requested_adaptive_budgets or any(
            budget < 1 for budget in requested_adaptive_budgets
        ):
            raise ValueError("adaptive budgets must be positive")
        adaptive_budgets = tuple(
            sorted(
                {min(n_outputs, int(budget)) for budget in requested_adaptive_budgets}
                | {n_outputs}
            )
        )
        adaptive_safe_threshold = (
            defaults.adaptive_safe_threshold
            if self.adaptive_safe_threshold is None
            else self.adaptive_safe_threshold
        )
        adaptive_unsafe_threshold = (
            defaults.adaptive_unsafe_threshold
            if self.adaptive_unsafe_threshold is None
            else self.adaptive_unsafe_threshold
        )
        if adaptive_safe_threshold < 0:
            raise ValueError("adaptive_safe_threshold must be non-negative")
        if adaptive_unsafe_threshold < adaptive_safe_threshold:
            raise ValueError(
                "adaptive_unsafe_threshold must be at least adaptive_safe_threshold"
            )
        adaptive_patience_down = (
            defaults.adaptive_patience_down
            if self.adaptive_patience_down is None
            else self.adaptive_patience_down
        )
        if adaptive_patience_down < 1:
            raise ValueError("adaptive_patience_down must be positive")
        adaptive_min_dwell_rounds = (
            defaults.adaptive_min_dwell_rounds
            if self.adaptive_min_dwell_rounds is None
            else self.adaptive_min_dwell_rounds
        )
        adaptive_cooldown_after_increase_rounds = (
            defaults.adaptive_cooldown_after_increase_rounds
            if self.adaptive_cooldown_after_increase_rounds is None
            else self.adaptive_cooldown_after_increase_rounds
        )
        if adaptive_min_dwell_rounds < 0:
            raise ValueError("adaptive_min_dwell_rounds must be non-negative")
        if adaptive_cooldown_after_increase_rounds < 0:
            raise ValueError(
                "adaptive_cooldown_after_increase_rounds must be non-negative"
            )
        spectral_diversity_weight = (
            defaults.spectral_diversity_weight
            if self.spectral_diversity_weight is None
            else self.spectral_diversity_weight
        )
        if not 0 <= spectral_diversity_weight <= 1:
            raise ValueError("spectral_diversity_weight must be within [0, 1]")

        return ResolvedCase(
            case_id=self.case_id,
            variant=self.variant,
            scheduler_mode=self.scheduler_mode,
            row_subsample=row_subsample,
            sketch_outputs=sketch_outputs,
            full_leaf_values=self.full_leaf_values,
            smoothing_alpha=smoothing_alpha,
            stabilization_threshold=threshold,
            destabilization_threshold=max(threshold, disable_threshold),
            min_warmup_rounds=min_warmup_rounds,
            scheduler_window=(
                defaults.scheduler_window
                if self.scheduler_window is None
                else self.scheduler_window
            ),
            scheduler_min_observations=(
                defaults.scheduler_min_observations
                if self.scheduler_min_observations is None
                else self.scheduler_min_observations
            ),
            scheduler_patience_on=(
                defaults.scheduler_patience_on
                if self.scheduler_patience_on is None
                else self.scheduler_patience_on
            ),
            scheduler_patience_off=(
                defaults.scheduler_patience_off
                if self.scheduler_patience_off is None
                else self.scheduler_patience_off
            ),
            scheduler_robust_k=(
                defaults.scheduler_robust_k
                if self.scheduler_robust_k is None
                else self.scheduler_robust_k
            ),
            scheduler_cooldown_rounds=(
                defaults.scheduler_cooldown_rounds
                if self.scheduler_cooldown_rounds is None
                else self.scheduler_cooldown_rounds
            ),
            scheduler_control_interval=(
                defaults.scheduler_control_interval
                if self.scheduler_control_interval is None
                else self.scheduler_control_interval
            ),
            adaptive_budgets=adaptive_budgets,
            adaptive_safe_threshold=adaptive_safe_threshold,
            adaptive_unsafe_threshold=adaptive_unsafe_threshold,
            adaptive_patience_down=adaptive_patience_down,
            adaptive_min_dwell_rounds=adaptive_min_dwell_rounds,
            adaptive_cooldown_after_increase_rounds=(
                adaptive_cooldown_after_increase_rounds
            ),
            spectral_diversity_weight=spectral_diversity_weight,
            collect_split_diagnostics=self.collect_split_diagnostics,
        )


@dataclass(frozen=True)
class ResolvedCase:
    """Experiment case whose parameters are ready for model construction."""

    case_id: str
    variant: Variant
    scheduler_mode: SchedulerMode
    row_subsample: float
    sketch_outputs: int
    full_leaf_values: bool
    smoothing_alpha: float
    stabilization_threshold: float
    destabilization_threshold: float
    min_warmup_rounds: int
    scheduler_window: int
    scheduler_min_observations: int
    scheduler_patience_on: int
    scheduler_patience_off: int
    scheduler_robust_k: float
    scheduler_cooldown_rounds: int
    scheduler_control_interval: int
    adaptive_budgets: Tuple[int, ...]
    adaptive_safe_threshold: float
    adaptive_unsafe_threshold: float
    adaptive_patience_down: int
    adaptive_min_dwell_rounds: int
    adaptive_cooldown_after_increase_rounds: int
    spectral_diversity_weight: float
    collect_split_diagnostics: bool


@dataclass(frozen=True)
class PlanDefaults:
    """Shared parameters used to resolve partial experiment cases."""

    row_subsample: float = 0.5
    sketch_outputs: int = 25
    smoothing_alpha: float = 0.9
    stabilization_threshold: float = 0.2
    destabilization_threshold: float = 0.2
    min_warmup_rounds: int = 5
    scheduler_window: int = 8
    scheduler_min_observations: int = 4
    scheduler_patience_on: int = 2
    scheduler_patience_off: int = 1
    scheduler_robust_k: float = 1.0
    scheduler_cooldown_rounds: int = 2
    scheduler_control_interval: int = 5
    adaptive_budgets: Tuple[int, ...] = (25, 50, 75, 100)
    adaptive_safe_threshold: float = 0.10
    adaptive_unsafe_threshold: float = 0.15
    adaptive_patience_down: int = 2
    adaptive_min_dwell_rounds: int = 5
    adaptive_cooldown_after_increase_rounds: int = 10
    spectral_diversity_weight: float = 0.25


@dataclass(frozen=True)
class PlannedRun:
    """A case paired with one explicit random seed."""

    scenario: Scenario
    case: ResolvedCase
    seed: int

    @property
    def key(self) -> Tuple[str, str, int]:
        return self.scenario.value, self.case.case_id, self.seed


def full_case(case_id: str = "full") -> ExperimentCase:
    """Return an explicit all-row, all-output reference configuration."""

    return ExperimentCase(
        case_id,
        Variant.FULL,
        row_subsample=1.0,
        use_all_outputs=True,
    )


def main_cases() -> Tuple[ExperimentCase, ...]:
    """Five aligned variants used for the primary comparison."""
    return (
        full_case(),
        ExperimentCase("sketchboost_topk", Variant.SKETCHBOOST_TOPK),
        ExperimentCase("hasboost_sigmoid_rand", Variant.HASBOOST_SIGMOID_RAND),
        ExperimentCase("hasboost_hyperbolic_rand", Variant.HASBOOST_HYPERBOLIC_RAND),
        ExperimentCase("hasboost_sigmoid_topk", Variant.HASBOOST_SIGMOID_TOPK),
    )


def ablation_cases() -> Tuple[ExperimentCase, ...]:
    """Cases that isolate gating, allocation, axes, and leaf recomputation."""
    return (
        full_case(),
        ExperimentCase("sketchboost_topk", Variant.SKETCHBOOST_TOPK),
        ExperimentCase("gated_joint_pq", Variant.HASBOOST_SIGMOID_RAND),
        ExperimentCase(
            "never_sketch",
            Variant.HASBOOST_SIGMOID_RAND,
            scheduler_mode=SchedulerMode.NEVER,
        ),
        ExperimentCase(
            "after_warmup_pq",
            Variant.HASBOOST_SIGMOID_RAND,
            scheduler_mode=SchedulerMode.AFTER_WARMUP,
        ),
        ExperimentCase("uniform_joint", Variant.HASBOOST_SIGMOID_UNIFORM),
        ExperimentCase(
            "rows_only_pq",
            Variant.HASBOOST_SIGMOID_RAND,
            use_all_outputs=True,
        ),
        ExperimentCase(
            "outputs_only_pq",
            Variant.HASBOOST_SIGMOID_RAND,
            row_subsample=1.0,
        ),
        ExperimentCase(
            "sampled_leaf_values",
            Variant.HASBOOST_SIGMOID_RAND,
            full_leaf_values=False,
        ),
        ExperimentCase("greedy_joint", Variant.HASBOOST_SIGMOID_TOPK),
    )


def scheduler_sweep_cases() -> Tuple[ExperimentCase, ...]:
    """A focused 3x3 sweep over scheduler threshold and EMA inertia."""
    cases = [full_case()]
    for threshold, alpha in product((0.1, 0.2, 0.3), (0.8, 0.9, 0.95)):
        cases.append(
            ExperimentCase(
                case_id=f"theta_{threshold:g}_alpha_{alpha:g}",
                variant=Variant.HASBOOST_SIGMOID_RAND,
                smoothing_alpha=alpha,
                stabilization_threshold=threshold,
            )
        )
    return tuple(cases)


def budget_sweep_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Vary rows and outputs separately instead of taking their full product."""
    cases = [full_case()]
    for row_subsample in (0.25, 0.5, 0.75):
        cases.append(
            ExperimentCase(
                case_id=f"rows_{row_subsample:g}",
                variant=Variant.HASBOOST_SIGMOID_RAND,
                row_subsample=row_subsample,
            )
        )

    seen_budgets = set()
    for requested in (10, 25, 50):
        budget = min(n_outputs, requested)
        if budget in seen_budgets:
            continue
        seen_budgets.add(budget)
        cases.append(
            ExperimentCase(
                case_id=f"outputs_{budget}",
                variant=Variant.HASBOOST_SIGMOID_RAND,
                sketch_outputs=budget,
            )
        )
    return tuple(cases)


def observability_cases() -> Tuple[ExperimentCase, ...]:
    """Full training plus a deterministic output-only shadow policy."""

    return (
        full_case(),
        ExperimentCase(
            "shadow_topk_outputs",
            Variant.HASBOOST_SIGMOID_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            collect_split_diagnostics=True,
        ),
    )


def scheduler_diagnostic_cases() -> Tuple[ExperimentCase, ...]:
    """Causal scheduler comparison with one output policy and fixed budgets."""

    cases = [full_case()]
    cases.append(
        ExperimentCase(
            "current_output_only",
            Variant.HASBOOST_SIGMOID_RAND,
            row_subsample=1.0,
            scheduler_mode=SchedulerMode.CURRENT,
            min_warmup_rounds=2,
            collect_split_diagnostics=True,
        )
    )
    for warmup in (2, 5, 10):
        cases.append(
            ExperimentCase(
                f"fixed_warmup_{warmup}",
                Variant.HASBOOST_SIGMOID_RAND,
                row_subsample=1.0,
                scheduler_mode=SchedulerMode.FIXED_WARMUP,
                min_warmup_rounds=warmup,
                collect_split_diagnostics=True,
            )
        )
    cases.extend(
        (
            ExperimentCase(
                "robust_window",
                Variant.HASBOOST_SIGMOID_RAND,
                row_subsample=1.0,
                scheduler_mode=SchedulerMode.ROBUST_WINDOW,
                min_warmup_rounds=5,
                destabilization_threshold=0.3,
                collect_split_diagnostics=True,
            ),
            ExperimentCase(
                "shadow_risk",
                Variant.HASBOOST_SIGMOID_RAND,
                row_subsample=1.0,
                scheduler_mode=SchedulerMode.SHADOW_RISK,
                min_warmup_rounds=5,
                stabilization_threshold=0.05,
                destabilization_threshold=0.10,
                collect_split_diagnostics=True,
            ),
        )
    )
    return tuple(cases)


def risk_budget_sweep_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Non-mutating split-risk curves for output method and budget."""

    cases = [full_case()]
    budgets = tuple(
        dict.fromkeys(min(n_outputs, budget) for budget in (10, 25, 50, 75))
    )
    variants = (
        ("history_rand", Variant.HASBOOST_SIGMOID_RAND),
        ("history_topk", Variant.HASBOOST_SIGMOID_TOPK),
    )
    for label, variant in variants:
        for budget in budgets:
            cases.append(
                ExperimentCase(
                    case_id=f"{label}_{budget}",
                    variant=variant,
                    scheduler_mode=SchedulerMode.NEVER,
                    row_subsample=1.0,
                    sketch_outputs=budget,
                    collect_split_diagnostics=True,
                )
            )
    return tuple(cases)


def rmt_offline_cases() -> Tuple[ExperimentCase, ...]:
    """A full-statistics source trajectory for representation studies."""

    return (full_case("full_trajectory"),)


def spectral_shadow_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Non-mutating output-policy comparison with aligned output budgets."""

    budgets = tuple(
        dict.fromkeys(min(n_outputs, budget) for budget in (25, 50, 75))
    )
    policies = (
        ('uniform', Variant.HASBOOST_SIGMOID_UNIFORM),
        ('current_topk', Variant.HASBOOST_CURRENT_TOPK),
        ('history_topk', Variant.HASBOOST_SIGMOID_TOPK),
        ('spectral_topk', Variant.HASBOOST_SPECTRAL_TOPK),
    )
    cases = [full_case()]
    for policy, variant in policies:
        cases.extend(
            ExperimentCase(
                case_id=f"{policy}_{budget}",
                variant=variant,
                scheduler_mode=SchedulerMode.NEVER,
                row_subsample=1.0,
                sketch_outputs=budget,
                collect_split_diagnostics=True,
            )
            for budget in budgets
        )
    return tuple(cases)


def adaptive_budget_diagnostic_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Long, non-mutating risk curves used to calibrate output budgets."""

    budgets = tuple(
        dict.fromkeys(min(n_outputs, budget) for budget in (25, 50, 75, 100))
    )
    policies = (
        ("current_topk", Variant.HASBOOST_CURRENT_TOPK),
        ("history_topk", Variant.HASBOOST_SIGMOID_TOPK),
    )
    cases = [full_case()]
    for policy, variant in policies:
        cases.extend(
            ExperimentCase(
                case_id=f"{policy}_{budget}",
                variant=variant,
                scheduler_mode=SchedulerMode.NEVER,
                row_subsample=1.0,
                sketch_outputs=budget,
                collect_split_diagnostics=True,
            )
            for budget in budgets
        )
    return tuple(cases)


def adaptive_budget_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Fixed-budget controls and adaptive current/history Top-K policies."""

    cases = [full_case()]
    policies = (
        ("current", Variant.HASBOOST_CURRENT_TOPK),
        ("history", Variant.HASBOOST_SIGMOID_TOPK),
    )
    for policy, variant in policies:
        for budget in tuple(
            dict.fromkeys(min(n_outputs, value) for value in (50, 75))
        ):
            cases.append(
                ExperimentCase(
                    case_id=f"fixed_{policy}_{budget}",
                    variant=variant,
                    scheduler_mode=SchedulerMode.FIXED_WARMUP,
                    row_subsample=1.0,
                    sketch_outputs=budget,
                    min_warmup_rounds=5,
                    collect_split_diagnostics=True,
                )
            )
        cases.append(
            ExperimentCase(
                case_id=f"adaptive_{policy}",
                variant=variant,
                scheduler_mode=SchedulerMode.ADAPTIVE_BUDGET,
                row_subsample=1.0,
                sketch_outputs=min(n_outputs, 75),
                min_warmup_rounds=5,
                collect_split_diagnostics=True,
            )
        )
    return tuple(cases)


def adaptive_budget_stability_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Tune patience and control cadence with explicit dwell and cooldown."""

    fixed_budget = min(n_outputs, 75)
    cases = [
        full_case(),
        ExperimentCase(
            case_id=f"fixed_current_{fixed_budget}",
            variant=Variant.HASBOOST_CURRENT_TOPK,
            scheduler_mode=SchedulerMode.FIXED_WARMUP,
            row_subsample=1.0,
            sketch_outputs=fixed_budget,
            min_warmup_rounds=5,
            collect_split_diagnostics=True,
        ),
    ]
    for patience, interval in product((2, 3, 4), (5, 10)):
        cases.append(
            ExperimentCase(
                case_id=f"adaptive_current_p{patience}_c{interval}",
                variant=Variant.HASBOOST_CURRENT_TOPK,
                scheduler_mode=SchedulerMode.ADAPTIVE_BUDGET,
                row_subsample=1.0,
                sketch_outputs=fixed_budget,
                min_warmup_rounds=5,
                scheduler_control_interval=interval,
                adaptive_patience_down=patience,
                adaptive_min_dwell_rounds=interval,
                adaptive_cooldown_after_increase_rounds=2 * interval,
                collect_split_diagnostics=True,
            )
        )
    return tuple(cases)


def row_budget_sweep_cases() -> Tuple[ExperimentCase, ...]:
    """Measure row-only subsampling while keeping every output."""

    cases = [full_case()]
    for policy, variant in (
        ("current", Variant.HASBOOST_CURRENT_TOPK),
        ("history", Variant.HASBOOST_SIGMOID_TOPK),
    ):
        for row_subsample in (0.9, 0.75, 0.5):
            label = int(round(100 * row_subsample))
            cases.append(
                ExperimentCase(
                    case_id=f"{policy}_rows_{label}",
                    variant=variant,
                    scheduler_mode=SchedulerMode.FIXED_WARMUP,
                    row_subsample=row_subsample,
                    use_all_outputs=True,
                    min_warmup_rounds=5,
                )
            )
    return tuple(cases)


def rmt_budget_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Collect aligned spectral state and current-Top-K risk labels."""

    budgets = tuple(dict.fromkeys(min(n_outputs, value) for value in (25, 50, 75)))
    cases = [full_case()]
    cases.extend(
        ExperimentCase(
            case_id=f"current_topk_{budget}",
            variant=Variant.HASBOOST_CURRENT_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            sketch_outputs=budget,
            collect_split_diagnostics=True,
        )
        for budget in budgets
    )
    cases.append(
        ExperimentCase(
            case_id=f"spectral_probe_{min(n_outputs, 75)}",
            variant=Variant.HASBOOST_SPECTRAL_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            sketch_outputs=min(n_outputs, 75),
            collect_split_diagnostics=True,
            spectral_diversity_weight=0.0,
        )
    )
    return tuple(cases)


def spectral_diversity_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Compare direct and redundancy-aware spectral output selection."""

    budget = min(n_outputs, 75)
    shared = {
        "scheduler_mode": SchedulerMode.NEVER,
        "row_subsample": 1.0,
        "sketch_outputs": budget,
        "collect_split_diagnostics": True,
    }
    cases = [
        full_case(),
        ExperimentCase(
            case_id=f"current_topk_{budget}",
            variant=Variant.HASBOOST_CURRENT_TOPK,
            **shared,
        ),
        ExperimentCase(
            case_id=f"spectral_topk_{budget}",
            variant=Variant.HASBOOST_SPECTRAL_TOPK,
            spectral_diversity_weight=0.0,
            **shared,
        ),
    ]
    for label, weight in (("w05_", 0.05), ("w10_", 0.10), ("", 0.25)):
        cases.append(
            ExperimentCase(
                case_id=f"spectral_diverse_{label}{budget}",
                variant=Variant.HASBOOST_SPECTRAL_DIVERSE,
                spectral_diversity_weight=weight,
                **shared,
            )
        )
    return tuple(cases)


def row_output_factorial_cases(n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Factor row and output selection for current/history Top-K policies."""

    output_budget = min(n_outputs, 75)
    cases = [full_case()]
    policies = (
        ("current", Variant.HASBOOST_CURRENT_TOPK),
        ("history", Variant.HASBOOST_SIGMOID_TOPK),
    )
    for policy, variant in policies:
        shared = {
            "variant": variant,
            "scheduler_mode": SchedulerMode.FIXED_WARMUP,
            "min_warmup_rounds": 5,
            "collect_split_diagnostics": True,
        }
        cases.extend(
            (
                ExperimentCase(
                    case_id=f"{policy}_rows_only",
                    row_subsample=0.5,
                    use_all_outputs=True,
                    **shared,
                ),
                ExperimentCase(
                    case_id=f"{policy}_outputs_only_{output_budget}",
                    row_subsample=1.0,
                    sketch_outputs=output_budget,
                    **shared,
                ),
                ExperimentCase(
                    case_id=f"{policy}_joint_50x{output_budget}",
                    row_subsample=0.5,
                    sketch_outputs=output_budget,
                    **shared,
                ),
            )
        )
    return tuple(cases)


def cases_for_scenario(scenario: Scenario, n_outputs: int) -> Tuple[ExperimentCase, ...]:
    """Return the deterministic case sequence for one scenario."""
    if scenario is Scenario.MAIN:
        return main_cases()
    if scenario is Scenario.ABLATIONS:
        return ablation_cases()
    if scenario is Scenario.SCHEDULER_SWEEP:
        return scheduler_sweep_cases()
    if scenario is Scenario.BUDGET_SWEEP:
        return budget_sweep_cases(n_outputs)
    if scenario is Scenario.OBSERVABILITY:
        return observability_cases()
    if scenario is Scenario.SCHEDULER_DIAGNOSTICS:
        return scheduler_diagnostic_cases()
    if scenario is Scenario.RISK_BUDGET_SWEEP:
        return risk_budget_sweep_cases(n_outputs)
    if scenario is Scenario.RMT_OFFLINE:
        return rmt_offline_cases()
    if scenario is Scenario.SPECTRAL_SHADOW:
        return spectral_shadow_cases(n_outputs)
    if scenario is Scenario.ADAPTIVE_BUDGET_DIAGNOSTICS:
        return adaptive_budget_diagnostic_cases(n_outputs)
    if scenario is Scenario.ADAPTIVE_BUDGET:
        return adaptive_budget_cases(n_outputs)
    if scenario is Scenario.ADAPTIVE_BUDGET_STABILITY:
        return adaptive_budget_stability_cases(n_outputs)
    if scenario is Scenario.ROW_OUTPUT_FACTORIAL:
        return row_output_factorial_cases(n_outputs)
    if scenario is Scenario.ROW_BUDGET_SWEEP:
        return row_budget_sweep_cases()
    if scenario is Scenario.RMT_BUDGET:
        return rmt_budget_cases(n_outputs)
    if scenario is Scenario.SPECTRAL_DIVERSITY:
        return spectral_diversity_cases(n_outputs)
    raise ValueError(f"Unsupported scenario: {scenario}")


def build_run_plan(
    scenario: Scenario,
    cases: Sequence[ExperimentCase],
    seeds: Iterable[int],
    defaults: PlanDefaults,
    n_outputs: int,
) -> Tuple[PlannedRun, ...]:
    """Take the Cartesian product of cases and unique seeds in stable order."""
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds:
        raise ValueError("at least one seed is required")
    resolved = tuple(case.resolve(defaults, n_outputs) for case in cases)
    runs = tuple(
        PlannedRun(scenario=scenario, case=case, seed=seed)
        for case in resolved
        for seed in unique_seeds
    )
    keys = [run.key for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("case ids must be unique within a scenario")
    return runs
