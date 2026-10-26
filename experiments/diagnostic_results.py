"""Pure result aggregation for HASBoost diagnostic experiments."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

def summarize_history(
    sketch: Any,
    trees_built: int,
    gain_risk_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Convert callback traces into compact JSON-serializable diagnostics."""
    scheduler_trace = list(getattr(sketch, "scheduler_trace", ()))
    decision_trace = list(getattr(sketch, "decision_trace", ()))
    all_split_trace = list(getattr(sketch, "split_diagnostics_trace", ()))
    split_trace = [item for item in all_split_trace if item.get('diagnostic_kind', 'current') == 'current']
    volatilities = [
        float(item["volatility"])
        for item in scheduler_trace
        if item.get("volatility") is not None
    ]
    activation_iteration: Optional[int] = None
    if decision_trace:
        activation_iteration = int(decision_trace[0]["iteration"])
    risk_trace = [item for item in split_trace if item.get("regret") is not None]
    split_regrets = [float(item["regret"]) for item in risk_trace]
    split_weights = [max(0.0, float(item["full_best_gain"])) for item in risk_trace]
    weighted_p95 = _weighted_quantile(split_regrets, split_weights, 0.95)
    measured_iterations = sorted({int(item["iteration"]) for item in split_trace})
    tree_risks = {
        iteration: _weighted_quantile(
            [
                float(item["regret"])
                for item in risk_trace
                if int(item["iteration"]) == iteration
            ],
            [
                max(0.0, float(item["full_best_gain"]))
                for item in risk_trace
                if int(item["iteration"]) == iteration
            ],
            0.95,
        )
        for iteration in measured_iterations
    }
    valid_tree_risks = [risk for risk in tree_risks.values() if risk is not None]
    applied_iterations = {int(item["iteration"]) for item in decision_trace}
    if gain_risk_threshold < 0:
        raise ValueError("gain_risk_threshold must be non-negative")
    evaluated_activations = [
        risk
        for iteration, risk in tree_risks.items()
        if iteration in applied_iterations and risk is not None
    ]
    early_regrets = [
        float(item["regret"]) for item in risk_trace if int(item["iteration"]) <= 5
    ]
    late_regrets = [
        float(item["regret"]) for item in risk_trace if int(item["iteration"]) > 10
    ]
    output_budgets = [
        int(item["output_budget"])
        for item in scheduler_trace
        if item.get("output_budget") is not None
    ]
    budget_transitions = sum(
        left != right for left, right in zip(output_budgets, output_budgets[1:])
    )
    full_budget_flags = [
        bool(item["budget_is_full"])
        for item in scheduler_trace
        if item.get("budget_is_full") is not None
    ]

    return {
        "history_decisions": len(decision_trace),
        "history_decision_rate": (
            len(decision_trace) / trees_built if trees_built else 0.0
        ),
        "activation_iteration": activation_iteration,
        "mean_volatility": mean(volatilities) if volatilities else None,
        "last_volatility": volatilities[-1] if volatilities else None,
        "mean_sampled_rows": (
            mean(item["rows"] for item in decision_trace) if decision_trace else None
        ),
        "mean_sampled_outputs": (
            mean(item["outputs"] for item in decision_trace) if decision_trace else None
        ),
        "all_indices_unique": (
            all(
                item["rows"] == item["unique_rows"]
                and item["outputs"] == item["unique_outputs"]
                for item in decision_trace
            )
            if decision_trace
            else None
        ),
        "split_diagnostic_iterations": len(measured_iterations),
        "split_diagnostic_nodes": len(split_trace),
        "prospective_diagnostic_nodes": len(all_split_trace) - len(split_trace),
        "split_risk_nodes": len(risk_trace),
        "spurious_split_rate": (
            mean(float(item.get("spurious_split", False)) for item in split_trace)
            if split_trace
            else None
        ),
        "split_match_rate": (
            mean(float(item["same_split"]) for item in split_trace)
            if split_trace
            else None
        ),
        "split_regret_mean": mean(split_regrets) if split_regrets else None,
        "split_regret_max": max(split_regrets) if split_regrets else None,
        "split_regret_weighted_p95": weighted_p95,
        "tree_risk_mean": (
            mean(valid_tree_risks)
            if valid_tree_risks
            else None
        ),
        "false_activation_rate": (
            mean(float(risk > gain_risk_threshold) for risk in evaluated_activations)
            if evaluated_activations
            else None
        ),
        "early_split_regret_mean": mean(early_regrets) if early_regrets else None,
        "late_split_regret_mean": mean(late_regrets) if late_regrets else None,
        "gain_risk_threshold": gain_risk_threshold,
        "mean_output_budget": mean(output_budgets) if output_budgets else None,
        "min_output_budget": min(output_budgets) if output_budgets else None,
        "max_output_budget": max(output_budgets) if output_budgets else None,
        "budget_transitions": budget_transitions if output_budgets else None,
        "full_budget_rate": mean(full_budget_flags) if full_budget_flags else None,
    }


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> Optional[float]:
    """Small local adapter that keeps experiment aggregation dependency-light."""

    if not values:
        return None
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return float(ordered[min(len(ordered) - 1, int(quantile * len(ordered)))][0])
    target = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def add_baseline_ratios(
    records: Iterable[Mapping[str, Any]],
    baseline_variant: str = "full",
) -> List[Dict[str, Any]]:
    """Add quality deltas and train/inference speedups relative to a baseline."""
    output = [dict(record) for record in records]
    baseline = next(
        (record for record in output if record.get("variant") == baseline_variant),
        None,
    )
    if baseline is None:
        return output

    for record in output:
        record["accuracy_delta_vs_full"] = (
            record["accuracy"] - baseline["accuracy"]
        )
        record["f1_weighted_delta_vs_full"] = (
            record["f1_weighted"] - baseline["f1_weighted"]
        )
        record["train_speedup_vs_full"] = (
            baseline["train_seconds"] / record["train_seconds"]
            if record["train_seconds"]
            else None
        )
        record["inference_speedup_vs_full"] = (
            baseline["inference_median_ms"] / record["inference_median_ms"]
            if record["inference_median_ms"]
            else None
        )
    return output


def add_grouped_baseline_ratios(
    records: Iterable[Mapping[str, Any]],
    baseline_case: str = "full",
    group_keys: Sequence[str] = ("dataset", "scenario", "seed"),
) -> List[Dict[str, Any]]:
    """Add ratios against the matching baseline for every seed and scenario."""
    output = [dict(record) for record in records]
    baselines = {
        tuple(record.get(key) for key in group_keys): record
        for record in output
        if record.get("case_id") == baseline_case
    }
    for record in output:
        group = tuple(record.get(key) for key in group_keys)
        baseline = baselines.get(group)
        if baseline is None:
            continue
        record["accuracy_delta_vs_full"] = record["accuracy"] - baseline["accuracy"]
        record["f1_weighted_delta_vs_full"] = (
            record["f1_weighted"] - baseline["f1_weighted"]
        )
        record["train_speedup_vs_full"] = (
            baseline["train_seconds"] / record["train_seconds"]
            if record["train_seconds"]
            else None
        )
        record["inference_speedup_vs_full"] = (
            baseline["inference_median_ms"] / record["inference_median_ms"]
            if record["inference_median_ms"]
            else None
        )
    return output


DEFAULT_AGGREGATE_METRICS: Tuple[str, ...] = (
    "accuracy",
    "f1_weighted",
    "f1_macro",
    "f1_micro",
    "log_loss",
    "train_seconds",
    "inference_median_ms",
    "history_decision_rate",
    "activation_iteration",
    "get_weights_total_ms",
    "get_indexers_total_ms",
    "accuracy_delta_vs_full",
    "f1_weighted_delta_vs_full",
    "train_speedup_vs_full",
    "inference_speedup_vs_full",
    "split_match_rate",
    "split_regret_mean",
    "split_regret_max",
    "split_regret_weighted_p95",
    "early_split_regret_mean",
    "late_split_regret_mean",
    "tree_risk_mean",
    "false_activation_rate",
    "spurious_split_rate",
    "mean_output_budget",
    "min_output_budget",
    "max_output_budget",
    "budget_transitions",
    "full_budget_rate",
    "cupy_pool_growth_mib",
    "gpu_peak_fit_delta_mib",
)


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    metrics: Sequence[str] = DEFAULT_AGGREGATE_METRICS,
    group_keys: Sequence[str] = ("dataset", "scenario", "case_id", "variant"),
) -> List[Dict[str, Any]]:
    """Aggregate repeated seeds into deterministic mean and sample-std rows."""
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key) for key in group_keys)].append(record)

    summary = []
    for group in sorted(grouped, key=lambda values: tuple(str(value) for value in values)):
        items = grouped[group]
        row: Dict[str, Any] = dict(zip(group_keys, group))
        row["runs"] = len(items)
        ordered_items = sorted(items, key=lambda item: item.get("seed", 0))
        row["seeds"] = ",".join(str(item.get("seed")) for item in ordered_items)
        for field in (
            "row_subsample",
            "sketch_outputs",
            "smoothing_alpha",
            "stabilization_threshold",
            "destabilization_threshold",
            "min_warmup_rounds",
            "scheduler_window",
            "scheduler_min_observations",
            "scheduler_patience_on",
            "scheduler_control_interval",
            "adaptive_safe_threshold",
            "adaptive_unsafe_threshold",
            "adaptive_patience_down",
            "adaptive_min_dwell_rounds",
            "adaptive_cooldown_after_increase_rounds",
            "spectral_diversity_weight",
            "scheduler_mode",
            "full_leaf_values",
        ):
            row[field] = items[0].get(field)

        for metric in metrics:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) is not None
                and isinstance(item.get(metric), (int, float))
                and not isinstance(item.get(metric), bool)
            ]
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        summary.append(row)
    return summary
