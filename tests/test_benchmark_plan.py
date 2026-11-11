"""Invariant tests for pure HASBoost experiment planning."""

import pytest
from hypothesis import given, strategies as st

from experiments.benchmark_plan import (
    ExperimentCase,
    PlanDefaults,
    Scenario,
    SchedulerMode,
    Variant,
    ablation_cases,
    adaptive_budget_cases,
    adaptive_budget_diagnostic_cases,
    adaptive_budget_stability_cases,
    budget_sweep_cases,
    build_run_plan,
    cases_for_scenario,
    observability_cases,
    risk_budget_sweep_cases,
    rmt_budget_cases,
    row_output_factorial_cases,
    row_budget_sweep_cases,
    scheduler_diagnostic_cases,
    scheduler_sweep_cases,
    spectral_shadow_cases,
    spectral_diversity_cases,
)


def test_adaptive_budget_patience_must_be_positive():
    case = ExperimentCase(
        case_id="invalid_adaptive_patience",
        variant=Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.ADAPTIVE_BUDGET,
        adaptive_patience_down=0,
    )

    with pytest.raises(ValueError, match="adaptive_patience_down"):
        case.resolve(PlanDefaults(), n_outputs=100)


@pytest.mark.parametrize(
    "field_name",
    ("adaptive_min_dwell_rounds", "adaptive_cooldown_after_increase_rounds"),
)
def test_adaptive_budget_timing_constraints_must_be_non_negative(field_name):
    case = ExperimentCase(
        case_id="invalid_timing",
        variant=Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.ADAPTIVE_BUDGET,
        **{field_name: -1},
    )

    with pytest.raises(ValueError, match=field_name):
        case.resolve(PlanDefaults(), n_outputs=100)


@given(
    seeds=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1),
    n_outputs=st.integers(min_value=1, max_value=500),
)
def test_run_plan_is_unique_and_has_expected_cardinality(seeds, n_outputs):
    cases = cases_for_scenario(Scenario.MAIN, n_outputs)

    plan = build_run_plan(
        Scenario.MAIN,
        cases,
        seeds,
        PlanDefaults(),
        n_outputs,
    )

    assert len(plan) == len(cases) * len(set(seeds))
    assert len({run.key for run in plan}) == len(plan)


@given(n_outputs=st.integers(min_value=1, max_value=500))
def test_every_resolved_budget_stays_inside_dataset_bounds(n_outputs):
    for scenario in Scenario:
        cases = cases_for_scenario(scenario, n_outputs)
        plan = build_run_plan(
            scenario,
            cases,
            [42],
            PlanDefaults(),
            n_outputs,
        )
        for run in plan:
            assert 0 < run.case.row_subsample <= 1
            assert 1 <= run.case.sketch_outputs <= n_outputs


@given(n_outputs=st.integers(min_value=1, max_value=500))
def test_budget_sweep_changes_one_axis_at_a_time(n_outputs):
    defaults = PlanDefaults(row_subsample=0.5, sketch_outputs=25)
    plan = build_run_plan(
        Scenario.BUDGET_SWEEP,
        budget_sweep_cases(n_outputs),
        [42],
        defaults,
        n_outputs,
    )

    for run in plan:
        if run.case.case_id.startswith("rows_"):
            assert run.case.sketch_outputs == min(defaults.sketch_outputs, n_outputs)
        if run.case.case_id.startswith("outputs_"):
            assert run.case.row_subsample == defaults.row_subsample


def test_scheduler_sweep_is_a_unique_three_by_three_grid_plus_baseline():
    cases = scheduler_sweep_cases()

    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
    assert cases[0].case_id == "full"


def test_ablation_plan_covers_every_requested_mechanism():
    case_ids = {case.case_id for case in ablation_cases()}

    assert case_ids == {
        "full",
        "sketchboost_topk",
        "gated_joint_pq",
        "never_sketch",
        "after_warmup_pq",
        "uniform_joint",
        "rows_only_pq",
        "outputs_only_pq",
        "sampled_leaf_values",
        "greedy_joint",
    }


def test_observability_uses_a_full_model_and_non_mutating_shadow_case():
    cases = observability_cases()

    assert [case.case_id for case in cases] == ["full", "shadow_topk_outputs"]
    assert cases[1].row_subsample == 1.0
    assert cases[1].collect_split_diagnostics is True


def test_scheduler_diagnostics_compare_rules_with_one_output_only_policy():
    cases = scheduler_diagnostic_cases()
    scheduled = cases[1:]

    assert {case.min_warmup_rounds for case in scheduled if case.case_id.startswith("fixed_")} == {
        2,
        5,
        10,
    }
    assert all(case.row_subsample == 1.0 for case in scheduled)
    assert all(case.collect_split_diagnostics for case in scheduled)


def test_risk_budget_sweep_is_shadow_only_and_changes_output_budget():
    cases = risk_budget_sweep_cases(100)
    shadows = cases[1:]

    assert {case.sketch_outputs for case in shadows} == {10, 25, 50, 75}
    assert {case.variant for case in shadows} == {
        Variant.HASBOOST_SIGMOID_RAND,
        Variant.HASBOOST_SIGMOID_TOPK,
    }
    assert all(case.scheduler_mode is SchedulerMode.NEVER for case in shadows)
    assert all(case.row_subsample == 1.0 for case in shadows)
    assert all(case.collect_split_diagnostics for case in shadows)


def test_spectral_shadow_is_output_only_and_never_changes_training():
    cases = spectral_shadow_cases(100)
    shadows = cases[1:]

    assert {case.sketch_outputs for case in shadows} == {25, 50, 75}
    assert {case.variant for case in shadows} == {
        Variant.HASBOOST_SIGMOID_UNIFORM,
        Variant.HASBOOST_CURRENT_TOPK,
        Variant.HASBOOST_SIGMOID_TOPK,
        Variant.HASBOOST_SPECTRAL_TOPK,
    }
    assert all(case.scheduler_mode is SchedulerMode.NEVER for case in shadows)
    assert all(case.row_subsample == 1.0 for case in shadows)
    assert all(case.collect_split_diagnostics for case in shadows)


def test_adaptive_budget_diagnostics_align_policies_and_budgets():
    cases = adaptive_budget_diagnostic_cases(100)
    shadows = cases[1:]

    assert {case.sketch_outputs for case in shadows} == {25, 50, 75, 100}
    assert {case.variant for case in shadows} == {
        Variant.HASBOOST_CURRENT_TOPK,
        Variant.HASBOOST_SIGMOID_TOPK,
    }
    assert all(case.scheduler_mode is SchedulerMode.NEVER for case in shadows)
    assert all(case.row_subsample == 1.0 for case in shadows)


def test_adaptive_budget_plan_contains_fixed_controls_and_two_adaptive_policies():
    cases = adaptive_budget_cases(100)
    adaptive = [
        case for case in cases
        if case.scheduler_mode is SchedulerMode.ADAPTIVE_BUDGET
    ]

    assert {case.case_id for case in adaptive} == {
        "adaptive_current",
        "adaptive_history",
    }
    assert all(case.sketch_outputs == 75 for case in adaptive)
    assert all(case.row_subsample == 1.0 for case in adaptive)


def test_adaptive_budget_stability_grid_has_explicit_transition_guards():
    cases = adaptive_budget_stability_cases(100)
    adaptive = [
        case for case in cases
        if case.scheduler_mode is SchedulerMode.ADAPTIVE_BUDGET
    ]

    assert len(cases) == 8
    assert len(adaptive) == 6
    assert {case.adaptive_patience_down for case in adaptive} == {2, 3, 4}
    assert {case.scheduler_control_interval for case in adaptive} == {5, 10}
    assert all(
        case.adaptive_min_dwell_rounds == case.scheduler_control_interval
        for case in adaptive
    )
    assert all(
        case.adaptive_cooldown_after_increase_rounds
        == 2 * case.scheduler_control_interval
        for case in adaptive
    )


def test_row_budget_sweep_keeps_all_outputs_and_varies_only_rows():
    cases = row_budget_sweep_cases()
    sampled = cases[1:]

    assert len(cases) == 7
    assert {case.row_subsample for case in sampled} == {0.5, 0.75, 0.9}
    assert all(case.use_all_outputs for case in sampled)
    assert all(case.scheduler_mode is SchedulerMode.FIXED_WARMUP for case in sampled)


def test_rmt_budget_plan_aligns_three_risk_labels_with_one_spectral_probe():
    cases = rmt_budget_cases(100)

    assert {case.case_id for case in cases[1:-1]} == {
        "current_topk_25",
        "current_topk_50",
        "current_topk_75",
    }
    assert cases[-1].case_id == "spectral_probe_75"
    assert all(case.scheduler_mode is SchedulerMode.NEVER for case in cases[1:])


def test_spectral_diversity_plan_changes_only_the_output_policy():
    cases = spectral_diversity_cases(100)
    policies = cases[1:]

    assert len(cases) == 6
    assert {case.sketch_outputs for case in policies} == {75}
    assert {case.spectral_diversity_weight for case in policies} == {
        None,
        0.0,
        0.05,
        0.10,
        0.25,
    }
    assert all(case.row_subsample == 1.0 for case in policies)


def test_row_output_factorial_changes_the_two_axes_independently():
    cases = row_output_factorial_cases(100)
    by_id = {case.case_id: case for case in cases}

    for policy in ("current", "history"):
        rows = by_id[f"{policy}_rows_only"]
        outputs = by_id[f"{policy}_outputs_only_75"]
        joint = by_id[f"{policy}_joint_50x75"]
        assert rows.row_subsample == 0.5 and rows.use_all_outputs
        assert outputs.row_subsample == 1.0 and outputs.sketch_outputs == 75
        assert joint.row_subsample == 0.5 and joint.sketch_outputs == 75
