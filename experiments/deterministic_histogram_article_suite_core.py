"""Pure comparison helpers for the article-dataset histogram suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def analyze_layout_dispatch(
    policy: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate every observed layout decision without requiring a full tree."""

    observed = sorted({str(record["selected_layout"]) for record in records})
    policies = sorted({str(record["policy"]) for record in records})
    nodes = sorted({int(record["nodes"]) for record in records})
    policy_exact = bool(records) and policies == [policy]
    if policy == "perf1":
        decisions_exact = all(
            str(record["selected_layout"]) == "perf1" for record in records
        )
    elif policy == "hybrid":
        decisions_exact = all(
            str(record["selected_layout"])
            == ("perf1" if int(record["nodes"]) <= 8 else "perf2")
            for record in records
        )
    else:
        raise ValueError(f"unsupported histogram layout policy: {policy}")
    return {
        "gate_passed": policy_exact and decisions_exact,
        "policy_exact": policy_exact,
        "decisions_exact": decisions_exact,
        "observed_layouts": observed,
        "observed_node_counts": nodes,
        "both_layouts_observed": observed == ["perf1", "perf2"],
    }


def aggregate_suite_status(dataset_reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Aggregate hard correctness and descriptive dispatch coverage."""

    failed = sorted(
        name
        for name, report in dataset_reports.items()
        if not bool(report["correctness_gate_passed"])
    )
    both_layouts = sorted(
        name
        for name, report in dataset_reports.items()
        if bool(report["dispatch_coverage"]["both_layouts_observed"])
    )
    return {
        "correctness_gate_passed": not failed and bool(dataset_reports),
        "failed_datasets": failed,
        "datasets_exercising_both_layouts": both_layouts,
        "dataset_count": len(dataset_reports),
    }
