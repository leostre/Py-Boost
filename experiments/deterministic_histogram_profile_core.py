"""Pure aggregation for H-HIST-PROFILE-1 timing records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import numpy as np


COMPONENT_FIELDS = (
    'allocation_ms',
    'row_filter_ms',
    'row_to_host_ms',
    'cpu_sort_ms',
    'row_to_device_ms',
    'kernel_ms',
    'cumsum_ms',
    'sibling_subtraction_ms',
)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def analyze_histogram_profile(
    records: Sequence[Mapping[str, object]],
    train_seconds: float,
    expected_nodes: Sequence[int] = (1, 2, 4, 8, 16),
    dominance_threshold: float = 0.35,
) -> dict[str, object]:
    """Aggregate synchronized histogram timings and apply the decision rule."""

    threshold = float(dominance_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError('dominance threshold must be in (0, 1)')
    if not math.isfinite(float(train_seconds)) or float(train_seconds) <= 0:
        raise ValueError('training time must be finite and positive')

    invalid_records = []
    normalized = []
    for index, raw in enumerate(records):
        missing = [
            field for field in (*COMPONENT_FIELDS, 'total_ms', 'kind', 'nodes')
            if field not in raw
        ]
        values = {
            field: float(raw.get(field, float('nan')))
            for field in (*COMPONENT_FIELDS, 'total_ms')
        }
        invalid_values = [
            field for field, value in values.items()
            if not math.isfinite(value) or value < 0
        ]
        if missing or invalid_values:
            invalid_records.append({
                'index': index,
                'missing': missing,
                'invalid_values': invalid_values,
            })
            continue
        normalized.append({
            **raw,
            **values,
            'kind': str(raw['kind']),
            'nodes': int(raw['nodes']),
        })

    required_kinds = {'diagnostic', 'structure'}
    nodes_by_kind = {
        kind: sorted({record['nodes'] for record in normalized if record['kind'] == kind})
        for kind in required_kinds
    }
    expected = sorted(int(value) for value in expected_nodes)
    coverage_gate_passed = (
        bool(normalized)
        and not invalid_records
        and all(nodes_by_kind[kind] == expected for kind in required_kinds)
    )

    component_totals = {
        field: sum(record[field] for record in normalized)
        for field in COMPONENT_FIELDS
    }
    total_ms = sum(record['total_ms'] for record in normalized)
    accounted_ms = sum(component_totals.values())
    residual_ms = total_ms - accounted_ms
    canonicalization_ms = sum(
        component_totals[field]
        for field in ('row_to_host_ms', 'cpu_sort_ms', 'row_to_device_ms')
    )
    postprocessing_ms = sum(
        component_totals[field]
        for field in ('cumsum_ms', 'sibling_subtraction_ms')
    )
    category_totals = {
        'canonicalization': canonicalization_ms,
        'kernel': component_totals['kernel_ms'],
        'postprocessing': postprocessing_ms,
    }
    category_shares = {
        key: (value / total_ms if total_ms else 0.0)
        for key, value in category_totals.items()
    }
    dominant_component = max(category_shares, key=category_shares.get, default=None)
    if dominant_component is None or category_shares[dominant_component] < threshold:
        next_action = 'controlled_microbenchmark'
        dominant_component = None
    else:
        next_action = f'optimize_{dominant_component}'

    grouped = defaultdict(list)
    for record in normalized:
        grouped[(record['kind'], record['nodes'])].append(record)
    by_path_and_nodes = []
    for (kind, nodes), group in sorted(grouped.items()):
        canonical = [
            sum(record[field] for field in (
                'row_to_host_ms', 'cpu_sort_ms', 'row_to_device_ms'
            ))
            for record in group
        ]
        postprocessing = [
            record['cumsum_ms'] + record['sibling_subtraction_ms']
            for record in group
        ]
        by_path_and_nodes.append({
            'kind': kind,
            'nodes': nodes,
            'calls': len(group),
            'total_ms_median': _percentile(
                [record['total_ms'] for record in group], 50
            ),
            'total_ms_p95': _percentile(
                [record['total_ms'] for record in group], 95
            ),
            'canonicalization_ms_median': _percentile(canonical, 50),
            'kernel_ms_median': _percentile(
                [record['kernel_ms'] for record in group], 50
            ),
            'postprocessing_ms_median': _percentile(postprocessing, 50),
        })

    return {
        'scientific_status': (
            'profile_complete' if coverage_gate_passed else 'profile_incomplete'
        ),
        'coverage_gate_passed': coverage_gate_passed,
        'record_count': len(records),
        'valid_record_count': len(normalized),
        'invalid_records': invalid_records,
        'nodes_by_kind': nodes_by_kind,
        'component_totals_ms': component_totals,
        'category_totals_ms': category_totals,
        'category_shares_of_total': category_shares,
        'profiled_histogram_total_ms': total_ms,
        'accounted_component_ms': accounted_ms,
        'accounted_fraction': accounted_ms / total_ms if total_ms else 0.0,
        'residual_ms': residual_ms,
        'residual_fraction': residual_ms / total_ms if total_ms else 0.0,
        'training_seconds': float(train_seconds),
        'profiled_histogram_fraction_of_training': (
            total_ms / (float(train_seconds) * 1000.0)
        ),
        'dominance_threshold': threshold,
        'dominant_component': dominant_component,
        'next_action': next_action,
        'by_path_and_nodes': by_path_and_nodes,
    }
