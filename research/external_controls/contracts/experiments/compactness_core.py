"""CPU-only model-complexity accounting and validation-only selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def normalize_multiclass_probabilities(probabilities, clip_floor=1e-6):
    """Normalize only the bounded mass drift caused by Py-Boost component clipping."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.all(np.isfinite(p)):
        raise ValueError('finite multiclass probability matrix required')
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError('probability outside [0, 1]')
    if not math.isfinite(clip_floor) or not 0 <= clip_floor < .5:
        raise ValueError('invalid clipping floor')
    totals = p.sum(axis=1, keepdims=True)
    # Clipping each component can change the total mass by at most d * floor.
    # The extra 1e-6 accommodates float32 softmax summation roundoff.
    if np.any(np.abs(totals - 1) > p.shape[1] * clip_floor + 1e-6):
        raise ValueError('mass drift exceeds component-clipping bound')
    return p / totals


@dataclass(frozen=True)
class Complexity:
    leaf_coefficients: int
    split_nodes: int
    leaves: int
    physical_trees: int
    stored_leaf_slots: int
    inference_payload_bytes: int


def count_tree_payload(test_format, offsets, values, output_groups) -> Complexity:
    """Count reachable coefficients, not padding or the single-leaf pseudo split."""
    nodes = np.asarray(test_format).reshape(-1, 4)
    offsets = np.asarray(offsets, dtype=np.int64)
    values = np.asarray(values)
    groups = np.asarray(output_groups, dtype=np.int64)
    if values.ndim != 2 or groups.ndim != 1 or values.shape[1] != len(groups):
        raise ValueError('leaf values and output groups disagree')
    if len(offsets) == 0 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError('invalid group offsets')
    if np.any(groups < 0) or np.any(groups >= len(offsets)):
        raise ValueError('output refers to an unknown group')
    total_leaves = total_splits = coefficients = 0
    for group, start in enumerate(offsets):
        end = int(offsets[group + 1]) if group + 1 < len(offsets) else len(nodes)
        if start < 0 or end > len(nodes) or start >= end:
            raise ValueError('invalid group extent')
        visited, leaf_ids, active = set(), set(), set()

        def visit(index):
            nonlocal total_splits
            if index in active:
                raise ValueError('cycle in tree structure')
            if index in visited:
                return
            if not 0 <= index < end - start:
                raise ValueError('child outside group extent')
            active.add(index)
            _, _, left, right = nodes[start + index]
            if not all(np.isfinite(v) and v == int(v) for v in (left, right)):
                raise ValueError('noninteger child pointer')
            pseudo = left == right and left < 0
            if pseudo and (index != 0 or end - start != 1):
                raise ValueError('pseudo split only allowed for a single-leaf group')
            total_splits += int(not pseudo)
            for pointer in (int(left), int(right)):
                if pointer < 0:
                    leaf_id = -pointer - 1
                    if leaf_id >= values.shape[0]:
                        raise ValueError('leaf value outside allocation')
                    leaf_ids.add(leaf_id)
                else:
                    visit(pointer)
            active.remove(index)
            visited.add(index)

        visit(0)
        if len(visited) != end - start:
            raise ValueError('unreachable formatted split nodes')
        total_leaves += len(leaf_ids)
        coefficients += len(leaf_ids) * int(np.count_nonzero(groups == group))
    payload = sum(np.asarray(a).nbytes for a in (test_format, offsets, values, output_groups))
    return Complexity(coefficients, total_splits, total_leaves, len(offsets),
                      int(values.size), int(payload))


@dataclass(frozen=True)
class ValidationCandidate:
    candidate_id: str
    parameters: int
    validation_loss: float

    def __post_init__(self):
        if self.parameters <= 0 or not math.isfinite(self.validation_loss) or self.validation_loss <= 0:
            raise ValueError('candidate needs positive parameters and finite positive loss')


def select_compact(candidates, tolerance):
    """Reference and compact model use validation losses only, never test scores."""
    if not candidates or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('nonempty candidates and finite nonnegative tolerance required')
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate identities')
    reference = min(candidates, key=lambda c: (c.validation_loss, c.parameters, c.candidate_id))
    eligible = [c for c in candidates if c.validation_loss <= (1 + tolerance) * reference.validation_loss]
    compact = min(eligible, key=lambda c: (c.parameters, c.validation_loss, c.candidate_id))
    return reference, compact


def validate_partition(train, validation, test, n_rows):
    """Every source-relative subset row belongs to exactly one partition."""
    parts = [np.asarray(p, dtype=np.int64) for p in (train, validation, test)]
    joined = np.concatenate(parts)
    if any(len(p) == 0 for p in parts) or len(joined) != n_rows:
        raise ValueError('empty or incomplete split')
    if not np.array_equal(np.sort(joined), np.arange(n_rows)):
        raise ValueError('overlap, missing row, or row outside dataset')
