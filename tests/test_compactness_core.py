"""Contracts: reachable coefficients, padding invariance, validation-only choice."""
import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.compactness_core import (
    ValidationCandidate, count_tree_payload, select_compact, validate_partition,
    normalize_multiclass_probabilities,
)


@given(outputs=st.integers(1, 40), padding=st.integers(0, 40))
def test_padding_does_not_change_learned_coefficients(outputs, padding):
    result = count_tree_payload(np.array([1, 0, -1, -2], dtype=np.float32),
                                np.array([0], dtype=np.int32),
                                np.zeros((2 + padding, outputs), dtype=np.float32),
                                np.zeros(outputs, dtype=np.int32))
    assert result.leaf_coefficients == 2 * outputs
    assert result.split_nodes == 1
    assert result.leaves == 2
    assert result.stored_leaf_slots == (2 + padding) * outputs


def test_single_leaf_pseudo_node_is_not_a_split():
    result = count_tree_payload([1, 0, -1, -1], [0], np.zeros((4, 3)), [0, 0, 0])
    assert (result.leaf_coefficients, result.split_nodes, result.leaves) == (3, 0, 1)


def test_output_groups_are_counted_separately():
    result = count_tree_payload([1, 0, -1, -2, 1, 0, -1, -1], [0, 1],
                                np.zeros((4, 3)), [0, 0, 1])
    assert (result.leaf_coefficients, result.split_nodes, result.leaves) == (5, 1, 3)


@pytest.mark.parametrize('nodes', [[1, 0, 0, -1], [1, 0, 9, -1], [1, 0, -9, -1]])
def test_invalid_structure_is_rejected(nodes):
    with pytest.raises(ValueError):
        count_tree_payload(nodes, [0], np.zeros((2, 1)), [0])


@given(parameters=st.lists(st.integers(1, 10000), min_size=1, max_size=15),
       epsilon=st.floats(0, .5, allow_nan=False, allow_infinity=False))
def test_selection_obeys_constraint_and_tolerance_monotonicity(parameters, epsilon):
    candidates = [ValidationCandidate(str(i), p, 1 + i / 20) for i, p in enumerate(parameters)]
    reference, chosen = select_compact(candidates, epsilon)
    assert reference.candidate_id == '0'
    eligible_parameters = [p for i, p in enumerate(parameters) if 1 + i / 20 <= 1 + epsilon]
    assert chosen.parameters == min(eligible_parameters)
    assert chosen.validation_loss <= (1 + epsilon) * reference.validation_loss
    assert select_compact(list(reversed(candidates)), epsilon) == (reference, chosen)
    assert select_compact(candidates, epsilon + .1)[1].parameters <= chosen.parameters


def test_partition_includes_each_row_once():
    validate_partition([0, 2], [1], [3], 4)
    with pytest.raises(ValueError):
        validate_partition([0, 1], [1], [3], 4)


def test_selection_cannot_read_test_loss():
    assert set(ValidationCandidate.__dataclass_fields__) == {
        'candidate_id', 'parameters', 'validation_loss'}
    with pytest.raises(ValueError):
        select_compact([ValidationCandidate('duplicate', 1, 1),
                        ValidationCandidate('duplicate', 2, 2)], .05)


def test_clipped_softmax_mass_is_bounded_and_normalized():
    # Exact softmax of [0, -100, ...], followed by the library's component clip.
    logits = np.array([[0.] + [-100.] * 99])
    exact = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    clipped = np.clip(exact, 1e-6, 1 - 1e-6)
    assert not np.allclose(clipped.sum(axis=1), 1, atol=1e-5)
    normalized = normalize_multiclass_probabilities(clipped)
    assert np.allclose(normalized.sum(axis=1), 1, atol=1e-14)
    assert np.array_equal(normalized.argmax(axis=1), clipped.argmax(axis=1))


@pytest.mark.parametrize('bad', [[[.9, .9]], [[-0.1, 1.1]], [[np.nan, 1]], [[0., 0.]]])
def test_normalization_does_not_hide_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        normalize_multiclass_probabilities(bad)
