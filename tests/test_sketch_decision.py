"""Regression and invariant tests for history-sketch axis handling."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from hypothesis import given, strategies as st


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "multioutput"
    / "sketch_decision.py"
)
SPEC = importlib.util.spec_from_file_location("sketch_decision", MODULE_PATH)
sketch_decision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sketch_decision
SPEC.loader.exec_module(sketch_decision)

SketchDecision = sketch_decision.SketchDecision
apply_output_indices = sketch_decision.apply_output_indices
resolve_leaf_row_indices = sketch_decision.resolve_leaf_row_indices
resolve_structure_indexers = sketch_decision.resolve_structure_indexers
marginal_probabilities = sketch_decision.marginal_probabilities


def test_output_indices_never_replace_feature_indices():
    """Regression: selecting output 1 must not reduce three input features."""

    default_rows = np.arange(4, dtype=np.uint64)
    feature_indices = np.arange(3, dtype=np.uint64)
    decision = SketchDecision(
        row_indices=np.array([0, 2], dtype=np.uint64),
        output_indices=np.array([1], dtype=np.uint64),
    )

    resolved = resolve_structure_indexers(default_rows, feature_indices, decision)

    np.testing.assert_array_equal(resolved.row_indices, [0, 2])
    np.testing.assert_array_equal(resolved.feature_indices, [0, 1, 2])
    np.testing.assert_array_equal(resolved.output_indices, [1])


@given(
    n_rows=st.integers(min_value=1, max_value=50),
    n_features=st.integers(min_value=1, max_value=50),
    n_outputs=st.integers(min_value=1, max_value=30),
    row_data=st.data(),
    output_data=st.data(),
)
def test_axis_separation_is_preserved_for_generated_shapes(
    n_rows,
    n_features,
    n_outputs,
    row_data,
    output_data,
):
    row_mask = row_data.draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
    output_mask = output_data.draw(
        st.lists(st.booleans(), min_size=n_outputs, max_size=n_outputs)
    )
    selected_rows = np.flatnonzero(row_mask)
    selected_outputs = np.flatnonzero(output_mask)
    if selected_rows.size == 0:
        selected_rows = np.array([0])
    if selected_outputs.size == 0:
        selected_outputs = np.array([0])
    default_rows = np.arange(n_rows, dtype=np.uint64)
    features = np.arange(n_features, dtype=np.uint64)
    decision = SketchDecision(
        row_indices=np.asarray(selected_rows, dtype=np.uint64),
        output_indices=np.asarray(selected_outputs, dtype=np.uint64),
    )

    resolved = resolve_structure_indexers(default_rows, features, decision)

    np.testing.assert_array_equal(resolved.feature_indices, features)
    np.testing.assert_array_equal(resolved.row_indices, decision.row_indices)
    np.testing.assert_array_equal(resolved.output_indices, decision.output_indices)


@given(
    n_rows=st.integers(min_value=1, max_value=30),
    n_outputs=st.integers(min_value=1, max_value=20),
    output_data=st.data(),
)
def test_output_sketch_reduces_gradient_and_matching_hessian_columns(
    n_rows,
    n_outputs,
    output_data,
):
    output_mask = output_data.draw(
        st.lists(st.booleans(), min_size=n_outputs, max_size=n_outputs)
    )
    selected_outputs = np.flatnonzero(output_mask)
    if selected_outputs.size == 0:
        selected_outputs = np.array([0])
    output_indices = np.asarray(selected_outputs, dtype=np.uint64)
    grad = np.arange(n_rows * n_outputs).reshape(n_rows, n_outputs)
    hess = grad + 1000

    sketched_grad, sketched_hess = apply_output_indices(grad, hess, output_indices)

    np.testing.assert_array_equal(sketched_grad, grad[:, output_indices])
    np.testing.assert_array_equal(sketched_hess, hess[:, output_indices])


@given(
    n_rows=st.integers(min_value=1, max_value=50),
    row_data=st.data(),
)
def test_full_leaf_estimation_uses_every_training_row(n_rows, row_data):
    row_mask = row_data.draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
    selected_rows = np.flatnonzero(row_mask)
    if selected_rows.size == 0:
        selected_rows = np.array([0])
    structure_rows = np.asarray(selected_rows, dtype=np.uint64)
    all_rows = np.arange(n_rows, dtype=np.uint64)

    full_rows = resolve_leaf_row_indices(structure_rows, all_rows, True)
    sampled_rows = resolve_leaf_row_indices(structure_rows, all_rows, False)

    np.testing.assert_array_equal(full_rows, all_rows)
    np.testing.assert_array_equal(sampled_rows, structure_rows)


def test_single_column_hessian_is_kept_as_structure_proxy():
    grad = np.arange(12).reshape(3, 4)
    hess = np.ones((3, 1))
    outputs = np.array([1, 3], dtype=np.uint64)

    sketched_grad, sketched_hess = apply_output_indices(grad, hess, outputs)

    np.testing.assert_array_equal(sketched_grad, grad[:, outputs])
    assert sketched_hess is hess


def test_article_marginals_are_sums_of_weight_matrix_axes():
    weights = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    row_probabilities, output_probabilities = marginal_probabilities(weights, tau=1.0)

    np.testing.assert_allclose(row_probabilities, np.array([7.0, 16.0]) / 23.0)
    np.testing.assert_allclose(output_probabilities, np.array([6.0, 8.0, 10.0]) / 24.0)


def test_zero_weights_fall_back_to_uniform_marginals():
    row_probabilities, output_probabilities = marginal_probabilities(
        np.zeros((3, 2)),
        tau=0.0,
    )

    np.testing.assert_allclose(row_probabilities, np.full(3, 1 / 3))
    np.testing.assert_allclose(output_probabilities, np.full(2, 1 / 2))


@given(
    n_rows=st.integers(min_value=1, max_value=10),
    n_outputs=st.integers(min_value=1, max_value=10),
    tau=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    data=st.data(),
)
def test_product_distribution_preserves_both_marginals(n_rows, n_outputs, tau, data):
    values = data.draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows * n_outputs,
            max_size=n_rows * n_outputs,
        )
    )
    weights = np.asarray(values).reshape(n_rows, n_outputs)

    row_probabilities, output_probabilities = marginal_probabilities(weights, tau=tau)
    product_distribution = np.outer(row_probabilities, output_probabilities)

    np.testing.assert_allclose(product_distribution.sum(axis=1), row_probabilities)
    np.testing.assert_allclose(product_distribution.sum(axis=0), output_probabilities)
    assert np.isclose(product_distribution.sum(), 1.0)
