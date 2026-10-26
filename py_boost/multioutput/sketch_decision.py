"""Pure data structures and helpers for history-based sketching decisions."""

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple


@dataclass(frozen=True)
class SketchDecision:
    """Indices selected for tree-structure search.

    Row indices address training objects. Output indices address gradient and
    Hessian columns. Feature indices deliberately do not belong to this object:
    feature sampling is an independent tree-building concern.
    """

    row_indices: Any
    output_indices: Any
    row_probabilities: Optional[Any] = None
    output_probabilities: Optional[Any] = None

    def __iter__(self) -> Iterator[Any]:
        """Preserve tuple unpacking used by the legacy sketch interface."""

        yield self.row_indices
        yield self.output_indices


@dataclass(frozen=True)
class StructureIndexers:
    """Resolved axes used during tree-structure search."""

    row_indices: Any
    feature_indices: Any
    output_indices: Optional[Any] = None


def resolve_structure_indexers(
    default_row_indices: Any,
    feature_indices: Any,
    decision: Optional[SketchDecision],
) -> StructureIndexers:
    """Merge a sketch decision without allowing output indices to replace features."""

    if decision is None:
        return StructureIndexers(default_row_indices, feature_indices)

    return StructureIndexers(
        row_indices=decision.row_indices,
        feature_indices=feature_indices,
        output_indices=decision.output_indices,
    )


def apply_output_indices(grad: Any, hess: Any, output_indices: Optional[Any]) -> Tuple[Any, Any]:
    """Restrict structure-search statistics to selected output columns."""

    if output_indices is None:
        return grad, hess

    sketched_grad = grad[:, output_indices]
    sketched_hess = hess if hess.shape[1] == 1 else hess[:, output_indices]
    return sketched_grad, sketched_hess


def resolve_leaf_row_indices(
    structure_row_indices: Any,
    all_row_indices: Any,
    full_leaf_values: bool,
) -> Any:
    """Choose rows for leaf-value estimation independently of structure search."""

    return all_row_indices if full_leaf_values else structure_row_indices


def marginal_probabilities(weights: Any, tau: float = 0.0) -> Tuple[Any, Any]:
    """Build article-defined row and output marginals from a weight matrix."""

    if tau < 0:
        raise ValueError("tau must be non-negative")
    if len(weights.shape) != 2 or min(weights.shape) < 1:
        raise ValueError("weights must be a non-empty two-dimensional matrix")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")

    row_scores = weights.sum(axis=1) + tau
    output_scores = weights.sum(axis=0) + tau

    row_total = row_scores.sum()
    output_total = output_scores.sum()
    if float(row_total) == 0:
        row_scores = row_scores * 0 + 1
        row_total = row_scores.sum()
    if float(output_total) == 0:
        output_scores = output_scores * 0 + 1
        output_total = output_scores.sum()

    return row_scores / row_total, output_scores / output_total
