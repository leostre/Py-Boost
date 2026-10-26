import cupy as cp

from py_boost.multioutput.sketch_decision import SketchDecision, marginal_probabilities


def _weighted_sample_without_replacement(
    probabilities: cp.ndarray,
    size: int,
) -> cp.ndarray:
    """Draw a weighted sample without replacement using Gumbel-top-k.

    CuPy does not implement ``choice(..., replace=False, p=...)``.  Adding
    independent Gumbel noise to log probabilities and keeping the largest
    keys implements the same Plackett-Luce sampling law entirely on the GPU.
    """
    population_size = int(probabilities.size)
    if not 1 <= size <= population_size:
        raise ValueError('sample size must be between 1 and population size')
    if size == population_size:
        return cp.arange(population_size, dtype=cp.uint64)

    positive_count = int(cp.count_nonzero(probabilities > 0).item())
    if positive_count < size:
        raise ValueError(
            'weighted sampling without replacement requires at least as many '
            'positive probabilities as selected elements; use tau > 0'
        )

    uniform = cp.random.random(probabilities.shape)
    tiny = cp.finfo(uniform.dtype).tiny
    uniform = cp.clip(uniform, tiny, 1 - cp.finfo(uniform.dtype).eps)
    gumbel = -cp.log(-cp.log(uniform))
    keys = cp.where(probabilities > 0, cp.log(probabilities) + gumbel, -cp.inf)
    return cp.argsort(keys)[-size:].astype(cp.uint64)


def sample_svd_sketch_heuristic(tensor: cp.ndarray, **kwargs):
    """
    Samples rows and columns using SVD sketching with smooth spectral thresholding.
    Applies smooth thresholding to singular values to enhance tail sensitivity during sketching.

    The smooth thresholding transforms singular values as:

    .. math::
        \sigma_i' = \sqrt{\sigma_i^2 + \lambda_i^2}

    where the adaptive threshold is:

    .. math::
        \lambda_i = \lambda_{\max} \cdot \left(\frac{n - i + 1}{n}\right)^\gamma

    and :math:`\lambda_{\max} = \text{lambda_max_ratio} \cdot \sigma_1`.

    Args:
        tensor: Input tensor of shape (m, n).
        subsample: Fraction of rows to sample (0, 1].
        sketch_outputs: Number of columns to sample.
        lambda_max_ratio: Maximum threshold ratio relative to largest singular value.
        gamma: Curvature parameter for threshold decay.

    Returns:
        tuple: (row_indices, column_indices) as cupy arrays.
    """
    lambda_max_ratio = kwargs.get('lambda_max_ratio', 0.01)
    gamma = kwargs.get('gamma', 1.0)
    sketch_outputs = kwargs.get('sketch_outputs', 1)
    subsample = kwargs.get('subsample', 0.5)

    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    U, s, Vh = cp.linalg.svd(tensor, full_matrices=False)

    n = len(s)
    lambda_max = lambda_max_ratio * max(s)  # scale by largest singular value
    i_values = cp.arange(n, 0, -1)
    lambda_i = lambda_max * (i_values / n) ** gamma
    s_lifted = cp.sqrt(s ** 2 + lambda_i ** 2)

    s_diag_root = cp.diag(cp.sqrt(s_lifted))

    row_norms = cp.linalg.norm(U @ s_diag_root, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_norms = cp.linalg.norm(s_diag_root @ Vh, axis=0)
    col_indexer = cp.sort(cp.argsort(col_norms)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_svd_sketch(tensor: cp.ndarray, **kwargs):
    """SVD sketch by row/column leverage scores

    Args:
        tensor: cp.ndarray, shape (m, n)
        subsample: float, row fraction, default 0.5
        sketch_outputs: int, number of columns to keep, default 1

    Returns:
        cp.ndarray, row indices
        cp.ndarray, column indices
    """
    sketch_outputs = kwargs.get('sketch_outputs', 1)
    subsample = kwargs.get('subsample', 0.5)
    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    U, s, Vh = cp.linalg.svd(tensor, full_matrices=False)
    s_diag_root = cp.diag(cp.sqrt(s))

    row_norms = cp.linalg.norm(U @ s_diag_root, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_norms = cp.linalg.norm(s_diag_root @ Vh, axis=0)
    col_indexer = cp.sort(cp.argsort(col_norms)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_topk_sketch(tensor: cp.ndarray, **kwargs):
    """Greedy rows and outputs from article-defined marginal probabilities.

    Args:
        tensor: cp.ndarray, shape (m, n)
        subsample: float, row fraction, default 0.5
        sketch_outputs: int, number of columns to keep, default 1

    Returns:
        cp.ndarray, row indices
        cp.ndarray, column indices
    """
    sketch_outputs = kwargs.get('sketch_outputs', 1)
    subsample = kwargs.get('subsample', 0.5)
    k_row = min(tensor.shape[0], max(1, int(tensor.shape[0] * subsample)))
    k_col = min(tensor.shape[1], max(1, sketch_outputs))

    tau = kwargs.get('tau', 0.0)
    row_probs, output_probs = marginal_probabilities(tensor, tau=tau)
    row_indexer = cp.sort(cp.argsort(row_probs)[-k_row:]).astype(cp.uint64)

    col_indexer = cp.sort(cp.argsort(output_probs)[-k_col:]).astype(cp.uint64)

    return SketchDecision(row_indexer, col_indexer, row_probs, output_probs)


def sample_random_sampling_sketch(tensor: cp.ndarray, **kwargs):
    """Sample row/output marginals independently and without replacement.

    Args:
        tensor: cp.ndarray, shape (m, n)
        subsample: float, row fraction, default 0.5
        sketch_outputs: int, number of columns to keep, default 1

    Returns:
        cp.ndarray, row indices
        cp.ndarray, column indices
    """
    sketch_outputs = kwargs.get('sketch_outputs', 1)
    subsample = kwargs.get('subsample', 0.5)
    k_row = min(tensor.shape[0], max(1, int(tensor.shape[0] * subsample)))
    k_col = min(tensor.shape[1], max(1, sketch_outputs))
    tau = kwargs.get('tau', 1e-3)

    row_probs, output_probs = marginal_probabilities(tensor, tau=tau)
    row_indexer = cp.sort(
        _weighted_sample_without_replacement(row_probs, k_row)
    )
    col_indexer = cp.sort(
        _weighted_sample_without_replacement(output_probs, k_col)
    )

    return SketchDecision(row_indexer, col_indexer, row_probs, output_probs)


def sample_random_projection_sketch(tensor: cp.ndarray, **kwargs):
    """Random projection sketch for rows; top columns by mean squared weight

    Args:
        tensor: cp.ndarray, shape (m, n)
        subsample: float, row fraction, default 0.5
        sketch_outputs: int, projection width and column count, default 1

    Returns:
        cp.ndarray, row indices
        cp.ndarray, column indices
    """
    sketch_outputs = kwargs.get('sketch_outputs', 1)
    subsample = kwargs.get('subsample', 0.5)
    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    P = cp.random.randn(tensor.shape[1], k_col, dtype=cp.float32)
    P /= cp.sqrt(k_col)
    projected = cp.dot(tensor, P)

    row_norms = cp.linalg.norm(projected, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_weights = (tensor ** 2).mean(axis=0)
    col_indexer = cp.sort(cp.argsort(col_weights)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


# svd_heuristic, svd, topk, rand, proj -> sketch callables for GradHessHistory
SUPPORTED_SKETCH_METHODS = {
    'svd_heuristic': sample_svd_sketch_heuristic,
    'svd': sample_svd_sketch,
    'topk': sample_topk_sketch,
    'rand': sample_random_sampling_sketch,
    'proj': sample_random_projection_sketch,
}
