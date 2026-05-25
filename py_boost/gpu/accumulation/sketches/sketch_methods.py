import cupy as cp


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
    """Top-k rows by L2 norm and top columns by mean squared value

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

    row_norms = cp.linalg.norm(tensor, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_weights = (tensor ** 2).mean(axis=0)
    col_indexer = cp.sort(cp.argsort(col_weights)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_random_sampling_sketch(tensor: cp.ndarray, **kwargs):
    """Importance sampling of rows and columns with uniform smoothing

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

    row_weights = cp.linalg.norm(tensor, axis=1) ** 2 + 1e-3
    row_probs = row_weights / row_weights.sum()

    smooth = 0.1
    row_probs = smooth * cp.ones_like(row_probs) / tensor.shape[0] + (1 - smooth) * row_probs
    row_indexer = cp.sort(cp.random.choice(cp.arange(tensor.shape[0]), size=k_row, 
                                           replace=True, p=row_probs)).astype(cp.uint64)

    col_weights = (tensor ** 2).mean(axis=0) + 1e-3
    col_probs = col_weights / col_weights.sum()
    col_probs = smooth * cp.ones_like(col_probs) / tensor.shape[1] + (1 - smooth) * col_probs
    col_indexer = cp.sort(cp.random.choice(cp.arange(tensor.shape[1]), size=k_col, 
                                           replace=True, p=col_probs)).astype(cp.uint64)

    return row_indexer, col_indexer


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
