import ctypes
import inspect
import logging

import cupy as cp
from py_boost.multioutput.sketching import GradSketch


def sample_svd_sketch(tensor: cp.ndarray, subsample: float, sketch_outputs: int):
    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    U, s, Vh = cp.linalg.svd(tensor, full_matrices=False)
    s_diag_root = cp.diag(cp.sqrt(s))

    row_norms = cp.linalg.norm(U @ s_diag_root, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_norms = cp.linalg.norm(s_diag_root @ Vh, axis=0)
    col_indexer = cp.sort(cp.argsort(col_norms)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_topk_sketch(tensor: cp.ndarray, subsample: float, sketch_outputs: int):
    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    row_norms = cp.linalg.norm(tensor, axis=1)
    row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

    col_weights = (tensor ** 2).mean(axis=0)
    col_indexer = cp.sort(cp.argsort(col_weights)[-k_col:]).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_random_sampling_sketch(tensor: cp.ndarray, subsample: float, sketch_outputs: int):
    k_row = max(1, int(tensor.shape[0] * subsample))
    k_col = max(1, sketch_outputs)

    row_weights = cp.linalg.norm(tensor, axis=1) ** 2 + 1e-3
    row_probs = row_weights / row_weights.sum()

    smooth = 0.1
    row_probs = smooth * cp.ones_like(row_probs) / tensor.shape[0] + (1 - smooth) * row_probs
    row_indexer = cp.sort(cp.random.choice(cp.arange(tensor.shape[0]), size=k_row, 
                                           replace=False, p=row_probs)).astype(cp.uint64)

    col_weights = (tensor ** 2).mean(axis=0) + 1e-3
    col_probs = col_weights / col_weights.sum()
    col_probs = smooth * cp.ones_like(col_probs) / tensor.shape[1] + (1 - smooth) * col_probs
    col_indexer = cp.sort(cp.random.choice(cp.arange(tensor.shape[1]), size=k_col, 
                                           replace=False, p=col_probs)).astype(cp.uint64)

    return row_indexer, col_indexer


def sample_random_projection_sketch(tensor: cp.ndarray, subsample: float, sketch_outputs: int):
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


class GradHessHistory(GradSketch):
    """Callback that accumulates grads/hess, schedules and applies Fedcore approximation."""

    def __init__(self, history_period: int = 10, derivative_threshold: float = 0.1, **kwargs):
        sketch_method = kwargs.get('sketch_method', 'svd')
        self.sketch_params = kwargs.get('sketch_params', {})

        # TODO: move decomposition params to decomposition callback
        match sketch_method:
            case 'svd':
                self.sketch = sample_svd_sketch
            case 'topk':
                self.sketch = sample_topk_sketch
            case 'rand':
                self.sketch = sample_random_sampling_sketch
            case 'proj':
                self.sketch = sample_random_projection_sketch
            case _:
                raise ValueError(f'Unknown sketching strategy {sketch_method}')

        # TODO: move subsampling params to subsampling callback
        self.subsample = kwargs.get('subsample', 0.5)
        self.sketch_outputs = kwargs.get('sketch_outputs', 1)

        self.history_period = int(history_period)
        self.derivative_threshold = derivative_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_approximation = False
        self._hist_grad, self._hist_hess = None, None
        self._current_iteration = 0

    def before_train(self, build_info):
        self.use_approximation = False
        self._hist_grad, self._hist_hess = None, None
        self._current_iteration = 0

    def before_iteration(self, build_info):
        self._current_iteration = build_info['num_iter'] + 1
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')

        # accumulate to history when grads and hesses are available
        if grad is not None and hess is not None:
            self._update_history(grad, hess)
            # check if we should apply approximation based on history
            self.use_approximation = self._scheduler()

    def _update_history(self, grad: cp.ndarray, hess: cp.ndarray):
        if self._hist_grad is None or len(self._hist_grad) < self.history_period:
            self._hist_grad = cp.stack([grad.copy()] * self.history_period)
            self._hist_hess = cp.stack([hess.copy()] * self.history_period)
        else:
            # sliding window: roll and replace the oldest grad and hess
            self._hist_grad = cp.roll(self._hist_grad, -1, axis=0)
            self._hist_hess = cp.roll(self._hist_hess, -1, axis=0)
            self._hist_grad[-1] = grad.copy()
            self._hist_hess[-1] = hess.copy()

    def _gaussian_smooth(self, data: cp.ndarray, sigma: float = 1.0) -> cp.ndarray:
        if len(data) < 3:
            return data

        kernel_size = min(5, len(data))
        x = cp.arange(kernel_size) - (kernel_size - 1) // 2
        kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / cp.sum(kernel)
        smoothed = cp.convolve(data, kernel, mode='same')
        return smoothed

    def _scheduler(self) -> bool:
        """
        Determine if gradient stabilization has occurred to enable approximation.
        Checks if gradient norms have stabilized by analyzing their derivatives:
        1. Compute L2 norms of historical gradients across iterations
        2. Apply Gaussian smoothing to reduce noise in gradient norms
        3. Calculate derivatives of smoothed gradient norms
        4. Check if average absolute derivative is below threshold

        Returns:
            bool: True if gradients have stabilized (low derivatives), enabling 
                approximation. False if more history needed or gradients are 
                still changing significantly.
        """
        # TODO: rewrite as dynamic observers
        if self._hist_grad is None or self._current_iteration < self.history_period:
            return False

        threshold = self.derivative_threshold
        grad_norms = cp.linalg.norm(self._hist_grad, axis=0)
        derivative = cp.gradient(self._gaussian_smooth(grad_norms))

        avg_recent_deriv = cp.mean(cp.abs(derivative))
        return avg_recent_deriv < threshold

    def get_indexers(self, tensor: cp.ndarray):
        """
        Compute row and column indexers based on the sketch method.

        Returns:
            Tuple of (row_indexer, col_indexer) where:
                - row_indexer: Indices of selected rows, shape (k_row,)
                - col_indexer: Indices of selected columns, shape (k_col,)
        """
        return self.sketch(tensor, self.subsample, self.sketch_outputs)

    def _set_indexers(self, row_indexer: cp.ndarray, col_indexer: cp.ndarray) -> None:
        stack = inspect.stack()
        target_method = 'build_tree'

        for frame_info in stack[1:]:
            if frame_info.function == target_method:
                frame = frame_info.frame
                try:
                    frame.f_locals['row_indexer'] = row_indexer
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
                    frame.f_locals['col_indexer'] = col_indexer
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
                except Exception:
                    pass
                finally:
                    break

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        if self.use_approximation:
            row_indexer, col_indexer = self.get_indexers(grad)
            self._set_indexers(row_indexer=row_indexer, col_indexer=col_indexer)
            self.use_approximation = False
        return grad, hess


class WeightedHistorySampling(GradHessHistory):
    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        if self.use_approximation:
            weights = self.get_weights(grad, hess)
            row_indexer, col_indexer = self.get_indexers(weights)
            self._set_indexers(row_indexer=row_indexer, col_indexer=col_indexer)
            self.use_approximation = False
        return grad, hess

    def get_weights(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """
        Compute weights based on differences between current and historical gradients and hessians.
        
        The weight for each element (i,j) is computed as:
            $$w_{i,j} = \\frac{1}{(g_{i,j} - \\mu_{g_{i,j}}) \\cdot (h_{i,j} - \\mu_{h_{i,j}})}$$
        
        where:
            - $g_{i,j}$: current gradient at position (i,j)
            - $\\mu_{g_{i,j}}$: historical mean gradient at position (i,j) 
            - $h_{i,j}$: current hessian at position (i,j)
            - $\\mu_{h_{i,j}}$: historical mean hessian at position (i,j)
        
        Args:
            grad: Current gradient tensor of shape (n, m).
            hess: Current hessian tensor of shape (n, m).

        Returns:
            weights: Weight tensor of shape (n, m).
        """
        def sigmoid(x):
            return 1 / (1 + cp.exp(-x))

        mean_grad = cp.mean(self._hist_grad, axis=0)
        mean_hess = cp.mean(self._hist_hess, axis=0)

        weights = -sigmoid((grad - mean_grad) * (hess - mean_hess))

        return weights
