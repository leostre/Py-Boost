import ctypes
import inspect
import logging

import cupy as cp
from py_boost.multioutput.sketching import GradSketch


class GradHessHistory(GradSketch):
    """Callback that accumulates grads/hess, schedules and applies Fedcore approximation."""

    def __init__(self, history_period: int = 10, derivative_threshold: float = 0.1, **kwargs):
        skecth_method = kwargs.get('skecth_method', 'topk')
        sketch_params = kwargs.get('sketch_params', {})
        sketch_outputs = kwargs.get('sketch_outputs', 1)
        match skecth_method:
            case 'filter':
                self.sketch = FilterSketch(sketch_outputs, **sketch_params)
            case 'svd':
                self.sketch = SVDSketch(sketch_outputs, **sketch_params)
            case 'topk':
                self.sketch = TopOutputsSketch(sketch_outputs)
            case 'rand':
                self.sketch = RandomSamplingSketch(sketch_outputs, **sketch_params)
            case 'proj':
                self.sketch = RandomProjectionSketch(sketch_outputs, **sketch_params)
            case _:
                raise ValueError('Unknown sketching strategy')

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

    # TODO: add sketch integration (self.sketch_method)
    def get_indexers(self, tensor: cp.ndarray, top_fraction: float):
        """
        Compute row and column indexers based on SVD decomposition and norm analysis.
        
        Performs truncated SVD on the input tensor and selects the top-k rows and columns
        based on their norms in the factorized space. The method:
        1. Computes SVD: $tensor = U \\cdot \\Sigma \\cdot V^H$
        2. Computes row norms from $U \\cdot \\sqrt{\\Sigma}$
        3. Computes column norms from $\\sqrt{\\Sigma} \\cdot V^H$
        4. Selects top fraction of rows and columns based on these norms

        Args:
            tensor: Input tensor of shape (n, m) to analyze.
            top_fraction: Fraction of top rows/columns to select (0.0 to 1.0).
                        For example, 0.1 selects top 10% of rows and columns.

        Returns:
            Tuple of (row_indexer, col_indexer) where:
                - row_indexer: Indices of selected rows, shape (k_row,) where 
                            k_row = max(1, int(n * top_fraction))
                - col_indexer: Indices of selected columns, shape (k_col,) where 
                            k_col = max(1, int(m * top_fraction))
        """
        U, s, Vh = cp.linalg.svd(tensor, full_matrices=False)
        s_diag_root = cp.diag(cp.sqrt(s))

        row_norms = cp.linalg.norm(U @ s_diag_root, axis=1)
        k_row = max(1, int(len(row_norms) * top_fraction))
        row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

        col_norms = cp.linalg.norm(s_diag_root @ Vh, axis=0)
        k_col = max(1, int(len(col_norms) * top_fraction))
        col_indexer = cp.sort(cp.argsort(col_norms)[-k_col:]).astype(cp.uint64)

        return row_indexer, col_indexer

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
            row_indexer, col_indexer = self.get_indexers(grad, top_fraction=0.5)
            self._set_indexers(row_indexer=row_indexer, col_indexer=col_indexer)
            self.use_approximation = False
        return grad, hess


class WeightedHistorySampling(GradHessHistory):
    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        if self.use_approximation:
            weights = self.get_weights(grad, hess)
            row_indexer, col_indexer = self.get_indexers(weights, top_fraction=0.5)
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
        mean_grad = cp.mean(self._hist_grad, axis=0)
        mean_hess = cp.mean(self._hist_hess, axis=0)

        diff = (grad - mean_grad) * (hess - mean_hess)

        # safety threshold for near-zero values of diffs
        safe_diff = cp.maximum(cp.abs(diff), 1e-10) * cp.sign(diff)
        weights = cp.reciprocal(safe_diff)

        return weights
