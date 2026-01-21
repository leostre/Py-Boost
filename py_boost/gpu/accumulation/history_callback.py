import ctypes
import inspect
import logging

import cupy as cp
from py_boost.gpu.accumulation.sketches.sketch_methods import (sample_random_projection_sketch,
                                                               sample_random_sampling_sketch,
                                                               sample_svd_sketch,
                                                               sample_svd_sketch_heuristic,
                                                               sample_topk_sketch)
from py_boost.multioutput.sketching import GradSketch


class GradHessHistory(GradSketch):
    """Callback that accumulates grads/hess, schedules and applies Fedcore approximation."""

    def __init__(self, history_period: int = 10, derivative_threshold: float = 0.1, **kwargs):
        sketch_method = kwargs.get('sketch_method', 'svd')

        # TODO: move decomposition params to decomposition callback
        match sketch_method:
            case 'svd_heuristic':
                self.sketch = sample_svd_sketch_heuristic
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
        # TODO: leave only sketch_params dict dep.injection, subsample 
        self.sketch_params = kwargs.get('sketch_params', {'subsample': self.subsample,
                                                          'sketch_outputs': self.sketch_outputs})

        self.history_period = int(history_period)
        self.stabilization_window = int(kwargs.get('stabilization_window', history_period))
        self.smoothing_alpha = kwargs.get('smoothing_alpha', 0.1 ** (1 / 9))
        self.derivative_threshold = derivative_threshold
        self.require_consecutive = kwargs.get('require_consecutive', False)
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

        if data.ndim > 1:
            data = data.flatten()

        data = cp.ascontiguousarray(data)
        if cp.any(cp.isnan(data)) or cp.any(cp.isinf(data)):
            return data

        kernel_size = min(5, len(data))
        x = cp.arange(kernel_size, dtype=cp.float32) - (kernel_size - 1) // 2
        kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / cp.sum(kernel)
        smoothed = cp.convolve(data, kernel, mode='same')
        return smoothed

    def _exponential_smooth(self, data: cp.ndarray, alpha: float) -> cp.ndarray:
        alpha = max(0.01, min(1.0, alpha))

        smoothed = cp.empty_like(data)
        smoothed[0] = data[0]
        for i in range(1, data.shape[0]):
            smoothed[i] = (1 - alpha) * data[i] + alpha * smoothed[i-1]

        return smoothed

    def _scheduler(self) -> bool:
        """
        Determine if gradient stabilization has occurred to enable approximation.
        Checks if gradient norms have stabilized by analyzing their derivatives:
        1. Compute L2 norm across outputs for each sample at each iteration
        2. Apply exponential smoothing along iteration dimension (axis=0)
        3. Calculate derivatives of smoothed gradient norms
        4. Check if average absolute derivative is below threshold

        Returns:
            bool: True if gradients have stabilized (low derivatives), enabling 
                approximation. False if more history needed or gradients are 
                still changing significantly.
        """
        # TODO: rewrite as dynamic observers
        min_history = max(self.history_period, self.stabilization_window)
        if (self._hist_grad is None or
            self._current_iteration < min_history or
            self._hist_grad.shape[0] < min_history):
            return False

        try:
            # # _hist_grad shape: (history_period, n_samples, n_outputs)
            # grad_norms = cp.linalg.norm(self._hist_grad.reshape(self._hist_grad.shape[0], -1), axis=1)
            # smoothed = self._gaussian_smooth(grad_norms)
            # derivative = cp.gradient(smoothed)
            # avg_recent_deriv = cp.mean(cp.abs(derivative))
            # return avg_recent_deriv < threshold
            # self._hist_grad = cp.ascontiguousarray(self._hist_grad)

            # grad_norms = cp.linalg.norm(self._hist_grad, axis=0)
            # derivative = cp.gradient(grad_norms)
            # avg_recent_deriv = cp.mean(cp.abs(derivative[0]))

            # _hist_grad shape: (history_period, n_samples, n_outputs)
            grad_norms = cp.linalg.norm(self._hist_grad, axis=2)  # (history_period, n_samples)
            smoothed_norms = self._exponential_smooth(grad_norms, self.smoothing_alpha)

            derivative = cp.gradient(smoothed_norms, axis=0)
            abs_derivative = cp.abs(derivative)
            recent_derivatives = abs_derivative[-self.stabilization_window:]

            if self.require_consecutive:
                stabilization_detected = cp.all(recent_derivatives < self.derivative_threshold)
            else:
                avg_abs_deriv_per_iteration = cp.mean(cp.abs(recent_derivatives), axis=1)
                avg_recent_deriv = cp.mean(avg_abs_deriv_per_iteration)
                stabilization_detected = avg_recent_deriv < self.derivative_threshold

            return stabilization_detected
        except Exception as e:
            self.logger.warning(f"Error in scheduler, disabling approximation: {e}")
            return False

    def get_indexers(self, tensor: cp.ndarray):
        """
        Compute row and column indexers based on the sketch method.

        Returns:
            Tuple of (row_indexer, col_indexer) where:
                - row_indexer: Indices of selected rows, shape (k_row,)
                - col_indexer: Indices of selected columns, shape (k_col,)
        """
        self.sketch_params = self.sketch_params or {'subsample': self.subsample, 'sketch_outputs': self.sketch_outputs}
        return self.sketch(tensor, **self.sketch_params)

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
                    break
                except Exception:
                    pass

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
            $$w_{i,j} = -sigmoid((g_{i,j} - \\mu_{g_{i,j}}) \\cdot (h_{i,j} - \\mu_{h_{i,j}}))$$
        
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
