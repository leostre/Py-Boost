import ctypes
import inspect
import logging

import cupy as cp
from py_boost.gpu.accumulation.sketches.sketch_methods import SUPPORTED_SKETCH_METHODS
from py_boost.multioutput.sketching import GradSketch


class GradHessHistory(GradSketch):
    """Callback that accumulates grads/hess, schedules and applies Fedcore approximation."""

    def __init__(self, **kwargs):
        sketch_method = kwargs.get('sketch_method', 'svd')

        self.sketch = SUPPORTED_SKETCH_METHODS.get(sketch_method)
        if self.sketch is None:
            raise ValueError(
                f'Unknown sketching strategy: {sketch_method}. '
                f'Available methods: {", ".join(SUPPORTED_SKETCH_METHODS.keys())}'
            )

        self.subsample = kwargs.get('subsample', 0.5)
        self.sketch_outputs = kwargs.get('sketch_outputs', 1)
        self.sketch_params = kwargs.get('sketch_params', {'subsample': self.subsample,
                                                          'sketch_outputs': self.sketch_outputs})

        self.stabilization_window = int(kwargs.get('stabilization_window', 10))
        self.smoothing_alpha = kwargs.get('smoothing_alpha', 0.1 ** (1 / 9))
        self.stabilization_threshold = 0.1

        self.eps = kwargs.get('eps', 1e-6)
        self.weight_transform = kwargs.get('weight_transform', 'sigmoidal')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_approximation = False
        self._curr_iteration = 0

        self.prev_grad_norms = None  # v_g^(t-1) - previous gradient norms
        self.prev_hess_norms = None  # v_h^(t-1) - previous hessian norms
        self.curr_grad_norms = None  # v_g^(t) - current gradient norms
        self.curr_hess_norms = None  # v_h^(t) - current hessian norms
        self.grad_history = None  # h_g^(t) - gradient history EMA
        self.hess_history = None  # h_h^(t) - hessian history EMA

    def before_train(self, build_info):
        self.use_approximation = False
        self._curr_iteration = 0
        self.prev_grad_norms = None
        self.prev_hess_norms = None
        self.grad_history = None
        self.hess_history = None
        self.curr_grad_norms = None
        self.curr_hess_norms = None

    def before_iteration(self, build_info):
        self._curr_iteration = build_info['num_iter'] + 1
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')

        if grad is not None and hess is not None:
            self.curr_grad_norms = cp.linalg.norm(grad, axis=1)
            self.curr_hess_norms = cp.linalg.norm(hess, axis=1)

            # check if we should apply approximation based on EMA history
            self.use_approximation = self._scheduler()
            self._update_history()

    def _update_history(self):
        if self.prev_grad_norms is None or self.prev_hess_norms is None:
            if self.curr_grad_norms is not None and self.curr_hess_norms is not None:
                self.prev_grad_norms = self.curr_grad_norms.copy()
                self.prev_hess_norms = self.curr_hess_norms.copy()
                self.grad_history = cp.zeros_like(self.curr_grad_norms)
                self.hess_history = cp.zeros_like(self.curr_hess_norms)
            return

        if self.curr_grad_norms is not None and self.prev_grad_norms is not None:
            grad_delta = self.curr_grad_norms - self.prev_grad_norms

            if self.grad_history is None:
                self.grad_history = grad_delta
            else:
                self.grad_history = (
                    self.smoothing_alpha * self.grad_history + (1 - self.smoothing_alpha) * grad_delta
                )

            self.prev_grad_norms = self.curr_grad_norms.copy()

        if self.curr_hess_norms is not None and self.prev_hess_norms is not None:
            hess_delta = self.curr_hess_norms - self.prev_hess_norms

            if self.hess_history is None:
                self.hess_history = hess_delta
            else:
                self.hess_history = (
                    self.smoothing_alpha * self.hess_history + (1 - self.smoothing_alpha) * hess_delta
                )

            self.prev_hess_norms = self.curr_hess_norms.copy()

    def _scheduler(self) -> bool:
        """
        Determine if gradient stabilization has occurred to enable approximation.
        Condition: ||∇v_g^(t)||1 / (||h_g^(t-1)||1 + eps) < stabilization_threshold
        """
        if (self.prev_grad_norms is None or self.curr_grad_norms is None or 
            self.grad_history is None or self._curr_iteration < 2):
            return False

        grad_delta = self.curr_grad_norms - self.prev_grad_norms

        delta_norm = cp.linalg.norm(grad_delta, ord=1)
        history_norm = cp.linalg.norm(self.grad_history, ord=1)
        stabilization_ratio = delta_norm / (history_norm + self.eps)

        use_approximation = stabilization_ratio < self.stabilization_threshold
        return use_approximation

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
        Compute adaptive weights based on deviations from historical gradient and hessian norms.

        For each element (i,j), the weight is computed as:
            1. Calculate deviation metrics:
                g_dev = (grad[i,j] / (|grad_history[i]| + eps)) - 1
                h_dev = (hess[i,j] / (|hess_history[i]| + eps)) - 1

            2. Compute the product deviation: deviation = g_dev * h_dev

            3. Transform via selected method:
                - 'sigmoidal': weight = 1 - sigmoid(deviation)
                - 'hyperbolic': weight = 1 - normalized(deviation)

        The weights are clipped to [0, 1], where lower values indicate larger deviations
        from historical behavior, suggesting higher uncertainty or novelty.

        Args:
            grad: Current gradient tensor of shape (n_samples, n_outputs).
            hess: Current hessian tensor of shape (n_samples, n_outputs).

        Returns:
            weights: Weight tensor of shape (n_samples, n_outputs) with values in [0, 1].
        """
        n_samples, n_outputs = grad.shape

        if self.grad_history is None or self.hess_history is None:
            return cp.ones((n_samples, n_outputs), dtype=cp.float32)

        # expand history to [n_samples, 1], rely on broadcasting to match [n_samples, n_outputs]
        grad_history_expanded = cp.abs(self.grad_history)[:, cp.newaxis]
        hess_history_expanded = cp.abs(self.hess_history)[:, cp.newaxis]

        grad_dev = grad / (grad_history_expanded + self.eps) - 1
        hess_dev = hess / (hess_history_expanded + self.eps) - 1
        deviation = grad_dev * hess_dev

        if self.weight_transform == 'sigmoidal':
            weights = 1 - 1 / (1 + cp.exp(-deviation))
        elif self.weight_transform == 'hyperbolic':
            min_val = cp.min(deviation)
            max_val = cp.max(deviation)
            if max_val > min_val:
                normalized = (deviation - min_val) / (max_val - min_val)
            else:
                normalized = cp.zeros_like(deviation)
            weights = 1 - normalized
        else:
            raise ValueError(
                f"Unknown weight_transform: {self.weight_transform}. "
                "Must be 'sigmoidal' or 'hyperbolic'."
            )

        return weights
