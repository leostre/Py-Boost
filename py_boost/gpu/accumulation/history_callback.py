import ctypes
import inspect
import logging

import cupy as cp
from py_boost.gpu.accumulation.sketches.sketch_methods import SUPPORTED_SKETCH_METHODS
from py_boost.multioutput.sketching import GradSketch


class GradHessHistory(GradSketch):
    """Grad/hess history sketching. Schedules row/col subsampling when norm deltas stabilize."""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: callback params. Supported keys:
                sketch_method: str, key in SUPPORTED_SKETCH_METHODS, default 'svd'
                subsample: float, row fraction for sketching, default 0.5
                sketch_outputs: int, number of output columns to keep, default 1
                sketch_params: dict, extra args for sketch callable
                stabilization_window: int, stored window size, default 10
                smoothing_alpha: float, EMA coefficient for norm history
                stabilization_threshold: float, ratio threshold to enable sketching, default 1.0
                eps: float, numerical stabilizer, default 1e-6
                weight_transform: str, 'sigmoidal' or 'hyperbolic' (subclasses), default 'sigmoidal'
        """
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
        self.stabilization_threshold = float(kwargs.get('stabilization_threshold', 1.0))

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
        self.curr_aggregated_norm = None
        self.aggregated_norm_history = None

    def before_train(self, build_info):
        """Reset iteration counter and all history buffers at train start.

        Args:
            build_info: Boosting build context passed by the trainer (unused).
        """
        self.use_approximation = False
        self._curr_iteration = 0
        self.prev_grad_norms = None
        self.prev_hess_norms = None
        self.grad_history = None
        self.hess_history = None
        self.curr_grad_norms = None
        self.curr_hess_norms = None

    def before_iteration(self, build_info):
        """Update norm statistics and decide whether to sketch on the next step.

        Reads ``grad`` and ``hess`` from ``build_info['data']['train']``, computes
        per-sample L2 norms, updates EMA history, and sets ``use_approximation`` via
        :meth:`_scheduler`.

        Args:
            build_info: Per-iteration context with ``num_iter`` and training tensors.
        """
        self._curr_iteration = build_info['num_iter'] + 1
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')

        if grad is not None and hess is not None:
            self.curr_grad_norms = cp.linalg.norm(grad, axis=1)
            self.curr_hess_norms = cp.linalg.norm(hess, axis=1)

            if self.prev_grad_norms is not None:
                grad_delta = self.curr_grad_norms - self.prev_grad_norms
                self.curr_aggregated_norm = cp.linalg.norm(grad_delta, ord=1)
                if self.aggregated_norm_history is None:
                    self.aggregated_norm_history = 0.0

            # check if we should apply approximation based on EMA history
            self.use_approximation = self._scheduler()
            self._update_history()

    def _update_history(self):
        """Apply EMA updates to per-sample norm deltas and aggregated L1 change."""
        if self.prev_grad_norms is None or self.prev_hess_norms is None:
            if self.curr_grad_norms is not None and self.curr_hess_norms is not None:
                self.prev_grad_norms = self.curr_grad_norms
                self.prev_hess_norms = self.curr_hess_norms
                self.grad_history = cp.zeros_like(self.curr_grad_norms)
                self.hess_history = cp.zeros_like(self.curr_hess_norms)
            return

        if self.curr_grad_norms is not None and self.prev_grad_norms is not None:
            grad_delta = self.curr_grad_norms - self.prev_grad_norms

            self.aggregated_norm_history =  (
                self.smoothing_alpha * self.aggregated_norm_history + (1 - self.smoothing_alpha) * self.curr_aggregated_norm
            )

            if self.grad_history is None:
                self.grad_history = grad_delta
            else:
                self.grad_history = (
                    self.smoothing_alpha * self.grad_history + (1 - self.smoothing_alpha) * grad_delta
                )

            self.prev_grad_norms = self.curr_grad_norms

        if self.curr_hess_norms is not None and self.prev_hess_norms is not None:
            hess_delta = self.curr_hess_norms - self.prev_hess_norms

            if self.hess_history is None:
                self.hess_history = hess_delta
            else:
                self.hess_history = (
                    self.smoothing_alpha * self.hess_history + (1 - self.smoothing_alpha) * hess_delta
                )

            self.prev_hess_norms = self.curr_hess_norms

    def _scheduler(self) -> bool:
        """Check if grad norm changes stabilized enough to sketch

        Uses curr_aggregated_norm / (aggregated_norm_history + eps) < stabilization_threshold

        Returns:
            bool, if approximation should run on next __call__
        """
        if (self.prev_grad_norms is None or self.curr_grad_norms is None or 
            self.grad_history is None or self._curr_iteration < 2):
            return False

        stabilization_ratio = self.curr_aggregated_norm / (self.aggregated_norm_history + self.eps)

        use_approximation = stabilization_ratio < self.stabilization_threshold
        return use_approximation

    def get_indexers(self, tensor: cp.ndarray):
        """Apply configured sketch to select row and column indices

        Args:
            tensor: cp.ndarray, 2d matrix to sketch (grad or weights)

        Returns:
            cp.ndarray, row indices
            cp.ndarray, column indices
        """
        self.sketch_params = self.sketch_params or {'subsample': self.subsample, 'sketch_outputs': self.sketch_outputs}
        return self.sketch(tensor, **self.sketch_params)

    def _set_indexers(self, row_indexer: cp.ndarray, col_indexer: cp.ndarray) -> None:
        """Pass indexers into build_tree frame locals"""
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
        """Inject row/col indexers when approximation is scheduled; return grad/hess unchanged

        Args:
            grad: cp.ndarray, gradients
            hess: cp.ndarray, hessians

        Returns:
            cp.ndarray, grad
            cp.ndarray, hess
        """
        if self.use_approximation:
            row_indexer, col_indexer = self.get_indexers(grad)
            self._set_indexers(row_indexer=row_indexer, col_indexer=col_indexer)
            self.use_approximation = False
        return grad, hess


class WeightedHistorySampling(GradHessHistory):
    """History-based sketching driven by per-element deviation weights.

    Like :class:`GradHessHistory`, but passes a weight tensor derived from
    grad/hess deviation from EMA norm history into the sketch method instead of
    raw gradients.
    """

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        """Sketch from adaptive weights when approximation is scheduled

        Args:
            grad: cp.ndarray, gradients
            hess: cp.ndarray, hessians

        Returns:
            cp.ndarray, grad
            cp.ndarray, hess
        """
        if self.use_approximation:
            weights = self.get_weights(grad, hess)
            row_indexer, col_indexer = self.get_indexers(weights)
            self._set_indexers(row_indexer=row_indexer, col_indexer=col_indexer)
            self.use_approximation = False
        return grad, hess

    def get_weights(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """Map grad/hess deviations from EMA history to sampling weights.

        Per element ``(i, j)``:

            g_dev = grad[i, j] / (|grad_history[i]| + eps) - 1
            h_dev = hess[i, j] / (|hess_history[i]| + eps) - 1
            deviation = g_dev * h_dev

        Then applies ``weight_transform``:

            * ``'sigmoidal'``: ``1 - sigmoid(deviation)``
            * ``'hyperbolic'``: ``1 - min-max(deviation)``

        Lower weights indicate larger deviation from historical norms.

        Args:
            grad: cp.ndarray, gradients
            hess: cp.ndarray, hessians

        Returns:
            cp.ndarray, weights of shape (n_samples, n_outputs)
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
