import logging

import cupy as cp
from py_boost.gpu.accumulation.sketches.sketch_methods import SUPPORTED_SKETCH_METHODS
from py_boost.multioutput.history_state import (
    decide_stable_regime,
    relative_volatility,
    update_scale_history,
)
from py_boost.multioutput.sketch_decision import SketchDecision, apply_output_indices
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
                smoothing_alpha: float, EMA coefficient for scale history
                stabilization_threshold: float, ratio threshold to enable sketching, default 0.2
                destabilization_threshold: float, threshold to disable sketching, defaults to enable threshold
                min_warmup_rounds: int, first round in which sketching may be enabled, default 2
                eps: float, numerical stabilizer, default 1e-6
                tau: float, smoothing constant for row/output marginal probabilities, default 1e-3
                weight_transform: str, 'sigmoidal' or 'hyperbolic' (subclasses), default 'sigmoidal'
        """
        sketch_method = kwargs.get('sketch_method', 'svd')

        self.sketch = SUPPORTED_SKETCH_METHODS.get(sketch_method)
        if self.sketch is None:
            raise ValueError(
                f'Unknown sketching strategy: {sketch_method}. '
                f'Available methods: {", ".join(SUPPORTED_SKETCH_METHODS.keys())}'
            )
        self.sketch_method = sketch_method

        self.subsample = kwargs.get('subsample', 0.5)
        self.sketch_outputs = kwargs.get('sketch_outputs', 1)
        self.tau = float(kwargs.get('tau', 1e-3))
        self.sketch_params = kwargs.get('sketch_params')

        self.smoothing_alpha = kwargs.get('smoothing_alpha', 0.1 ** (1 / 9))
        self.stabilization_threshold = float(kwargs.get('stabilization_threshold', 0.2))
        self.destabilization_threshold = float(
            kwargs.get('destabilization_threshold', self.stabilization_threshold)
        )
        self.min_warmup_rounds = int(kwargs.get('min_warmup_rounds', 2))

        if not 0 <= self.smoothing_alpha < 1:
            raise ValueError('smoothing_alpha must satisfy 0 <= alpha < 1')
        if self.tau < 0:
            raise ValueError('tau must be non-negative')
        if self.destabilization_threshold < self.stabilization_threshold:
            raise ValueError('destabilization_threshold must be at least stabilization_threshold')

        self.eps = kwargs.get('eps', 1e-6)
        self.weight_transform = kwargs.get('weight_transform', 'sigmoidal')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_approximation = False
        self._stable_regime = False
        self._curr_iteration = 0
        self.decisions_applied = 0
        self.last_decision_shape = None
        self.last_unique_decision_shape = None

        self.prev_grad_norms = None  # v_g^(t-1) - previous gradient norms
        self.prev_hess_norms = None  # v_h^(t-1) - previous hessian norms
        self.curr_grad_norms = None  # v_g^(t) - current gradient norms
        self.curr_hess_norms = None  # v_h^(t) - current hessian norms
        self.grad_history = None  # h_g^(t) - gradient history EMA
        self.hess_history = None  # h_h^(t) - hessian history EMA
        self.reference_grad_history = None  # h_g^(t-1), used for current weights
        self.reference_hess_history = None  # h_h^(t-1), used for current weights
        self.current_volatility = None

    def before_train(self, build_info):
        """Reset iteration counter and all history buffers at train start.

        Args:
            build_info: Boosting build context passed by the trainer (unused).
        """
        self.use_approximation = False
        self._stable_regime = False
        self._curr_iteration = 0
        self.decisions_applied = 0
        self.last_decision_shape = None
        self.last_unique_decision_shape = None
        self.prev_grad_norms = None
        self.prev_hess_norms = None
        self.grad_history = None
        self.hess_history = None
        self.curr_grad_norms = None
        self.curr_hess_norms = None
        self.reference_grad_history = None
        self.reference_hess_history = None
        self.current_volatility = None

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

            self.reference_grad_history = self.grad_history
            self.reference_hess_history = self.hess_history

            if self.prev_grad_norms is not None and self.reference_grad_history is not None:
                self.current_volatility = relative_volatility(
                    self.curr_grad_norms,
                    self.prev_grad_norms,
                    self.reference_grad_history,
                    self.eps,
                )
            else:
                self.current_volatility = None

            scheduler_decision = self._scheduler()
            self._stable_regime = scheduler_decision.stable_regime
            self.use_approximation = self._stable_regime
            self._update_history()

    def _update_history(self):
        """Update EMA histories of gradient/Hessian magnitudes after scheduling."""
        if self.curr_grad_norms is None or self.curr_hess_norms is None:
            return

        self.grad_history = update_scale_history(
            self.grad_history,
            self.curr_grad_norms,
            self.smoothing_alpha,
        )
        self.hess_history = update_scale_history(
            self.hess_history,
            self.curr_hess_norms,
            self.smoothing_alpha,
        )
        self.prev_grad_norms = self.curr_grad_norms
        self.prev_hess_norms = self.curr_hess_norms

    def _scheduler(self):
        """Check if grad norm changes stabilized enough to sketch

        Uses ||s_t - s_(t-1)||_1 / (||h_g^(t-1)||_1 + eps).

        Returns:
            SchedulerDecision for the current iteration
        """
        return decide_stable_regime(
            iteration=self._curr_iteration,
            volatility=self.current_volatility,
            was_stable=self._stable_regime,
            enable_threshold=self.stabilization_threshold,
            disable_threshold=self.destabilization_threshold,
            min_warmup_rounds=self.min_warmup_rounds,
        )

    def get_indexers(self, tensor: cp.ndarray):
        """Apply configured sketch to select row and output indices

        Args:
            tensor: cp.ndarray, 2d matrix to sketch (grad or weights)

        Returns:
            cp.ndarray, row indices
            cp.ndarray, output indices
        """
        sketch_params = dict(self.sketch_params or {})
        sketch_params.setdefault('subsample', self.subsample)
        sketch_params.setdefault('sketch_outputs', self.sketch_outputs)
        sketch_params.setdefault('tau', self.tau)
        return self.sketch(tensor, **sketch_params)

    def _decision_tensor(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """Return the tensor from which row and output indices are selected."""

        return grad

    def make_decision(self, grad: cp.ndarray, hess: cp.ndarray):
        """Return an explicit structure-search decision when sketching is active."""

        if not self.use_approximation:
            return None

        tensor = self._decision_tensor(grad, hess)
        selection = self.get_indexers(tensor)
        if isinstance(selection, SketchDecision):
            decision = selection
        else:
            row_indices, output_indices = selection
            decision = SketchDecision(row_indices=row_indices, output_indices=output_indices)
        self.use_approximation = False
        self.decisions_applied += 1
        self.last_decision_shape = (
            int(decision.row_indices.size),
            int(decision.output_indices.size),
        )
        self.last_unique_decision_shape = (
            int(cp.unique(decision.row_indices).size),
            int(cp.unique(decision.output_indices).size),
        )
        return decision

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        """Apply selected output columns for compatibility with the sketch protocol.

        Tree builders should prefer :meth:`make_decision`, because a pair of
        gradient/Hessian matrices cannot also carry the selected row indices.

        Args:
            grad: cp.ndarray, gradients
            hess: cp.ndarray, hessians

        Returns:
            cp.ndarray, grad
            cp.ndarray, hess
        """
        decision = self.make_decision(grad, hess)
        if decision is None:
            return grad, hess
        return apply_output_indices(grad, hess, decision.output_indices)


class WeightedHistorySampling(GradHessHistory):
    """History-based sketching driven by per-element deviation weights.

    Like :class:`GradHessHistory`, but passes a weight tensor derived from
    grad/hess deviation from EMA norm history into the sketch method instead of
    raw gradients.
    """

    def __init__(self, **kwargs):
        weighted_kwargs = dict(kwargs)
        weighted_kwargs.setdefault('sketch_method', 'rand')
        super().__init__(**weighted_kwargs)

    def _decision_tensor(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """Select rows and outputs from adaptive history weights."""

        return self.get_weights(grad, hess)

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

        grad_history = self.reference_grad_history
        hess_history = self.reference_hess_history
        if grad_history is None or hess_history is None:
            return cp.ones((n_samples, n_outputs), dtype=cp.float32)

        # expand history to [n_samples, 1], rely on broadcasting to match [n_samples, n_outputs]
        grad_history_expanded = cp.abs(grad_history)[:, cp.newaxis]
        hess_history_expanded = cp.abs(hess_history)[:, cp.newaxis]

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
