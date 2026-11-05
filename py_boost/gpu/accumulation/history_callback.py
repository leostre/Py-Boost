import logging
from collections import deque

import cupy as cp
from py_boost.gpu.accumulation.sketches.sketch_methods import SUPPORTED_SKETCH_METHODS
from py_boost.multioutput.history_state import (
    AdaptiveBudgetConfig,
    AdaptiveBudgetState,
    RobustSchedulerConfig,
    RobustSchedulerState,
    SchedulerDecision,
    advance_adaptive_budget,
    advance_robust_scheduler,
    decide_fixed_warmup,
    decide_stable_regime,
    relative_volatility,
    resolve_budget_levels,
    update_scale_history,
)
from py_boost.multioutput.output_subspace import (
    diversity_aware_topk,
    spectral_output_policy,
)
from py_boost.multioutput.sketch_decision import SketchDecision, apply_output_indices
from py_boost.multioutput.sketching import GradSketch
from py_boost.multioutput.split_diagnostics import summarize_split_risks
from py_boost.multioutput.budget_policy import gradient_energy_budget


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
                scheduler_mode: 'current', 'never', 'fixed_warmup',
                    'robust_window', 'shadow_risk', or 'adaptive_budget'
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
        self.configured_sketch_outputs = int(self.sketch_outputs)
        self.tau = float(kwargs.get('tau', 1e-3))
        self.sketch_params = kwargs.get('sketch_params')

        self.smoothing_alpha = kwargs.get('smoothing_alpha', 0.1 ** (1 / 9))
        self.stabilization_threshold = float(kwargs.get('stabilization_threshold', 0.2))
        self.destabilization_threshold = float(
            kwargs.get('destabilization_threshold', self.stabilization_threshold)
        )
        self.min_warmup_rounds = int(kwargs.get('min_warmup_rounds', 2))
        scheduler_aliases = {
            'gated': 'current',
            'after_warmup': 'fixed_warmup',
        }
        requested_scheduler = str(kwargs.get('scheduler_mode', 'current'))
        self.scheduler_mode = scheduler_aliases.get(requested_scheduler, requested_scheduler)
        supported_schedulers = {
            'current', 'never', 'fixed_warmup', 'robust_window', 'shadow_risk',
            'adaptive_budget',
        }
        if self.scheduler_mode not in supported_schedulers:
            raise ValueError(
                f'Unknown scheduler_mode: {requested_scheduler}. '
                f'Available modes: {", ".join(sorted(supported_schedulers))}'
            )

        self.scheduler_window = int(kwargs.get('scheduler_window', 8))
        self.scheduler_min_observations = int(
            kwargs.get('scheduler_min_observations', min(4, self.scheduler_window))
        )
        self.scheduler_patience_on = int(kwargs.get('scheduler_patience_on', 2))
        self.scheduler_patience_off = int(kwargs.get('scheduler_patience_off', 1))
        self.scheduler_robust_k = float(kwargs.get('scheduler_robust_k', 1.0))
        self.scheduler_cooldown_rounds = int(kwargs.get('scheduler_cooldown_rounds', 2))
        self.split_diagnostics_enabled = bool(
            kwargs.get('split_diagnostics', False)
            or self.scheduler_mode == 'adaptive_budget'
        )
        self.split_diagnostics_top_k = int(kwargs.get('split_diagnostics_top_k', 5))
        self.split_output_attribution = bool(
            kwargs.get('split_output_attribution', False)
        )
        self.split_refinement_shadow = bool(
            kwargs.get('split_refinement_shadow', False)
        )
        self.split_refinement_apply = bool(
            kwargs.get('split_refinement_apply', False)
        )
        self.split_refinement_shortlists = {
            int(budget): int(size)
            for budget, size in kwargs.get('split_refinement_shortlists', {}).items()
        }
        self.split_diagnostics_control_interval = int(
            kwargs.get('split_diagnostics_control_interval', 5)
        )
        self.shadow_seed = int(kwargs.get('shadow_seed', kwargs.get('seed', 42)))
        self.record_selection_indices = bool(kwargs.get('record_selection_indices', False))
        self.adaptive_budget_levels = tuple(
            int(value)
            for value in kwargs.get('adaptive_budgets', (25, 50, 75, 100))
        )
        self.adaptive_safe_threshold = float(
            kwargs.get('adaptive_safe_threshold', 0.10)
        )
        self.adaptive_unsafe_threshold = float(
            kwargs.get('adaptive_unsafe_threshold', 0.15)
        )
        self.adaptive_patience_down = int(kwargs.get('adaptive_patience_down', 2))
        self.adaptive_min_dwell_rounds = int(
            kwargs.get('adaptive_min_dwell_rounds', 5)
        )
        self.adaptive_cooldown_after_increase_rounds = int(
            kwargs.get('adaptive_cooldown_after_increase_rounds', 10)
        )

        if not 0 <= self.smoothing_alpha < 1:
            raise ValueError('smoothing_alpha must satisfy 0 <= alpha < 1')
        if self.tau < 0:
            raise ValueError('tau must be non-negative')
        if self.destabilization_threshold < self.stabilization_threshold:
            raise ValueError('destabilization_threshold must be at least stabilization_threshold')
        if self.split_diagnostics_top_k < 0:
            raise ValueError('split_diagnostics_top_k must be non-negative')
        if self.split_diagnostics_control_interval < 1:
            raise ValueError('split_diagnostics_control_interval must be positive')

        self._robust_config = RobustSchedulerConfig(
            enable_threshold=self.stabilization_threshold,
            disable_threshold=self.destabilization_threshold,
            min_warmup_rounds=self.min_warmup_rounds,
            window_size=self.scheduler_window,
            min_observations=self.scheduler_min_observations,
            patience_on=self.scheduler_patience_on,
            patience_off=self.scheduler_patience_off,
            robust_k=self.scheduler_robust_k,
            cooldown_rounds=self.scheduler_cooldown_rounds,
        )

        self.eps = kwargs.get('eps', 1e-6)
        self.weight_transform = kwargs.get('weight_transform', 'sigmoidal')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_approximation = False
        self._stable_regime = False
        self._curr_iteration = 0
        self.decisions_applied = 0
        self.last_decision_shape = None
        self.last_unique_decision_shape = None
        self.scheduler_trace = []
        self.decision_trace = []
        self.shadow_decision_trace = []
        self.split_diagnostics_trace = []
        self.split_refinement_trace = []
        self.control_timings = {}
        self._robust_state = RobustSchedulerState()
        self._adaptive_config = None
        self._adaptive_state = None
        self._adaptive_output_budget = None
        self._adaptive_probe_budget = None

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
        self.scheduler_trace = []
        self.decision_trace = []
        self.shadow_decision_trace = []
        self.split_diagnostics_trace = []
        self.split_refinement_trace = []
        self.control_timings = {}
        self._robust_state = RobustSchedulerState()
        self._adaptive_config = None
        self._adaptive_state = None
        self._adaptive_output_budget = None
        self._adaptive_probe_budget = None
        self.sketch_outputs = self.configured_sketch_outputs
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

            scheduler_decision = self._scheduler(n_outputs=int(grad.shape[1]))
            self._stable_regime = scheduler_decision.stable_regime
            self.use_approximation = self._stable_regime
            self.scheduler_trace.append(
                {
                    'iteration': self._curr_iteration,
                    'volatility': self.current_volatility,
                    'scheduler_signal': scheduler_decision.volatility,
                    'signal_name': (
                        'split_regret_weighted_p95'
                        if self.scheduler_mode in {'shadow_risk', 'adaptive_budget'}
                        else 'gradient_norm_volatility'
                    ),
                    'stable_regime': scheduler_decision.stable_regime,
                    'score': scheduler_decision.score,
                    'observations': scheduler_decision.observations,
                    'reason': scheduler_decision.reason,
                    'scheduler_mode': self.scheduler_mode,
                    'output_budget': self._adaptive_output_budget,
                    'probe_output_budget': self._adaptive_probe_budget,
                    'budget_is_full': (
                        self._adaptive_output_budget == int(grad.shape[1])
                        if self._adaptive_output_budget is not None
                        else None
                    ),
                    'budget_transitioned': getattr(
                        scheduler_decision, 'budget_transitioned', False
                    ),
                    'budget_dwell_remaining': getattr(
                        scheduler_decision, 'budget_dwell_remaining', 0
                    ),
                    'budget_cooldown_remaining': getattr(
                        scheduler_decision, 'budget_cooldown_remaining', 0
                    ),
                }
            )
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

    def _scheduler(self, n_outputs=None):
        """Check if grad norm changes stabilized enough to sketch

        Uses ||s_t - s_(t-1)||_1 / (||h_g^(t-1)||_1 + eps).

        Returns:
            SchedulerDecision for the current iteration
        """
        if self.scheduler_mode == 'never':
            return SchedulerDecision(
                volatility=self.current_volatility,
                stable_regime=False,
                score=self.current_volatility,
                observations=int(self.current_volatility is not None),
                reason='disabled',
            )
        if self.scheduler_mode == 'fixed_warmup':
            return decide_fixed_warmup(
                iteration=self._curr_iteration,
                full_rounds=self.min_warmup_rounds,
            )
        if self.scheduler_mode == 'adaptive_budget':
            if n_outputs is None:
                raise ValueError('adaptive budget scheduling requires n_outputs')
            if self._adaptive_config is None:
                initial_budget = min(self.configured_sketch_outputs, n_outputs)
                levels = resolve_budget_levels(
                    (*self.adaptive_budget_levels, initial_budget),
                    n_outputs,
                )
                self._adaptive_config = AdaptiveBudgetConfig(
                    budgets=levels,
                    safe_threshold=self.adaptive_safe_threshold,
                    unsafe_threshold=self.adaptive_unsafe_threshold,
                    min_warmup_rounds=self.min_warmup_rounds,
                    patience_down=self.adaptive_patience_down,
                    min_dwell_rounds=self.adaptive_min_dwell_rounds,
                    cooldown_after_increase_rounds=(
                        self.adaptive_cooldown_after_increase_rounds
                    ),
                )
                self._adaptive_state = AdaptiveBudgetState(n_outputs)
            observation = self._split_risk_for_iteration(
                self._curr_iteration - 1, self._adaptive_state.current_budget)
            probe_observation = self._split_risk_for_iteration(
                self._curr_iteration - 1, self._adaptive_probe_budget)
            self._adaptive_state, budget_decision = advance_adaptive_budget(
                iteration=self._curr_iteration,
                observation=observation,
                state=self._adaptive_state,
                config=self._adaptive_config,
                probe_observation=probe_observation,
            )
            self.sketch_outputs = budget_decision.output_budget
            self._adaptive_output_budget = budget_decision.output_budget
            self._adaptive_probe_budget = budget_decision.probe_budget
            return SchedulerDecision(
                volatility=observation,
                stable_regime=budget_decision.sketching_enabled,
                score=observation,
                observations=budget_decision.observations,
                reason=budget_decision.reason,
                budget_transitioned=budget_decision.transitioned,
                budget_dwell_remaining=budget_decision.dwell_remaining,
                budget_cooldown_remaining=budget_decision.cooldown_remaining,
            )
        if self.scheduler_mode in {'robust_window', 'shadow_risk'}:
            observation = self.current_volatility
            if self.scheduler_mode == 'shadow_risk':
                observation = self._split_risk_for_iteration(self._curr_iteration - 1)
            self._robust_state, decision = advance_robust_scheduler(
                iteration=self._curr_iteration,
                observation=observation,
                state=self._robust_state,
                config=self._robust_config,
            )
            return decision
        return decide_stable_regime(
            iteration=self._curr_iteration,
            volatility=self.current_volatility,
            was_stable=self._stable_regime,
            enable_threshold=self.stabilization_threshold,
            disable_threshold=self.destabilization_threshold,
            min_warmup_rounds=self.min_warmup_rounds,
        )

    def _split_risk_for_iteration(self, iteration: int, output_budget=None):
        records = [
            item for item in self.split_diagnostics_trace
            if item['iteration'] == iteration
            and (output_budget is None or item.get('diagnostic_output_budget') == output_budget)
        ]
        summary = summarize_split_risks(records)
        # A zero-gain spurious split must not disappear under gain-weighted p95.
        return 1.0 if summary.spurious_split_count else summary.weighted_quantile

    def needs_split_diagnostics(self) -> bool:
        """Return whether this tree needs full-versus-sketched split diagnostics."""

        if not self.split_diagnostics_enabled:
            return False
        if self.scheduler_mode == 'adaptive_budget':
            return (
                self._curr_iteration == self.min_warmup_rounds
                or (
                    self._curr_iteration > self.min_warmup_rounds
                    and (
                        self._curr_iteration - self.min_warmup_rounds
                    ) % self.split_diagnostics_control_interval == 0
                )
            )
        if self.scheduler_mode != 'shadow_risk' or not self._stable_regime:
            return True
        return self._curr_iteration % self.split_diagnostics_control_interval == 0

    def record_split_diagnostics(self, group: int, depth: int, records) -> None:
        """Store JSON-serializable node records emitted by the tree builder."""

        for record in records:
            self.split_diagnostics_trace.append(
                {
                    'iteration': self._curr_iteration,
                    'group': int(group),
                    'depth': int(depth),
                    'applied_to_structure': self._stable_regime,
                    'gradient_norm_volatility': self.current_volatility,
                    'output_budget': self._adaptive_output_budget,
                    'probe_output_budget': self._adaptive_probe_budget,
                    **record,
                }
            )

    def record_split_refinement(self, group: int, depth: int, entry) -> None:
        """Store one exact-refinement batch for an active depth."""

        self.split_refinement_trace.append({
            'iteration': self._curr_iteration,
            'group': int(group),
            'depth': int(depth),
            **entry,
        })

    def get_indexers(self, tensor: cp.ndarray, random_state=None, output_budget=None):
        """Apply configured sketch to select row and output indices

        Args:
            tensor: cp.ndarray, 2d matrix to sketch (grad or weights)

        Returns:
            cp.ndarray, row indices
            cp.ndarray, output indices
        """
        sketch_params = dict(self.sketch_params or {})
        sketch_params.setdefault('subsample', self.subsample)
        sketch_params['sketch_outputs'] = (
            self.sketch_outputs if output_budget is None else output_budget
        )
        sketch_params.setdefault('tau', self.tau)
        if random_state is not None:
            sketch_params['random_state'] = random_state
        return self.sketch(tensor, **sketch_params)

    @staticmethod
    def _as_decision(selection) -> SketchDecision:
        if isinstance(selection, SketchDecision):
            return selection
        row_indices, output_indices = selection
        return SketchDecision(row_indices=row_indices, output_indices=output_indices)

    def _decision_tensor(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """Return the tensor from which row and output indices are selected."""

        return grad

    def make_decision(self, grad: cp.ndarray, hess: cp.ndarray):
        """Return an explicit structure-search decision when sketching is active."""

        if not self.use_approximation:
            return None

        tensor = self._decision_tensor(grad, hess)
        decision = self._as_decision(self.get_indexers(tensor))
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
        self.decision_trace.append(
            {
                'iteration': self._curr_iteration,
                'rows': self.last_decision_shape[0],
                'outputs': self.last_decision_shape[1],
                'unique_rows': self.last_unique_decision_shape[0],
                'unique_outputs': self.last_unique_decision_shape[1],
            }
        )
        if self.record_selection_indices:
            self.decision_trace[-1].update({
                'output_indices': [int(index) for index in decision.output_indices.get()],
                'all_rows': int(decision.row_indices.size) == int(grad.shape[0]),
                'row_indices': (None if int(decision.row_indices.size) == int(grad.shape[0])
                                else [int(index) for index in decision.row_indices.get()]),
            })
        return decision

    def make_shadow_decision(self, grad: cp.ndarray, hess: cp.ndarray, output_budget=None):
        """Select diagnostic indices without changing the model's random stream."""

        tensor = self._decision_tensor(grad, hess)
        random_state = cp.random.RandomState(self.shadow_seed + self._curr_iteration)
        output_budget = output_budget if output_budget is not None else (
            self._adaptive_probe_budget
            if self.scheduler_mode == 'adaptive_budget'
            else None
        )
        decision = self._as_decision(
            self.get_indexers(
                tensor,
                random_state=random_state,
                output_budget=output_budget,
            )
        )
        self.shadow_decision_trace.append(
            {
                'iteration': self._curr_iteration,
                'rows': int(decision.row_indices.size),
                'outputs': int(decision.output_indices.size),
            }
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


class CurrentGradientSampling(GradHessHistory):
    """Select outputs by the current squared-gradient energy."""

    def __init__(self, **kwargs):
        current_kwargs = dict(kwargs)
        current_kwargs.setdefault('sketch_method', 'topk')
        super().__init__(**current_kwargs)

    def _decision_tensor(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
        """Match SketchBoost TopOutputsSketch ranking with non-negative weights."""

        return grad ** 2


class CurrentGradientEnergySampling(CurrentGradientSampling):
    """Use the same current-energy ranking, varying only its output budget."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_energy_fraction = float(kwargs.get('output_energy_fraction', 0.90))
        self.output_min_fraction = float(kwargs.get('output_min_fraction', 0.50))
        if not 0 < self.output_energy_fraction <= 1 or not 0 < self.output_min_fraction <= 1:
            raise ValueError('output fractions must be within (0, 1]')

    def get_indexers(self, tensor, random_state=None, output_budget=None):
        if output_budget is None:
            output_budget = gradient_energy_budget(
                tensor.sum(axis=0), self.output_energy_fraction,
                max(1, int(cp.ceil(self.output_min_fraction * tensor.shape[1]))), xp=cp)
        return super().get_indexers(tensor, random_state=random_state, output_budget=output_budget)

    def make_decision(self, grad, hess):
        decision = super().make_decision(grad, hess)
        budget = grad.shape[1] if decision is None else decision.output_indices.size
        self.scheduler_trace[-1].update(output_budget=int(budget),
                                        budget_is_full=int(budget) == int(grad.shape[1]))
        return decision


class EmaGradientEnergySampling(CurrentGradientSampling):
    """Isolate memory: same squared-gradient energy, same deterministic selector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_energy_history = None

    def before_train(self, build_info):
        super().before_train(build_info)
        self.output_energy_history = None

    def before_iteration(self, build_info):
        super().before_iteration(build_info)
        gradient = build_info['data']['train'].get('grad')
        if gradient is not None:
            self.output_energy_history = update_scale_history(
                self.output_energy_history, cp.sum(gradient ** 2, axis=0), self.smoothing_alpha)

    def _decision_tensor(self, grad, hess):
        return cp.broadcast_to(self.output_energy_history / grad.shape[0], grad.shape)


class OutputSubspaceTracker:
    """Track a rolling output covariance and its empirical-null spectrum."""

    def __init__(
        self,
        window: int = 16,
        null_permutations: int = 5,
        null_rows: int = 2048,
        null_interval: int = 5,
        beta: float = 0.9,
        tau: float = 1e-6,
        seed: int = 42,
    ):
        if window < 2:
            raise ValueError('window must be at least two')
        if null_permutations < 1:
            raise ValueError('null_permutations must be positive')
        if null_rows < 2:
            raise ValueError('null_rows must be at least two')
        if null_interval < 1:
            raise ValueError('null_interval must be positive')
        if not 0 <= beta <= 1:
            raise ValueError('beta must satisfy 0 <= beta <= 1')
        if tau < 0:
            raise ValueError('tau must be non-negative')

        self.window = int(window)
        self.null_permutations = int(null_permutations)
        self.null_rows = int(null_rows)
        self.null_interval = int(null_interval)
        self.beta = float(beta)
        self.tau = float(tau)
        self.seed = int(seed)
        self._covariances = deque(maxlen=self.window)
        self._null_covariances_history = deque(maxlen=self.window)
        self._latest_null_covariances = None
        self._previous_spike_basis = None

    def reset(self) -> None:
        """Discard all observations from the previous training run."""

        self._covariances.clear()
        self._null_covariances_history.clear()
        self._latest_null_covariances = None
        self._previous_spike_basis = None

    @staticmethod
    def _normalized_gh(gradient, hessian):
        hessian = cp.broadcast_to(hessian, gradient.shape)
        eps = cp.finfo(gradient.dtype).eps
        gradient = gradient / (cp.sqrt(cp.mean(gradient ** 2)) + eps)
        hessian = hessian / (cp.sqrt(cp.mean(hessian ** 2)) + eps)
        return gradient, hessian

    def _null_covariances(self, iteration, gradient, hessian):
        rows = gradient.shape[0]
        selected_rows = min(rows, self.null_rows)
        rng = cp.random.RandomState(self.seed + 10_000 * iteration)
        row_index = rng.permutation(rows)[:selected_rows]
        gradient = gradient[row_index]
        hessian = hessian[row_index]
        outputs = gradient.shape[1]
        nulls = cp.empty(
            (self.null_permutations, outputs, outputs),
            dtype=gradient.dtype,
        )
        permuted_gradient = cp.empty_like(gradient)
        permuted_hessian = cp.empty_like(hessian)
        for permutation in range(self.null_permutations):
            for output in range(outputs):
                permuted_gradient[:, output] = gradient[
                    rng.permutation(selected_rows), output
                ]
                permuted_hessian[:, output] = hessian[
                    rng.permutation(selected_rows), output
                ]
            nulls[permutation] = (
                permuted_gradient.T @ permuted_gradient
                + permuted_hessian.T @ permuted_hessian
            ) / (2 * selected_rows)
        return nulls

    def observe(self, iteration, gradient, hessian):
        """Update state and return protected probabilities once the window is full."""

        gradient, hessian = self._normalized_gh(gradient, hessian)
        rows = gradient.shape[0]
        covariance = (
            gradient.T @ gradient + hessian.T @ hessian
        ) / (2 * rows)
        self._covariances.append(covariance)
        if (
            self._latest_null_covariances is None
            or (iteration - 1) % self.null_interval == 0
        ):
            self._latest_null_covariances = self._null_covariances(
                iteration,
                gradient,
                hessian,
            )
        self._null_covariances_history.append(self._latest_null_covariances)
        if len(self._covariances) < self.window:
            return None

        rolling_covariance = cp.mean(cp.stack(tuple(self._covariances)), axis=0)
        eigenvalues, eigenvectors = cp.linalg.eigh(rolling_covariance)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        rolling_null = cp.mean(
            cp.stack(tuple(self._null_covariances_history)),
            axis=0,
        )
        null_eigenvalues = cp.linalg.eigvalsh(rolling_null)[:, ::-1]
        null_p95 = cp.quantile(null_eigenvalues, 0.95, axis=0)
        summary = spectral_output_policy(
            eigenvalues.get(),
            eigenvectors.get(),
            null_p95.get(),
            beta=self.beta,
            tau=self.tau,
        )
        component_indices = cp.asarray(summary.significant_component_indices, dtype=cp.int64)
        current_basis = eigenvectors[:, component_indices]
        subspace_drift = None
        if (
            self._previous_spike_basis is not None
            and self._previous_spike_basis.shape[1] > 0
            and current_basis.shape[1] > 0
        ):
            singular_values = cp.linalg.svd(
                self._previous_spike_basis.T @ current_basis,
                compute_uv=False,
            )
            overlap = cp.sum(singular_values ** 2) / max(
                self._previous_spike_basis.shape[1],
                current_basis.shape[1],
            )
            subspace_drift = float(1 - cp.clip(overlap, 0, 1))
        self._previous_spike_basis = current_basis.copy()
        probabilities = cp.asarray(summary.probabilities, dtype=gradient.dtype)
        return probabilities, summary, subspace_drift


class SpectralHistorySampling(WeightedHistorySampling):
    """Output-only history sketch using a rolling empirical-null eigenspace."""

    def __init__(self, **kwargs):
        spectral_kwargs = dict(kwargs)
        spectral_kwargs.setdefault('sketch_method', 'topk')
        super().__init__(**spectral_kwargs)
        self.spectral_window = int(kwargs.get('spectral_window', 16))
        self.spectral_null_permutations = int(
            kwargs.get('spectral_null_permutations', 5)
        )
        self.spectral_null_rows = int(kwargs.get('spectral_null_rows', 2048))
        self.spectral_null_interval = int(kwargs.get('spectral_null_interval', 5))
        self.spectral_beta = float(kwargs.get('spectral_beta', 0.9))
        self.spectral_tau = float(kwargs.get('spectral_tau', 1e-6))
        self.spectral_diversity_weight = float(
            kwargs.get('spectral_diversity_weight', 0.0)
        )
        if not 0 <= self.spectral_diversity_weight <= 1:
            raise ValueError('spectral_diversity_weight must satisfy 0 <= value <= 1')
        self.output_subspace_tracker = OutputSubspaceTracker(
            window=self.spectral_window,
            null_permutations=self.spectral_null_permutations,
            null_rows=self.spectral_null_rows,
            null_interval=self.spectral_null_interval,
            beta=self.spectral_beta,
            tau=self.spectral_tau,
            seed=self.shadow_seed,
        )
        self._spectral_probabilities = None
        self._spectral_loading_profiles = None
        self.spectral_trace = []

    @property
    def spectral_probabilities(self):
        """Return the current protected output distribution, if available."""

        return self._spectral_probabilities

    def before_train(self, build_info):
        super().before_train(build_info)
        self.output_subspace_tracker.reset()
        self._spectral_probabilities = None
        self._spectral_loading_profiles = None
        self.spectral_trace = []

    def _update_output_subspace(self, gradient, hessian):
        update = self.output_subspace_tracker.observe(
            self._curr_iteration,
            gradient,
            hessian,
        )
        if update is None:
            return
        self._spectral_probabilities, summary, subspace_drift = update
        self._spectral_loading_profiles = summary.loading_profiles
        budget = min(self.sketch_outputs, gradient.shape[1])
        top_outputs = self._select_spectral_outputs(budget)
        self.spectral_trace.append(
            {
                'iteration': self._curr_iteration,
                'spike_count': summary.spike_count,
                'significant_component_indices': list(summary.significant_component_indices),
                'effective_rank': summary.effective_rank,
                'spike_energy_fraction': summary.spike_energy_fraction,
                'relative_spike_margin': summary.relative_spike_margin,
                'subspace_drift': subspace_drift,
                'top_outputs': [int(value) for value in top_outputs.get()],
                'probabilities': list(summary.probabilities),
                'loading_profiles': [list(row) for row in summary.loading_profiles],
            }
        )

    def _select_spectral_outputs(self, budget):
        if self.spectral_diversity_weight <= 0:
            return cp.sort(cp.argsort(self._spectral_probabilities)[-budget:]).astype(
                cp.uint64
            )
        selected = diversity_aware_topk(
            self._spectral_probabilities.get(),
            self._spectral_loading_profiles,
            budget,
            diversity_weight=self.spectral_diversity_weight,
        )
        return cp.asarray(sorted(selected), dtype=cp.uint64)

    def before_iteration(self, build_info):
        super().before_iteration(build_info)
        train = build_info['data']['train']
        gradient = train.get('grad')
        hessian = train.get('hess')
        if gradient is not None and hessian is not None:
            self._update_output_subspace(gradient, hessian)

    def get_indexers(
        self,
        tensor: cp.ndarray,
        random_state=None,
        output_budget=None,
    ):
        decision = self._as_decision(
            super().get_indexers(
                tensor,
                random_state=random_state,
                output_budget=output_budget,
            )
        )
        if self._spectral_probabilities is None:
            output_indices = cp.arange(tensor.shape[1], dtype=cp.uint64)
            return SketchDecision(
                decision.row_indices,
                output_indices,
                decision.row_probabilities,
                None,
            )
        budget = min(
            self.sketch_outputs if output_budget is None else output_budget,
            tensor.shape[1],
        )
        output_indices = self._select_spectral_outputs(budget)
        return SketchDecision(
            decision.row_indices,
            output_indices,
            decision.row_probabilities,
            self._spectral_probabilities,
        )
