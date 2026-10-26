"""GPU shell around pure validation-quality budget planning."""
from dataclasses import asdict
import time
import numpy as np
from py_boost.callbacks.callback import Callback
from experiments.adaptive_core import affordable_depth
from experiments.risk_budget_core import Quality, WindowGain, deficit, plan_depth, Decision


class QualityEfficiencyController(Callback):
    def __init__(self, *, mode, reference, parameter_cap, seed, task, horizon, emit, stop_on_quality=True):
        if mode not in ('fixed', 'alternating', 'latest', 'history', 'deceleration') or horizon < 100:
            raise ValueError('unsupported controller configuration')
        self.mode, self.reference, self.cap = mode, Quality(**reference), int(parameter_cap)
        self.seed, self.task, self.horizon, self.emit = seed, task, horizon, emit
        if type(stop_on_quality) is not bool:
            raise ValueError('stop_on_quality must be a boolean')
        self.stop_on_quality = stop_on_quality
        self.depth, self.reason = 5, 'common_warmup'
        self.trace, self.checks, self.windows, self.frames, self.epochs = [], [], [], [], []
        self.overhead, self.confirmations = 0., 0
        self.block_start = None
        self.block_depths = []

    def before_train(self, info):
        import cupy as cp
        self.outputs = int(info['model'].base_score.size)
        self.parameters = self.outputs
        if affordable_depth(5, self.cap - self.parameters, self.outputs) < 5:
            raise ValueError('cap cannot admit the common first depth-5 tree')
        n = info['data']['train']['target'].shape[0]
        self.probe = np.random.default_rng(self.seed + 70000).choice(n, min(n, 512), replace=False)
        valid_rows = info['data']['valid']['target'][0].shape[0]
        # Deployment accumulates float32. Do not change the float64 training ensemble.
        self.validation_prediction = cp.empty((valid_rows, self.outputs), dtype=cp.float32)
        self.validation_prediction[:] = info['model'].base_score

    def before_iteration(self, info):
        started = time.perf_counter()
        step = info['num_iter']
        if step % 20 == 0:
            train = info['data']['train']
            g, h = train['grad'][self.probe].get(), train['hess'][self.probe].get()
            if h.shape[1] == 1:
                h = np.broadcast_to(h, g.shape).copy()
            self.frames.append((g, h))
            self.epochs.append(step)
        self.applied_depth = affordable_depth(self.depth, self.cap - self.parameters, self.outputs)
        if self.applied_depth == 0:
            raise RuntimeError('previous iteration failed to stop before exhaustion')
        info['builder'].params['max_depth'] = self.applied_depth
        self.overhead += time.perf_counter() - started

    def after_iteration(self, info):
        from experiments.a100_runner import quality
        started = time.perf_counter()
        tree = info['model'].models[-1]
        if tree.ngroups != 1:
            raise ValueError('controller requires SingleSplitter')
        self.parameters += int(tree.max_leaves * tree.values.shape[1])
        if self.parameters > self.cap:
            raise RuntimeError('coefficient cap exceeded')
        step = info['num_iter'] + 1
        self.validation_prediction += info['data']['valid']['last_tree']['preds'][0]
        if step > 60:
            self.block_depths.append(self.applied_depth)
        cap_stop = affordable_depth(1, self.cap - self.parameters, self.outputs) == 0
        stop, stop_reason = cap_stop or step == self.horizon, ('cap' if cap_stop else 'horizon')
        if step % 20 == 0 or stop:
            valid = info['data']['valid']
            p = info['model'].loss.postprocess_output(self.validation_prediction).get()
            target = valid['target'][0].get()
            measured = quality(target, p, self.task)
            gap = deficit(Quality(**measured), self.reference)
            if step >= 60 and step % 20 == 0:
                self.confirmations = self.confirmations + 1 if gap == 0 else 0
            completed = None
            censored = False
            if self.block_start is not None and step - self.block_start[0] == 20:
                start_step, start_p, start_gap, requested = self.block_start
                if requested in (3, 5) and all(d == requested for d in self.block_depths):
                    completed = WindowGain(requested, start_step, step,
                                           self.parameters - start_p, start_gap, gap)
                    self.windows.append(completed)
                else:
                    censored = True
            elif self.block_start is not None and stop and step > self.block_start[0]:
                censored = True  # Incomplete terminal windows are not rate observations.
            if self.stop_on_quality and self.confirmations >= 2:
                stop, stop_reason = True, 'quality_confirmed_twice'
            decision = Decision(self.depth, 'terminal' if stop else 'common_warmup')
            if not stop and step >= 60:
                if self.mode == 'fixed':
                    decision = Decision(5, 'fixed_depth')
                elif self.mode == 'alternating':
                    decision = Decision(3 if ((step - 60) // 20) % 2 == 0 else 5, 'alternating_schedule')
                else:
                    decision = plan_depth(step=step, remaining_parameters=self.cap-self.parameters,
                        remaining_trees=self.horizon-step, current_deficit=gap, windows=self.windows,
                        history=1 if self.mode == 'latest' else 2, incumbent=self.depth,
                        deceleration=self.mode == 'deceleration')
                    if not self.stop_on_quality and decision.reason == 'quality_confirmation_pending':
                        decision = Decision(decision.depth, 'target_met_budget_continuation', decision.forecasts)
                self.block_start = (step, self.parameters, gap, decision.depth)
                self.block_depths = []
            check = dict(step=step, parameters=self.parameters, validation=measured,
                         requested_depth_before=self.depth,
                         deficit=gap, confirmations=self.confirmations, next_depth=decision.depth,
                         decision_reason=decision.reason, stop=stop, stop_reason=stop_reason if stop else None,
                         completed_window=asdict(completed) if completed else None,
                         censored_window=censored, forecasts=[asdict(f) for f in decision.forecasts])
            self.checks.append(check)
            self.emit('quality_efficiency_check', **check)
            self.depth, self.reason = decision.depth, decision.reason
        self.trace.append(dict(step=step, parameters=self.parameters, applied_depth=self.applied_depth,
                               next_depth=self.depth, next_reason=self.reason, stop=stop))
        self.overhead += time.perf_counter() - started
        return stop
