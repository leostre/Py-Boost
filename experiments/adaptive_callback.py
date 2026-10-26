"""Effectful adapter for causal growth and bounded G/H history diagnostics."""
from collections import deque
import math
import time
import numpy as np
from py_boost.callbacks.callback import Callback
from experiments.adaptive_core import (
    growth_decision, affordable_depth, spectral_profile, spectral_state, output_budget,
    stable_profile_window,
)
from experiments.mechanism_core import channel_budget, static_budget


class CompactnessController(Callback):
    def __init__(self, *, mode, spectral, parameter_cap, seed, task, emit, improvement_floor=.002,
                 budget_channels='joint', static_fraction=None, static_depth_ceiling=7, activation_step=60):
        self.mode, self.spectral, self.cap = mode, spectral, int(parameter_cap)
        self.seed, self.task, self.emit = seed, task, emit
        channel_budget(1, 1, 7, budget_channels)
        if static_fraction is not None:
            static_budget(1, 0, static_fraction, static_depth_ceiling, activation_step)
            if spectral != 'none':
                raise ValueError('static and spectral budgets cannot be combined')
        self.budget_channels = budget_channels
        self.static_fraction, self.static_depth_ceiling = static_fraction, static_depth_ceiling
        self.activation_step = activation_step
        self.improvement_floor = float(improvement_floor)
        if not np.isfinite(self.improvement_floor) or not 0 <= self.improvement_floor < 1:
            raise ValueError('invalid validation improvement floor')
        self.losses, self.trace = [], []
        self.frames = deque(maxlen=4)
        self.profiles = deque(maxlen=4)
        self.frame_epochs = deque(maxlen=4)
        self.spectral_checks = []
        self.depth, self.depth_ceiling = 3, 7
        self.confirmed = None
        self.topk = None
        self.overhead = 0.

    def before_train(self, info):
        self.outputs = int(info['model'].base_score.size)
        self.parameters = self.outputs
        if affordable_depth(3, self.cap - self.parameters, self.outputs) < 1:
            raise ValueError('parameter cap cannot admit the first tree')
        n = info['data']['train']['target'].shape[0]
        self.probe = np.random.default_rng(self.seed + 70000).choice(n, min(n, 512), replace=False)
        sketch = info['builder'].multioutput_sketch
        if (self.spectral != 'none' or self.static_fraction is not None) and not hasattr(sketch, 'topk'):
            raise ValueError('spectral budget policy requires TopOutputsSketch')
        self.topk = int(getattr(sketch, 'topk', self.outputs))
        if self.spectral != 'none' or self.static_fraction is not None:
            # No reduction until a causal four-check window becomes reproducible.
            self.topk = self.outputs
            sketch.topk = self.topk

    def before_iteration(self, info):
        started = time.perf_counter()
        step = info['num_iter']
        if self.static_fraction is not None:
            budget = static_budget(self.outputs, step, self.static_fraction,
                                   self.static_depth_ceiling, self.activation_step)
            self.topk, self.depth_ceiling = budget.topk, budget.depth_ceiling
            info['builder'].multioutput_sketch.topk = self.topk
        state = None
        if self.spectral != 'none' and step % 20 == 0:
            train = info['data']['train']
            g = train['grad'][self.probe].get()
            h = train['hess'][self.probe].get()
            if h.shape[1] == 1:
                h = np.broadcast_to(h, g.shape).copy()
            frame = spectral_profile(g, h)
            self.frames.append((g, h))
            self.frame_epochs.append(step)
            self.profiles.append(frame)
            if self.spectral == 'history':
                # Axis convention in saved trace: probe rows x outputs x epochs.
                # Center/normalize within each epoch so large early G does not dominate.
                current = np.mean(self.profiles, axis=0)
                # Epochs are dependent observations: do not multiply the sample count.
                effective_rows = len(g)
            else:
                current, effective_rows = frame, len(g)
            noise = float(np.linalg.norm(spectral_profile(g[::2], h[::2]) -
                                         spectral_profile(g[1::2], h[1::2])))
            state = spectral_state(current, effective_rows)
            shifts = [float(np.linalg.norm(b - a)) for a, b in
                      zip(list(self.profiles)[:-1], list(self.profiles)[1:])]
            stable = stable_profile_window(shifts, noise, step, len(self.profiles))
            shift = (float(np.linalg.norm(current - self.confirmed)) if self.confirmed is not None
                     else float(np.linalg.norm(current)))
            event = stable and (self.confirmed is None or shift > 2 * max(noise, 1e-12))
            invalidated = not stable and self.confirmed is not None and shift > 2 * max(noise, 1e-12)
            if invalidated:
                self.topk = output_budget(self.outputs, state, False)
                self.depth_ceiling = 7
                self.confirmed = None
            if event:
                budget = channel_budget(self.outputs, output_budget(self.outputs, state, True),
                    min(7, max(3, math.ceil(math.log2(max(1., state['effective_rank']))) + 1)),
                    self.budget_channels)
                self.topk, self.depth_ceiling = budget.topk, budget.depth_ceiling
                self.confirmed = current.copy()
            info['builder'].multioutput_sketch.topk = self.topk
            check = dict(step=step, spectral=self.spectral, noise=noise, shift=shift,
                         budget_channels=self.budget_channels,
                         latest_shift=shifts[-1] if shifts else None,
                         median_shift=float(np.median(shifts)) if shifts else None,
                         stable=stable, budget_event=event, invalidated=invalidated, topk=self.topk,
                         depth_ceiling=self.depth_ceiling, frames=len(self.frames), **state)
            self.spectral_checks.append(check)
            self.emit('spectral_check', **check)
        feasible = affordable_depth(min(self.depth, self.depth_ceiling),
                                     self.cap - self.parameters, self.outputs)
        if feasible == 0:
            raise RuntimeError('controller failed to stop at the previous tree')
        info['builder'].params['max_depth'] = feasible
        self.applied_depth = feasible
        self.overhead += time.perf_counter() - started

    def after_iteration(self, info):
        import cupy as cp
        started = time.perf_counter()
        tree = info['model'].models[-1]
        if tree.ngroups != 1:
            raise ValueError('this experimental controller requires SingleSplitter')
        self.parameters += int(tree.max_leaves * tree.values.shape[1])
        if self.parameters > self.cap:
            raise RuntimeError('actual learned coefficients exceeded the cap')
        valid = info['data']['valid']
        p = info['model'].loss.postprocess_output(valid['ensemble'][0])
        target = valid['target'][0]
        if self.task == 'multiclass':
            p = p / p.sum(axis=1, keepdims=True)
            score = float(-cp.log(cp.clip(p[cp.arange(len(target)), target], 1e-7, 1-1e-7)).mean())
        else:
            p = cp.clip(p, 1e-7, 1-1e-7)
            score = float(-(target * cp.log(p) + (1-target)*cp.log1p(-p)).mean())
        self.losses.append(score)
        step = info['num_iter'] + 1
        decision = growth_decision(mode=self.mode, step=step, depth=self.applied_depth,
                                   losses=self.losses, remaining=self.cap-self.parameters,
                                   outputs=self.outputs, max_depth=self.depth_ceiling,
                                   improvement_floor=self.improvement_floor)
        self.depth = decision.depth
        row = {'step': step, 'parameters': self.parameters, 'validation_loss': score,
               'depth_ceiling': self.depth_ceiling,
               'depth': self.applied_depth, 'next_depth': decision.depth,
               'topk': self.topk, 'stop': decision.stop, 'reason': decision.reason}
        self.trace.append(row)
        self.overhead += time.perf_counter() - started
        if step % 20 == 0 or decision.stop:
            self.emit('growth_check', **row)
        return decision.stop
