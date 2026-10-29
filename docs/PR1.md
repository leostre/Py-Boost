# PR1 — HASBoost semantics: summary and review notes

Branch: `hasboost/20261026/pr01-semantics` (3 commits on top of the `experiments` lineage)

```
f7bdb96 Publish HASBoost 01: semantics
b2842af docs: enhance GradHessHistory and sketch methods with detailed docstrings
dc08787 feat: add gradient/hessians history accumulation (#1)
```

Sources: branch diff vs merge-base, intermediate commits extracted with `git show`,
and the research snapshot `py_boost/gpu/accumulation/history_callback copy.py`
(raw leftover, still contains unresolved `<<<<<<< HEAD` conflict markers — not runnable).

---

## 1. What the branch introduces

### New modules

| File | Content |
|---|---|
| `py_boost/gpu/accumulation/history_callback.py` | `GradHessHistory` (a `GradSketch` callback): EMA of per-sample grad/Hess norms, relative-volatility signal, stable-regime scheduler; `WeightedHistorySampling` subclass adds deviation-based sampling weights |
| `py_boost/gpu/accumulation/sketches/sketch_methods.py` | Sketch registry `SUPPORTED_SKETCH_METHODS`: `svd_heuristic` (smooth spectral thresholding), `svd` (leverage scores), `topk`, `rand` (Gumbel-top-k), `proj` |
| `py_boost/multioutput/sketch_decision.py` | `SketchDecision` (row/output indices + optional `p_i, q_j` marginals), `StructureIndexers`, `marginal_probabilities`, leaf-row resolution |
| `py_boost/multioutput/history_state.py` | Pure scheduling rules: `update_scale_history` (EMA), `relative_volatility`, `decide_stable_regime` (warmup + hysteresis) — cupy-free, unit-testable |
| `py_boost/gpu/history_boosting.py` | `HistoryBasedBoostingModel`: wires the sketch, forces `subsample=1.0` + `full_leaf_values=True` |

### Modified core

- `boosting.py`: new `full_leaf_values` flag, `**kwargs` catch-all.
- `tree.py`: builder consumes `sketch.make_decision()` — rows *and* outputs subsampled
  for structure search; features kept separate; guard requiring
  `target_splitter='Single'`; leaf values re-computed on all rows when enabled.
- `quantization/utils.py`: numba threading-layer fallback; `__init__.py`: version fallback.

### Aim

Implement **HASBoost — history-aware sketching for multi-output gradient boosting**:
early iterations run on full data while gradient statistics are volatile; once
per-sample norm changes stabilize (low volatility), rows and outputs are subsampled
(weighted by deviation from the norm history) to cut structure-search cost, while
full-data leaf re-computation (`full_leaf_values`) preserves quality. Per
`docs/research/01-semantics.md`, this PR is a **publication of accumulated research
code** matching the original HASBoost formulation (axis separation, full leaf
recomputation, `p_i/q_j` marginals), not a new scientific run; historical GPU
measurements were not repeated on this build.

---

## 2. Design decisions (Q&A findings)

### Gumbel-top-k sampling
CuPy lacks `choice(replace=False, p=...)`. Keys `log p_i + G_i`, `G_i ~ Gumbel(0,1)`,
top-k = exact Plackett–Luce sampling without replacement, fully vectorized on GPU.
History: upstream used `cp.random.choice(replace=True, p=...)` (with replacement,
outputs only); the first branch commit (`dc08787`) still sampled rows *with*
replacement and hardcoded `smooth=0.1`; HEAD is the exact law with configurable `tau`.
Pre-`make_decision` versions injected indexers into `build_tree` via
`inspect.stack()` + `PyFrame_LocalsToFast` frame mutation — removed at HEAD.

### `SketchDecision` holds both indexers and probabilities
Indexers are the actionable result; marginals `p_i, q_j` are the provenance of the
draw (the article-defined distributions), obtained for free during sampling and kept
for diagnostics/ablations. `__iter__` preserves legacy `(rows, cols)` unpacking.

### Hysteresis (`stabilization` / `destabilization` thresholds)
Schmitt trigger: enter stable regime when `volatility < enable`; leave only when
`volatility ≥ disable` — prevents enable/disable chatter. Unit-tested
(`test_scheduler_applies_warmup_and_hysteresis`, all four transitions).
**Default `disable = enable` (0.2), i.e. hysteresis degenerates unless widened.**
No ablation exists for it (experiments predate the parameter).
`min_warmup_rounds` default `2` (1-based counter `num_iter + 1`), validated `≥ 1`.

### Volatility
`volatility = ‖v_g^(t) − v_g^(t−1)‖₁ / (‖EMA(v_g)^(t−1)‖₁ + ε)`,
`v_g^(t)` = per-sample L2 norms of gradient rows. Scheduling uses **grad norms only**
(hess history feeds the weights, not the scheduler). Low volatility ⇒ sketching.

### `relative_volatility` vs original `grad_delta`
Numerator identical. Denominator changed: original normalized by an EMA over
**aggregated deltas** ("change vs typical changes"); published version normalizes by
the L1 of the per-sample EMA of **raw norms** ("change vs signal magnitude").
Same spirit, different reference frame — this is part of the "semantics" rewrite.

### `_decision_tensor` hook
Template-method hook. Base returns raw `grad`; `WeightedHistorySampling` overrides
it with the accentuation matrix
`weights = transform((g/(|g_hist|+ε) − 1) · (h/(|h_hist|+ε) − 1))`
(sigmoidal or min-max "hyperbolic"), computed against the **pre-update** history
snapshot (no look-ahead). Lower weight = larger deviation ⇒ deviating rows/outputs
are sampled *less*. `make_decision` normalizes both return contracts
(`SketchDecision` from `topk`/`rand`; legacy 2-tuples from `svd`/`svd_heuristic`/`proj`).

### `subsample=1.0` vs `history_subsample`
`subsample=1.0` disarms the base Bernoulli sampler so the sketch decision is the
single owner of row sampling (otherwise two samplers compose and corrupt the
volatility signal). `history_subsample` only preserves the user's value in `params`
as provenance. A cleaner contract: `subsample=None` support + explicit conflict
validation instead of silent override (see §4).

### `make_decision` call site
Invoked inside `DepthwiseTreeBuilder.build_tree` because only there are structure-search
grad/hess in scope, and the legacy `__call__(grad, hess)` protocol cannot carry row
indices. State stays in the callback (py-boost's per-iteration hook point); the
`_fit`/`CallbackPipeline` alternative in `history_boosting.py` is commented out
(abandoned). Suggested cleanup: hoist the call above the output-group loop.

### Axis semantics in the builder
`col_indexer` indexes **dataset features** (`colsampler()` / full `arange(X.shape[1])`)
and is passed to `depthwise_grow_tree` **unchanged**; `resolve_structure_indexers`
guarantees features never come from the decision (regression:
`test_output_indices_never_replace_feature_indices`). `output_indices` index
gradient/Hess columns (output space) — a disjoint axis. The `'Single'` guard only
forbids grouped-output splitters; under `Single` the decision applies to the full
output width, so **output sampling is moved, not removed**, and feature `colsample`
remains fully functional.

### Structure-search data regime
Adaptive: warmup/volatile iterations build structure on **all rows** (base sampler
disarmed); stable iterations build on `decision.row_indices` (~`subsample`) and
`decision.output_indices`. Effective sketch rate = `decisions_applied / ntrees`.
Unavoidable O(n) floor per iteration: volatility is computed on full grad norms, and
grad/hess + leaf values (`full_leaf_values`) are full-data.

---

## 3. Hyperparameters

**13 new knobs**: model level `full_leaf_values`; sketch level `sketch_method`,
`sketch_outputs`, `sketch_params`, `smoothing_alpha` (default `0.1**(1/9) ≈ 0.774`),
`stabilization_threshold` (0.2), `destabilization_threshold` (= enable),
`min_warmup_rounds` (2), `eps` (1e-6), `tau` (1e-3), `weight_transform`
(`sigmoidal`/`hyperbolic`); inside `sketch_params`: `lambda_max_ratio` (0.01),
`gamma` (1.0). `subsample` is repurposed, not new. Net vs `b2842af`:
+`destabilization_threshold`, `min_warmup_rounds`, `tau`; −`stabilization_window`;
`stabilization_threshold` default changed 1.0 → 0.2.

**kwargs reachability issues**
- `lambda_max_ratio`, `gamma` are only reachable via the nested `sketch_params` dict;
  top-level values are silently swallowed by `GradientBoosting`'s `**kwargs`.
- The `**kwargs` catch-all turns typos into silent no-ops (previously `TypeError`).
- `stabilization_window` (existed at `b2842af`) is silently discarded at HEAD.
- Sweep grids in `experiments/runners/*` and `ablations/*` cover only
  `sketch_method`, `sketch_outputs`, `smoothing_alpha` (0.7–0.95),
  `stabilization_threshold` (0.5–2.0, old default 1.0); the new HEAD parameters
  (`destabilization_threshold`, `min_warmup_rounds`, `tau`, `weight_transform`)
  are reachable via config but not wired into any sweep.

**EMA initialization**: all generations start from **zeros**, not from the 0th norms
(`copy.py`: `zeros_like`; HEAD: `current_scale * 0`) — uncorrected warm-up bias in
all of them. The warm-start-from-first-gradient behavior the author remembers exists
only in `FullGHistorySampling` (snapshot-only, behind conflict markers, not published).

---

## 4. Identified issues

1. `destabilization_threshold` defaults to the enable threshold ⇒ hysteresis is
   effectively off unless configured; never ablated.
2. Silent `**kwargs` swallowing (typo footgun) at both model and sketch level.
3. Sketch-method-specific args locked behind nested `sketch_params`; `tau` defaults
   inside `topk` (0.0) / `rand` (1e-3) are dead code on the callback path
   (`get_indexers` `setdefault`s 1e-3).
4. Inconsistent sketch return contracts (`SketchDecision` vs tuple) — normalization
   happens at runtime in `make_decision`.
5. `history_subsample` is written into `params` after `super().__init__` — API wart;
   one name (`subsample`) carries two meanings.
6. Diverged scheduler semantics vs original `grad_delta` logic (documented, but
   worth flagging for the paper-implementation correspondence claim).
7. `history_callback copy.py` contains unresolved conflict markers — should not ship.

---

## 5. Recommendations

1. Return `SketchDecision` from all five sketch methods (leverage scores as
   normalized probabilities for `svd`/`proj`); keep tuple normalization only for
   external legacy sketches.
2. Clean sampler disarm: accept `subsample=None` in the base class, raise on
   conflict with decision-capable sketches; introduce a first-class
   `history_subsample` param routed only into sketch kwargs.
3. Validate hyperparameters loudly: explicit signature or unknown-key warning;
   route per-method args through `sketch_params` consistently.
4. Hoist `make_decision` above the output-group loop in `build_tree`; formalize a
   small `DecisionProvider` protocol (callback keeps the state).
5. Default `destabilization_threshold` above the enable threshold (e.g. 2×) so the
   hysteresis the scheduler implements actually engages out of the box; add it to
   the ablation grid together with `min_warmup_rounds`.

### Feature-axis acceleration direction

Features are the last uncut axis of structure-search cost
(`rows × features × bins × depth`; currently only rows and outputs are sketched).
Options, in increasing ambition:

- `colsample` already samples features per tree, unconditionally (tree.py:512–515).
- Extend `SketchDecision` with `feature_indices` (field already exists in
  `StructureIndexers`) for history-gated feature sampling; preserve the
  outputs-never-alias-features invariant.
- Importance-weighted feature marginals from a features×outputs coupling tensor
  (cheap `scatter_add` on quantized bins), reusing `marginal_probabilities` +
  Gumbel-top-k and the EMA/deviation pattern — a fully symmetric treatment of all
  three axes under one stable-regime gate.
- Two-stage split screening (candidate features → verify top splits on all features).

Caveat: unlike row/output sketching, feature sampling has no `full_leaf_values`-style
unbiasedness correction — it changes which splits exist, so it stays a
bias/variance (regularization) trade.

---

## 6. Test coverage

- `tests/test_history_state.py` — volatility arithmetic, zero-init EMA, warmup +
  hysteresis transitions, EMA bounds (property-based).
- `tests/test_sketch_decision.py` — axis separation regression, output-index
  application (incl. single-column Hess proxy), full-leaf row resolution, marginal
  formula, uniform fallback, product-distribution contract (property-based).
- `tests/test_history_boosting_config.py` — `subsample → history_subsample`
  rerouting, `subsample=1.0`, `full_leaf_values=True`.
- `tools/hasboost_cpu_tests.py` — CPU entry point. All tests are cupy-free by design.
