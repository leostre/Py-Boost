# PR1 — HASBoost semantics: summary and review notes

Branch: `hasboost/20261026/pr01-semantics` (GitHub PR #3). The original review
covered these commits:

```
f7bdb96 Publish HASBoost 01: semantics
b2842af docs: enhance GradHessHistory and sketch methods with detailed docstrings
dc08787 feat: add gradient/hessians history accumulation (#1)
```

Sources: the branch diff, historical commits, and a research snapshot. The snapshot's
`history_callback copy.py` contains merge-conflict markers and is absent from this PR.
The [source manifest](research/source_manifest.json) records the original publication
snapshot, before the review fixes described here.

---

## 1. What the branch introduces

### New modules

| File | Content |
|---|---|
| `py_boost/gpu/accumulation/history_callback.py` | `GradHessHistory` (a `GradSketch` callback): EMA of per-sample grad/Hess norms, relative-volatility signal, stable-regime scheduler; `WeightedHistorySampling` subclass adds deviation-based sampling weights |
| `py_boost/gpu/accumulation/sketches/sketch_methods.py` | Sketch registry `SUPPORTED_SKETCH_METHODS`: `svd_heuristic` (smooth spectral thresholding), `svd` (leverage scores), `topk`, `rand` (Gumbel-top-k), `proj` |
| `py_boost/multioutput/sketch_decision.py` | `SketchDecision` (row/output indices + optional `p_i, q_j` marginals), `StructureIndexers`, `marginal_probabilities`, leaf-row resolution |
| `py_boost/multioutput/history_state.py` | Pure scheduling rules: `update_scale_history` (EMA), `relative_volatility`, `decide_stable_regime` (warmup + hysteresis) — cupy-free, unit-testable |
| `py_boost/gpu/history_boosting.py` | `HistoryBasedBoostingModel`: routes `history_subsample` to the sketch and enables `full_leaf_values` |

### Modified core

- `boosting.py`: `full_leaf_values`, support for `subsample=None`, and early detection of competing row samplers.
- `tree.py`: builder consumes `sketch.make_decision()` — rows *and* outputs subsampled
  for structure search; features kept separate; guard requiring
  `target_splitter='Single'`; leaf values re-computed on all rows when enabled.
- `quantization/utils.py`: numba threading-layer fallback; `__init__.py`: version fallback.

### Aim

Implement **HASBoost — history-aware sketching for multi-output gradient boosting**:
early iterations run on full data while gradient statistics are volatile; once
per-sample norm changes stabilize (low volatility), rows and outputs are subsampled
(weighted by deviation from the norm history) to cut structure-search cost. Leaf
values are estimated on all rows and outputs of the chosen structure. Per
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

### `SketchDecision` is the result of every built-in method
The builder uses row and output indices. `topk` ranks the normalized scores
`p_i, q_j`. `rand` uses them as weights for sampling without replacement. These
weights describe the sequential draws and differ from the probabilities of inclusion
in the final set. Use `sketch_params={'record_probabilities': True}` to attach the
scores to the decision for diagnostics. SVD and projection methods return indices
without probability metadata. `__iter__` preserves `(rows, outputs)` unpacking, and
`GradHessHistory.make_decision` adapts the index pair from legacy external methods.

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
snapshot. The transforms decrease with the signed product of the two deviations.
The base `GradHessHistory` supplies signed gradients and rejects `topk` and `rand`
at construction. Both methods are available through `WeightedHistorySampling`,
which supplies non-negative weights. Subclasses that provide their own decision
tensor are responsible for matching the method's input requirements.

### `subsample=1.0` vs `history_subsample`
`HistoryBasedBoostingModel(history_subsample=0.4, ...)` gives the history sketch
the row fraction. The builder uses all rows while the sketch is inactive. The
ordinary sampler is absent by default (`subsample=None`), including from the
callback pipeline. `GradientBoosting` allows `None` or `1.0` with a sketch that
provides `make_decision`; a competing sampler raises `ValueError` at construction.
For compatibility, a numeric `subsample` passed alone to `HistoryBasedBoostingModel`
still sets the history fraction and emits a `FutureWarning`.

An option supplied only inside `sketch_params` remains valid. Conflicting values
for `subsample`, `sketch_outputs`, or `tau` at the two levels are rejected. The
effective fraction is stored in `params['history_subsample']`. Unused model options
and unrecognized options of the built-in history callbacks emit warnings. Custom
factories still receive the settings they may need, such as `seed`.

### `make_decision` call site
`DepthwiseTreeBuilder.build_tree` calls the decision provider after preparing the
gradient and Hessian for structure search. In the supported single-group path it
calls the provider once per tree, including when there is only one output. The
single output is retained and the selected rows are used. The old output-only
`__call__(grad, hess)` path remains available for external sketches.

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
Adaptive: warmup/volatile iterations build structure on **all rows**; stable
iterations build on `decision.row_indices` (~`history_subsample`) and
`decision.output_indices`. Effective sketch rate = `decisions_applied / ntrees`.
Unavoidable O(n) floor per iteration: volatility is computed on full grad norms, and
grad/hess + leaf values (`full_leaf_values`) are full-data.

---

## 3. Hyperparameters

Model options include `full_leaf_values` and `history_subsample`. Sketch options include `sketch_method`,
`sketch_outputs`, `sketch_params`, `smoothing_alpha` (default `0.1**(1/9) ≈ 0.774`),
`stabilization_threshold` (0.2), `destabilization_threshold` (= enable),
`min_warmup_rounds` (2), `eps` (1e-6), `tau` (1e-3), `weight_transform`
(`sigmoidal`/`hyperbolic`); inside `sketch_params`: `lambda_max_ratio` (0.01),
`gamma` (1.0) for `svd_heuristic`. Net vs `b2842af`:
+`destabilization_threshold`, `min_warmup_rounds`, `tau`; −`stabilization_window`;
`stabilization_threshold` default changed 1.0 → 0.2.

**Configuration and existing sweeps**
- Put `lambda_max_ratio` and `gamma` in `sketch_params`. Passing either at the top
  level to a built-in history callback emits an unused-parameter warning.
- `stabilization_window` existed at `b2842af` and now emits an unused-parameter warning.
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

## 4. Review changes

The fixes cover incompatible signed-gradient methods, ownership of row selection
and configuration conflicts, single-output row selection, and the common sketch
result. The regression checks inspect the actual structure and leaf calls during
a short fit on synthetic data.

---

## 5. Research questions

1. Compare the two volatility normalizations, warmup, hysteresis and `tau` before
   choosing defaults. The current default sets the two thresholds equal. Adding
   the same `tau` to all scores leaves the `topk` ordering unchanged in exact arithmetic.
2. The original SketchBoost and its `SketchBoost` class use one tree for all outputs.
   For a broader Py-Boost comparison, add a separate grouped-output control and
   report actual tree and leaf counts alongside training time and quality.

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
- Two-stage split screening (screen features, then score retained candidates more accurately).
  Measure how often screening discards the best feature.

Sampling changes the candidate splits or the statistics used to score them.
Recomputing leaf values on all rows and outputs estimates coefficients for the
chosen structure. It cannot restore a split eliminated during structure search.

---

## 6. Test coverage

- `tests/test_history_state.py` — volatility arithmetic, zero-init EMA, warmup +
  hysteresis transitions, EMA bounds (property-based).
- `tests/test_sketch_decision.py` — axis separation regression, output-index
  application (incl. single-column Hess proxy), full-leaf row resolution, marginal
  formula, uniform fallback, product-distribution contract (property-based).
- `tests/test_history_boosting_config.py` — deprecated `subsample` alias, disabled
  base row sampler and full leaf values.
- `tests/test_history_gpu.py` — configuration errors, all five built-in methods,
  opt-in probabilities, tuple compatibility and actual fits with one and three outputs.
- `tools/hasboost_cpu_tests.py` — CPU entry point, with the GPU module skipped explicitly.

```text
python -X utf8 tools/hasboost_cpu_tests.py tests -q -ra
python -X utf8 -m pytest tests/test_history_gpu.py -q -ra
```

The review run passed 13 CPU tests (the GPU module was skipped there) and
11 GPU tests on a GeForce RTX 4080 with CuPy 13.3.0. The GPU run used short
ASCII include paths for NVRTC and disabled CUB accelerators for the test process
to work around local header lookup failures.

Both fit checks used 192 rows, eight features and three trees. Structure search
used 192, 96 and 96 rows. Leaf estimation used all 192 rows and every output at
each iteration. In the three-output case, structure search used two outputs on
the last two iterations. The four feature indices came from `colsample` in both
fits. These are correctness checks on synthetic data.
