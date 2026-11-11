# 04. GPU determinism proof: end-to-end reproducibility and atomic flip rate

Scope: the opt-in deterministic histogram / leaf-value kernels in
`py_boost/gpu/utils.py` (`deterministic_histogram`, `deterministic_histogram_layout`,
`deterministic_leaf_values`). Two previously open questions were closed with
dedicated proving scripts and GPU artifacts:

- **Q2** — is a full model fit bitwise reproducible run-to-run in deterministic mode?
- **Q3** — how often does the default atomic mode actually change tree structure
  between identical seeded runs (the null hypothesis the deterministic kernels
  were built against, never previously measured)?

## Environment and protocol

- Hardware: NVIDIA GeForce RTX 5080, compute capability 12.0, cupy 13.6.0.
- Workload: synthetic 6-class problem, 32 768 rows x 16 features, crossentropy,
  30 trees, max_depth 6, `subsample=1.0`, `colsample=1.0`, `full_leaf_values=True`,
  `min_gain_to_split=0`, `lr=0.1`, seed 17.
- Protocol: R replicas of the identical fit per arm; every replica compared
  against replica 0 at prediction, tree-structure, and leaf-value level.
  Atomic arm: both deterministic flags off (only GPU scheduling differs between
  replicas). Control arm: both flags on (must be clean, else the atomic arm is
  uninterpretable).
- Scripts: `experiments/prove_end_to_end_determinism.py`,
  `experiments/prove_atomic_flip_rate.py`, shared harness
  `experiments/proving_common.py`. Artifacts:
  `results/proving/*.json` (config, tolerances, environment, per-node
  element-wise records, verdicts).

## Observation 1 (Q2): deterministic mode is bitwise reproducible end-to-end — PASS

- 3 replicas: predictions, every tree structure, every leaf value — bitwise
  identical (SHA-256 fingerprints).
- Atomic control arm deviates in the same check, so the test has
  discriminative power (not vacuously green).
- Reproduced across three separate process invocations on different runs:
  the deterministic control arm's fingerprints are byte-identical across all
  artifacts, i.e. reproducibility survives process restarts.

## Observation 2 (Q3, strict comparison): atomic mode is not bitwise reproducible

Strict = any integer difference in (feature, bin, NaN direction, child topology).

- 0/21 replica pairs had bitwise-identical predictions (7 pairs x 3 repetitions).
- Strict tree flip rate 34.8–48.1% (Wilson 95% CI ~±7pp) — the spread across
  repetitions is itself atomic noise in the experiment.
- First structural divergence at tree index 4–9 of 30 — divergence starts early
  and compounds through the ensemble.
- Prediction drift despite full structure churn: max abs diff 3.58e-7 / 4.17e-7
  per pair = **3 x 2^-23 and 7 x 2^-24**, i.e. 3 and 7 float32 ulps at logit
  magnitude in [1,2). Pure reordered-accumulation rounding, nothing else.
- Leaf-value-only divergences (structure identical, values differ): 15–23 trees
  per pair; worst leaf-value drift 1.02e-6.

## Observation 3 (Q3, tolerant comparison): all atomic noise is sub-tolerance

Tolerant element-wise classification of every divergent split node
(defaults: `bin_tol=1` adjacent bins; gain band `atol + rtol*|g|`):

| class | criterion |
|---|---|
| near_tie | same feature, same NaN direction, children aligned, bin differs by <= 1 |
| gain_equivalent | gains agree within the audit band (or leaf vs borderline split) |
| hard_flip | everything else |

Results (three repetitions at gain_atol=5e-3, gain_rtol=5e-5):

| repetition | strict flips | near_tie | gain_equivalent | hard_flip |
|---|---|---|---|---|
| run 1 | 73 trees | 54 | 19 | **0** |
| run 2 | 97 trees | 74 | 23 | **0** |
| run 3 | 101 trees | 71 | 30 | **0** |

- Exact internal consistency: near_tie + gain_equivalent = strict flips in
  every run; zero residue.
- **Pooled: 0 hard flips in 630 tree comparisons -> Wilson 95% upper bound
  on the hard-flip tree rate ~= 0.6%.**
- Every root divergence occurred at depth 5 (deepest level): atomic noise
  perturbs fringe near-tie decisions, never early high-stakes splits.

## Observation 4: tolerance scale matters, and the intended sweep mislabeled itself

- The repo's refinement-application audit (`5e-3 + 5e-5*|g|`,
  `py_boost/gpu/utils.py`) answers a different question — whether two
  *different computations* picked the same winner — and is one to two orders
  of magnitude looser than the measured atomic noise floor (~1e-6 relative,
  sqrt(n)*u walk).
- Three artifacts tagged `atol1e-4/1e-5/1e-6` all record `gain_atol=0.005`:
  the flags never reached the script; they are repetitions, not a sweep.
  Lesson: always verify `config.tolerances` in the artifact.
- Consequence: "0 hard flips" is currently proven only at application-audit
  scale. Behavior at gain_atol in {1e-4, 1e-5, 1e-6} is still unmeasured.

## Observation 5: integer paths were never non-deterministic

`count_histogram` (uint64 `cp.add.at`), node counts (`scatter_add` on uint64),
node ids, and NaN counts are exactly associative under reordering — the noise
enters only through two float kernels: `histogram_kernel_idx` (grad/hess
histograms) and `accumulate_gh_kernel` (leaf values).

## Verdicts by claim type

| claim | requirement | verdict |
|---|---|---|
| accuracy / metrics | any | atomic mode sufficient (drift <= 4.2e-7 on logits) |
| structure equivalence at audit tolerance 5e-3 | atomic | sufficient: 0/630 hard flips, bound 0.6% |
| bitwise fingerprints, exact split identities, byte-level artifact comparison | deterministic | atomic impossible (0/21 pairs); deterministic proven (Q2 PASS) |
| research diagnostics on exact identities (split traces, refinement audits) | deterministic | implementation is necessary and inevitable — no upper-level switch can remove atomics from those two kernels |

Recommended policy: keep deterministic kernels opt-in (default off);
require them for any experiment archiving fingerprints or comparing exact
split identities.

## Open items

1. Real tolerance sweep (verify `config.tolerances` after each run):
   `python3 experiments/prove_atomic_flip_rate.py --gain-atol 1e-4 --gain-rtol 2e-6 --tag sweep1e-4`
   (likewise 1e-5/2e-7, 1e-6/2e-8).
2. Cost side of the deterministic path (canonicalization overhead:
   host sort + PCIe per histogram call; perf1/perf2-hybrid dispatch policy) —
   never measured; no timing artifacts exist under `results/`.
3. Scale dependence: atomic noise grows as sqrt(n) per bin; this proof is at
   32k rows, the repo benchmarks go to ~1M rows.
4. The flip-rate Wilson bounds bound, not eliminate, rare hard flips;
   `--replicas` tightens them.

## Artifact index (results/proving/)

| file | content |
|---|---|
| `end_to_end_determinism_NVIDIA_GeForce_RTX_5080.json` | Q2 PASS, 3 replicas, strict + tolerant detail |
| `atomic_flip_rate_NVIDIA_GeForce_RTX_5080.json` | Q3 at 5e-3 (Nov 3) |
| `atomic_flip_rate_atol{1e-4,1e-5,1e-6}_...json` | Q3 repetitions at 5e-3 (mislabeled tags, Nov 5) |
| `atomic_flip_rate_smoke_...json`, `end_to_end_determinism_smoke_...json` | tiny-config smoke runs |