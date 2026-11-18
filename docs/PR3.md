# 04. GPU determinism proof: end-to-end reproducibility and atomic flip rate

Scope: the opt-in deterministic histogram / leaf-value kernels in
`py_boost/gpu/utils.py` (`deterministic_histogram`, `deterministic_histogram_layout`,
`deterministic_leaf_values`). The original report addressed two questions with
dedicated GPU checks and artifacts.

- **Q2** — is a full model fit bitwise reproducible run-to-run in deterministic mode?
- **Q3** — how often does the default atomic mode actually change tree structure
  between identical seeded runs (the null hypothesis the deterministic kernels
  were built against, never previously measured)?

## Historical RTX 5080 results

The historical observations below come from the original RTX 5080 report.
The restored harness uses its own versioned synthetic workload. Recomputing
the historical numbers at tighter tolerances requires the original Q3 split
comparisons or the corresponding trees with debug data.

### Environment and protocol

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
- The reported maximum absolute logit differences of 3.58e-7 and 4.17e-7
  were measured on the training inputs.
- Leaf-value-only divergences (structure identical, values differ): 15–23 trees
  per pair; worst leaf-value drift 1.02e-6.

## Observation 3 (Q3, tolerant comparison): zero hard flips at the recorded tolerances

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
- Every first split divergence within a tree occurred at depth 5 in these runs.

## Observation 4: tolerance scale matters, and the intended sweep mislabeled itself

- The repo's refinement-application audit (`5e-3 + 5e-5*|g|`,
  `py_boost/gpu/utils.py`) answers a different question — whether two
  *different computations* picked the same winner — and is one to two orders
  of magnitude looser than the measured atomic noise floor (~1e-6 relative,
  sqrt(n)*u walk).
- Three artifacts tagged `atol1e-4/1e-5/1e-6` all record `gain_atol=0.005`:
  the flags never reached the script; they are repetitions, not a sweep.
  Lesson: always verify `config.tolerances` in the artifact.
- The RTX 5080 result of zero hard flips applies to the recorded audit
  tolerances. The RTX 4080 follow-up below reports the tighter sweep on a
  new workload.

## Observation 5: integer paths were never non-deterministic

`count_histogram` (uint64 `cp.add.at`), node counts (`scatter_add` on uint64),
node ids, and NaN counts are exactly associative under reordering — the noise
enters only through two float kernels: `histogram_kernel_idx` (grad/hess
histograms) and `accumulate_gh_kernel` (leaf values).

## Historical conclusions for the RTX 5080 workload

| claim | requirement | verdict |
|---|---|---|
| training-input logits | compare raw logits | maximum observed atomic difference was about 4.2e-7 |
| split differences at the recorded audit tolerances | compare matched paths | 0/630 hard flips |
| bitwise fingerprints and exact split identities | repeated seeded fits | the deterministic arm matched bitwise, and 0/21 atomic pairs matched on logits |

Recommended policy: keep deterministic kernels opt-in (default off);
require them for any experiment archiving fingerprints or comparing exact
split identities.

## Open items

1. Replaying the historical RTX 5080 sweep requires the original split-level
   records. The new RTX 4080 sweep is complete.
2. Measure the runtime and memory costs of deterministic training across
   dispatch layouts and workload sizes.
3. Check how exact split differences and held-out predictions vary with
   dataset size, tree count and the choice of held-out split.

## Artifact index (results/proving/)

| file | content |
|---|---|
| `end_to_end_determinism_NVIDIA_GeForce_RTX_5080.json` | Q2 PASS, 3 replicas, strict + tolerant detail |
| `atomic_flip_rate_NVIDIA_GeForce_RTX_5080.json` | Q3 at 5e-3 (Nov 3) |
| `atomic_flip_rate_atol{1e-4,1e-5,1e-6}_...json` | Q3 repetitions at 5e-3 (mislabeled tags, Nov 5) |
| `atomic_flip_rate_smoke_...json`, `end_to_end_determinism_smoke_...json` | tiny-config smoke runs |

## RTX 4080 follow-up on 2026-11-18

The follow-up used source commit `9b95dd1`, an RTX 4080, Windows, CuPy 13.3.0
and CUDA 12.6. The test processes disabled CuPy accelerators. Training used
cross-entropy, 30 trees of depth at most 6, learning rate 0.1, a minimum of 10
rows per leaf, full leaf values, all rows and columns, the `hybrid` histogram
layout and seed 17. Both deterministic flags changed together.

### Q3 rerun and CPU tolerance sweep

The `balanced_gaussian_v1` workload had 32,768 rows, 16 features and six
classes. Three separate processes each fitted eight models per arm and
compared replicas 1–7 with replica 0 of the same arm. All 24 deterministic
models matched bitwise on structures, leaf values and training-input logits
across processes.

In the atomic arm, 442 of 630 tree comparisons differed under the strict
criterion. The only difference in all 820 differing split comparisons was
the bin index (820 of 38,766 compared split paths). The largest absolute
logit difference on training inputs was 6.91e-6.

The table reclassifies the same saved comparisons on CPU at each gain
tolerance. The bin tolerance remained 1. Each changed tree is counted once
under its most severe split category.

| `gain_atol` | `gain_rtol` | `near_tie` trees | `gain_equivalent` trees | `hard_flip` trees out of 630 |
|---:|---:|---:|---:|---:|
| `5e-3` | `5e-5` | 131 | 311 | 0 |
| `1e-4` | `2e-6` | 131 | 311 | 0 |
| `1e-5` | `2e-7` | 131 | 288 | 23 (3.7%) |
| `1e-6` | `2e-8` | 131 | 103 | 208 (33.0%) |

At the tightest tolerance, all 252 `hard_flip` nodes were at depths 4–5
(root at depth 0). Their bin gaps had a median of 3 and ranged from 2 to 14.
These rates describe the observed pairs. Trees from the same fit and
comparisons with a shared reference are dependent.

### Held-out checks

Each dataset used one fixed train/test split for both arms. The synthetic
test used independent draws around the original Q3 class centers
(`holdout_seed=170817`). Pendigits and Satimage used their original train/test
files. Helena and MNIST reused the archived 12,000-row sample and split
(`hybrid_fit_1_seed314`, `split_seed=242`). The held-out runs used the same
three-process, eight-model-per-arm protocol. All 24 deterministic fits on
each dataset matched bitwise on structure, leaf values and test logits.

D is the deterministic result and A is the range across atomic fits.
Log loss is computed from raw logits in float64. The switch counts are the
numbers of test objects with different top-1 classes per pair, over 24
comparisons between modes and 21 comparisons within the atomic arm.

| Dataset | Train / test rows | Accuracy %, D / A | Macro-F1 %, D / A | Log loss, D / A | Class switches, A vs D / A vs A |
|---|---:|---:|---:|---:|---:|
| Synthetic Q3 | 32,768 / 8,192 | 96.667 / 96.667 | 96.670 / 96.670 | 0.134790 / 0.134775–0.134817 | 0 / 0 |
| Pendigits | 7,494 / 3,498 | 95.855 / 95.769–95.826 | 95.883 / 95.795–95.855 | 0.167359 / 0.170071–0.170649 | 14–21 / 0–13 |
| Satimage | 4,435 / 2,000 | 89.850 / 89.850 | 88.252 / 88.252 | 0.292365 / 0.292127–0.292476 | 0 / 0 |
| Helena | 9,600 / 2,400 | 30.583 / 30.583–30.625 | 13.452 / 13.452–13.472 | 3.004249 / 3.004036–3.004240 | 0–1 / 0–1 |
| MNIST | 9,600 / 2,400 | 92.708 / 92.708 | 92.615 / 92.615 | 0.301700 / 0.301542–0.301717 | 0 / 0 |

- On the synthetic test, the largest logit difference between atomic fits
  reached 0.286. Their predicted classes agreed on all 8,192 test objects.
- On Pendigits, the deterministic model gave 1–3 more correct answers and
  lower log loss than each atomic fit.
- On Helena, the same object switched classes in 7 of the 24 comparisons
  between modes. Each switch gave the atomic model one extra correct answer.
  Log loss also favored the atomic mode by 0.000009–0.000213.
- Satimage and MNIST agreed exactly on classes, accuracy and macro-F1.
  Their log-loss differences were small and changed sign across pairs.

The follow-up supports using the deterministic mode when an experiment
requires exact split or prediction comparisons. Held-out quality effects
varied by dataset on these fixed splits.

### Evidence retained for this follow-up

The local experiment archive contains `q3-summary.json` for the three raw Q3
runs and nine CPU replays, `q3-heldout-summary.json` for the synthetic and
Pendigits checks, and `q3-real-heldout-summary.json` for Satimage, Helena
and MNIST. The summaries record input and split fingerprints, per-pair
metrics where applicable, and checksums of the raw JSON or NPZ files.

## Running the restored Q2 and Q3 checks

```sh
python experiments/prove_end_to_end_determinism.py --replicas 2 --ntrees 3 --nrows 512 --nfeatures 8 --nclasses 4 --max-depth 3 --min-data-in-leaf 5 --tag smoke
python experiments/prove_atomic_flip_rate.py --replicas 2 --ntrees 3 --nrows 512 --nfeatures 8 --nclasses 4 --max-depth 3 --min-data-in-leaf 5 --tag smoke
```

The JSON records the effective tolerances, workload fingerprint and source
fingerprint. The comparison checks raw logits before softmax. It matches nodes
by output group and path from the root. An internal node contributes one
comparison if either replica splits at that path. A near tie requires matching
feature, NaN direction and child types, with thresholds in neighboring bins
within the configured distance. A tree is assigned its most severe split
category. Leaf-value tolerances apply to trees with matching structure. The
Wilson intervals are descriptive because trees from one fit and comparisons
against the same reference replica are dependent.

Q3 can reclassify split decisions on CPU from a report produced by the restored
harness. Use the full JSON file, including the paired split decisions and their
gains. Unspecified tolerances come from the source report. Leaf-value results
are copied at their saved tolerances.

```sh
python experiments/prove_atomic_flip_rate.py --replay saved-q3.json --gain-atol 1e-4 --gain-rtol 2e-6 --tag tighter
```

For the historical sweep, recover the per-node `elementwise` records for both
arms or the original models with debug split arrays. The aggregate counts in
this note show the original tolerance and cannot determine the counts at a
new tolerance.
