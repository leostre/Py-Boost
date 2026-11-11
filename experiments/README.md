### Experiment launch guide

#### Reproducible HASBoost diagnostics

Install the pinned experiment dependencies and run a small GPU check first:

```bash
pip install -r experiments/requirements.txt
python -m experiments.benchmark_helena \
  --scenario main --datasets helena --max-samples 3000 --ntrees 5 --seed 42
```

The benchmark supports deterministic performance, observability, scheduler,
and trajectory scenarios:

```bash
# Five primary variants, three seeds, 12,000 Helena rows and 50 trees.
python -m experiments.benchmark_helena --scenario main --datasets helena

# Mechanism ablations: gating, p_i/q_j versus uniform allocation, one-axis
# sketches, full-data leaf values, and random versus greedy selection.
python -m experiments.benchmark_helena \
  --scenario ablations --datasets helena --ntrees 30 --seed 42

# Scheduler grid: theta in {0.1, 0.2, 0.3}, alpha in {0.8, 0.9, 0.95}.
python -m experiments.benchmark_helena \
  --scenario scheduler_sweep --datasets helena --ntrees 30 --seed 42

# Row budgets {0.25, 0.5, 0.75} and output budgets {10, 25, 50}, varied separately.
python -m experiments.benchmark_helena \
  --scenario budget_sweep --datasets helena --ntrees 30 --seed 42

# Stage A: keep the model full while a deterministic output-only policy runs
# in shadow mode. Saves per-node full/sketched Top-K and normalized gain regret.
python -m experiments.benchmark_helena \
  --scenario observability --datasets helena --ntrees 50 --seed 42

# Stage B: compare the current rule, K={2,5,10}, a robust volatility window,
# and the shadow split-risk scheduler at one fixed output-only budget.
python -m experiments.benchmark_helena \
  --scenario scheduler_diagnostics --datasets helena --ntrees 50 --seed 42

python -m experiments.analyze_scheduler_diagnostics \
  results/diagnostics/<scheduler-report>.json

# Non-mutating risk curves for output budgets {10, 25, 50, 75} and both
# history-weighted random sampling and history Top-K selection.
python -m experiments.benchmark_helena \
  --scenario risk_budget_sweep --datasets helena --ntrees 50 --seed 42

# Repeat the same scheduler cases without diagnostic histogram overhead when
# measuring end-to-end speed.
python -m experiments.benchmark_helena \
  --scenario scheduler_diagnostics --datasets helena --ntrees 50 --seed 42 \
  --disable-case-diagnostics

# Stage C: capture the full optimization trajectory. `ghz` supports every
# planned representation ablation; use `z --trajectory-window 8` for a smaller run.
python -m experiments.benchmark_helena \
  --scenario rmt_offline --datasets helena --ntrees 50 --seed 42 \
  --trajectory ghz

python -m experiments.analyze_optimization_history \
  results/diagnostics/artifacts/<run-id>/trajectories/<trajectory>.npz \
  --representations g z newton_step gh_separate --windows 4 8 16 --n-null 20

# Hardware-friendly pilot: keep the exact observed covariance, but evaluate
# every fourth window and cap only the empirical-null row sample.
python -m experiments.analyze_optimization_history \
  results/diagnostics/artifacts/<run-id>/trajectories/<trajectory>.npz \
  --representations g z newton_step gh_separate --windows 4 8 16 \
  --n-null 20 --stride 4 --null-max-rows 4096

# Test whether subspace rotation predicts the next tree's measured split risk.
python -m experiments.analyze_spectral_risk \
  results/spectral/spectral_windows.csv \
  results/diagnostics/artifacts/<run-id>/split_diagnostics/<trace>.json

# H-RMT-2: record exact leave-one-output split attributions, then compare them
# with the spectral output ranking.
python -m experiments.benchmark_helena \
  --scenario observability --datasets helena --ntrees 50 --seed 42 \
  --split-output-attribution
python -m experiments.analyze_output_attribution \
  results/spectral/spectral_windows.json \
  results/diagnostics/artifacts/<run-id>/split_diagnostics/<trace>.json \
  --budgets 10 25 50

# Stage D: compare uniform, current-gradient Top-K, history Top-K, and online
# spectral Top-K at budgets {25, 50, 75} without changing the fitted trees.
python -m experiments.benchmark_helena \
  --scenario spectral_shadow --datasets helena --ntrees 50 --seeds 42 52 62 \
  --spectral-window 16 --spectral-null-permutations 5 \
  --spectral-null-rows 2048 --spectral-null-interval 5
python -m experiments.analyze_spectral_shadow \
  results/diagnostics/<one-or-more-shadow-reports>.json --min-iteration 16

# Calibrate output-budget risk on a longer Helena trajectory, then compare
# fixed controls with the risk-controlled adaptive budget.
python -m experiments.benchmark_helena \
  --scenario adaptive_budget_diagnostics --datasets helena \
  --max-samples 12000 --ntrees 100 --seeds 42 52 62
python -m experiments.benchmark_helena \
  --scenario adaptive_budget --datasets helena \
  --max-samples 12000 --ntrees 100 --seeds 42 52 62
python -m experiments.analyze_adaptive_budget \
  results/diagnostics/<adaptive-budget-report>.json

# Transfer the adaptive policy to a compact MediaMill subset. Dataset-sized
# levels avoid a redundant 100/101 transition.
python -m experiments.benchmark_helena \
  --scenario adaptive_budget --datasets mediamill \
  --max-samples 10000 --ntrees 50 --seeds 42 52 62 \
  --adaptive-budgets 25 50 75 101

# Separate row and output effects: full, rows-only, outputs-only, and joint
# sketches for both current-gradient and history Top-K policies.
python -m experiments.benchmark_helena \
  --scenario row_output_factorial --datasets mediamill \
  --max-samples 10000 --ntrees 50 --seeds 42 52 62
python -m experiments.analyze_row_output_factorial \
  results/diagnostics/<row-output-factorial-report>.json

# Stabilize the adaptive controller, then confirm the selected p2/c10 rule
# with paired seeds on both compact reference datasets.
python -m experiments.benchmark_helena \
  --scenario adaptive_budget_stability --datasets helena \
  --max-samples 12000 --ntrees 100 --seeds 42 52 62
python -m experiments.benchmark_helena \
  --scenario adaptive_budget_stability --datasets helena mediamill \
  --case-ids full adaptive_current_p2_c10 \
  --max-samples 12000 --ntrees 100 --seeds 42 52 62 72 82
python -m experiments.analyze_paired_confirmation \
  results/diagnostics/<confirmation-report>.json \
  --candidate-cases adaptive_current_p2_c10 \
  --output-dir results/diagnostics/adaptive_budget_confirmation/analysis

# Isolate row sampling while retaining every output. This scenario disables
# split diagnostics so its time and sampled GPU-memory peak are comparable.
python -m experiments.benchmark_helena \
  --scenario row_budget_sweep --datasets helena \
  --max-samples 12000 --ntrees 100 --seeds 42 52 62

# Build per-iteration safe-budget labels and evaluate whether compact spectral
# state transfers between Helena and MediaMill better than training progress.
python -m experiments.benchmark_helena \
  --scenario rmt_budget --datasets helena mediamill \
  --max-samples 10000 --ntrees 50 --seeds 42 52 62 \
  --spectral-null-permutations 3 --spectral-null-rows 1024
python -m experiments.analyze_rmt_budget \
  results/diagnostics/<rmt-budget-report>.json \
  --output-dir results/diagnostics/rmt_budget_cross_dataset/analysis

# Compare direct spectral Top-K with sequential redundancy suppression at the
# same 75-output budget. All policies remain shadow-only in this diagnostic.
python -m experiments.benchmark_helena \
  --scenario spectral_diversity --datasets helena mediamill \
  --max-samples 10000 --ntrees 50 --seeds 42 52 62 \
  --spectral-null-permutations 3 --spectral-null-rows 1024

# CPU-only smoke control with known latent rank and a regime change.
python -m experiments.generate_synthetic_trajectory
python -m experiments.analyze_optimization_history \
  results/spectral/synthetic_trajectory.npz --windows 4 8 16 --n-null 20

# Compact SketchBoost-derived validation suite. Defaults retain complete Birds
# and Yeast and a stratified 10,000-row MediaMill subset.
python -m experiments.benchmark_helena \
  --scenario main --datasets birds yeast mediamill --ntrees 30
```

Use `--dry-plan` to inspect every resolved case without downloading data or
starting a GPU run. OpenML datasets are cached by immutable data ID. MediaMill
uses the stable train/test files from LIBSVM Tools because the legacy
`scikit-multilearn` catalog endpoint is no longer available.

Each run writes JSON, per-run CSV, and seed-aggregated CSV files under
`results/diagnostics/`. Reports include task quality, end-to-end fit time,
repeated inference time, scheduler activation, row/output sketch sizes,
index-uniqueness checks, split regret, false activations, and timings for
`get_weights` and `get_indexers`. Diagnostic runs include the cost of the extra
full histogram and must not be used as speed measurements; rerun them with
`--disable-case-diagnostics` for that comparison.
Combine independent scenario reports with:

```bash
python -m experiments.aggregate_benchmark_reports \
  --input-dir results/diagnostics/final
```

#### 1. Environment setup

- **Install the main project package** (from the repo root):
  - Ensure you have a compatible Python version.
  - Install the project in editable mode:
    ```bash
    pip install -e .
    ```
  - Optionally install tools like MLflow and Jupyter:
    ```bash
    pip install mlflow jupyter
    ```

- **Install additional dependencies used by `experiments`**  
  Minimal set (no versions specified, adjust as needed):
  ```bash
  pip install \
    numpy \
    pandas \
    scikit-learn \
    cupy \
    catboost \
    lightgbm \
    hydra-core \
    omegaconf \
    mlflow \
    iterstrat \
    ucimlrepo \
    scipy \
    huggingface-hub \
    pyarrow
  ```
  Notes:
  - `numpy`, `pandas`, `scikit-learn`: core scientific/ML stack.
  - `cupy`: GPU computations (pick the right wheel for your CUDA, e.g. `cupy-cuda11x`).
  - `catboost`, `lightgbm`: baseline and analogue models.
  - `hydra-core`, `omegaconf`: configuration system and CLI (Hydra + OmegaConf).
  - `mlflow`: experiment and artifact tracking.
  - `iterstrat`: `MultilabelStratifiedKFold` for multi-label CV.
  - `ucimlrepo`: UCI datasets loader (see `experiments/dataset_loading.py`).
  - `scipy`: ARFF support (`mediamill`).
  - `huggingface-hub`: dataset downloads for MBD.
  - `pyarrow` (or `fastparquet`): backend for `pandas.read_parquet`.

- **Check GPU availability** (if you plan GPU experiments):
  - CUDA drivers must be installed.
  - `cupy` must import successfully and see your GPU.

#### 2. Data preparation (`prepare_datasets.ipynb`)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open the notebook:
   - `experiments/prepare_datasets.ipynb`
3. Execute all cells in order:
   - The notebook downloads / prepares data for datasets defined in  
     `experiments/config/datasets/default.yaml` (e.g. `mnist`, `cifar10`, `mbd`, etc.).
   - Make sure the data paths match the project configuration (typically under `experiments/data` or as defined in `py_boost.paths.EXPERIMENTS_DATA_PATH`).
4. After the notebook finishes successfully, the datasets are ready for experiments.

##### Adding custom datasets

To plug in your own datasets:

1. **(Optional) Reuse `prepare_datasets.ipynb` for preprocessing**
   - Add cells that:
     - Load your raw data from the original source.
     - Perform any heavy preprocessing / feature engineering.
     - Save the processed data under a stable path (ideally under the directory used in `py_boost.paths.EXPERIMENTS_DATA_PATH` or `experiments/data`).

2. **Create a loader in `experiments/data/load_data.py`**
   - Add a function similar to the existing ones (`load_mediamill`, `load_mbd`, etc.), for example:
     ```python
     def load_my_dataset():
         # Load from files prepared in step 1
         features = ...
         target = ...
         metadata = DatasetMetadata(
             name="my_dataset",
             source="custom",
             shape=features.shape,
         )
         return {
             "features": features,
             "target": target,
             "metadata": metadata,
         }
     ```
   - If your dataset is naturally split into folds or comes with pre-defined train/test splits, you can also model it after `load_mnist` / `load_cifar10` and then register it in `SPECIAL_LOADERS` in the same file.

3. **Register the dataset in the Hydra datasets config**
   - Open `experiments/config/datasets/default.yaml` and add a new entry, for example:
     ```yaml
     my_dataset:
       id: my_dataset
       source: custom
       method:
         _target_: data.load_data.load_my_dataset
     ```
   - After this, any experiment config that includes `datasets: default` can see `my_dataset` and you can control inclusion/exclusion via `skip_datasets` and `skip_first` in the scenario YAMLs (e.g. `hyperbolic.yaml`, `baselines.yaml`, etc.).

#### 3. Experiments structure

The main components live under `experiments/`:

- **Core layer** (`experiments/core/`):
  - `experiment.py`:
    - `ExperimentContext`: per-dataset context (dataset name, task type, number of classes, CV, timeout, search keys).
    - `BaseExperiment`: base class with overridable hooks:
      - `build_search_space(context) -> Mapping[str, Sequence[Any]]`
      - `build_default_params(context) -> dict`
      - `make_model(model_factory, params, context)`
      - `fit_model(model, X_train, y_train, X_test, y_test, context, fold_idx)`
      - `methods_to_time(context)`
      - `before_fold(context, fold_idx, params, X_train, y_train, X_test, y_test)`
      - `after_fold(context, fold_idx, model, y_test, probas, predictions, fold_metrics)`
      - `after_dataset(context, all_results)`
      - `wants_results_dataframe()`
    - `ExperimentRunner`: iterates over datasets and hyperparameter combinations and delegates execution to `core.process_runner.run_experiment_in_process`.
  - `process_runner.py`: runs a single hyperparameter configuration in a separate process with MLflow logging and CV.
  - `gpu.py`: GPU helpers (`GPUTimer`, `initialize_gpu_settings`, `nuclear_cleanup`, `safe_predict`).
  - `metrics.py`: common metric config (`METRICS`, `PRED_THR`, `ROCAUC_SCORE`, post-processing, aggregation).
  - `cv.py`: cross-validation config (`RANDOM_STATE`, `FOLDS`).
  - `model_timing.py`: `time_patch_methods`, `log_timing_data` for low-level GPU timing.
  - `mlflow_utils.py`: helpers for starting runs and logging parameters.

- **Concrete experiments** (`experiments/runners/`):
  - `FundamentalsExperiment`: main multi-label GPU boosting experiment (sketch-based search space, default parameters, timing of `get_weights` / `get_indexers`).
  - `CorruptedLabelsExperiment`: adds `label_corruption` to the search space and corrupts training labels in `before_fold`.
  - `AnaloguesExperiment`: uses a simplified search space (`learning_rate`, `subsample`) and a `fit_model` override that supports models with `eval_set`.
  - `BaselinesExperiment`: baseline experiments that enable per-configuration results CSVs and aggregate them per dataset in `after_dataset`.
  - `SigmoidWeightedExperiment`: sigmoid-weighted history baselines that filter sketch methods and log dataset-level MLflow summaries with aggregated results and histories.

- **Models** (`experiments/models.py`):
  - `HyperbolicHistoryBoost` (via `HyperbolicWeightedHistorySampling`).
  - `PreparedLGBM` (multi-output LGBM wrapper).
  - `PreparedCatBoost` (CatBoost with default experiment-friendly params).

- **Configs** (`experiments/config/`):
  - Main scenario configs:
    - `hyperbolic.yaml`, `hyperbolic_corrupted.yaml`
    - `baselines.yaml`, `baselines_corrupted.yaml`
    - `sigmoid.yaml`, `sigmoid_corrupted.yaml`
    - `catboost.yaml`, `lgbm.yaml`, `trial.yaml`, `tmp_mnist_hyperbolic.yaml`
  - Datasets config:
    - `datasets/default.yaml`

- **Scripts** (`experiments/bin/`):
  - Convenience shell scripts to run specific scenarios:
    - `run_hyperbolic.sh`, `run_hyperbolic_corrupted.sh`
    - `run_baselines.sh`, `run_baselines_corrupted.sh`
    - `run_sigmoid.sh`, `run_sigmoid_corrupted.sh`
    - `run_analogues_catboost.sh`, `run_analogues_lgbm.sh`
    - `run_trial.sh`, `run_tmp_mnist_hyperbolic.sh`

#### 4. Unified Hydra entrypoint

All experiments are launched through a single entrypoint:

```bash
python -m experiments.exp --config-name <config_name>
```

Where `<config_name>` is one of the files in `experiments/config/` without the `.yaml` extension.

Examples:

- Hyperbolic experiment:
  ```bash
  python -m experiments.exp --config-name hyperbolic
  ```
- Hyperbolic with corrupted labels:
  ```bash
  python -m experiments.exp --config-name hyperbolic_corrupted
  ```
- Baselines:
  ```bash
  python -m experiments.exp --config-name baselines
  python -m experiments.exp --config-name baselines_corrupted
  ```
- Sigmoid / hyperbolic weighted experiment:
  ```bash
  python -m experiments.exp --config-name sigmoid
  python -m experiments.exp --config-name sigmoid_corrupted
  ```
- Analogues with CatBoost / LGBM:
  ```bash
  python -m experiments.exp --config-name catboost
  python -m experiments.exp --config-name lgbm
  ```
- Quick trial:
  ```bash
  python -m experiments.exp --config-name trial
  ```

Hydra config defines:

- `experiment._target_`: a class in `experiments.runners.*` (type of experiment).
- `model._target_`: model factory (e.g. `experiments.models.HyperbolicHistoryBoost`).
- `datasets`: set of datasets (see `config/datasets/default.yaml`).

#### 5. Running via shell scripts

Instead of calling `python -m experiments.exp` directly, you can use scripts under `experiments/bin`:

Examples:

- `bin/run_hyperbolic.sh`
- `bin/run_hyperbolic_corrupted.sh`
- `bin/run_baselines.sh`
- `bin/run_baselines_corrupted.sh`
- `bin/run_sigmoid.sh`
- `bin/run_sigmoid_corrupted.sh`
- `bin/run_analogues_catboost.sh`
- `bin/run_analogues_lgbm.sh`
- `bin/run_trial.sh`
- `bin/run_tmp_mnist_hyperbolic.sh`

Run from the repo root:

```bash
cd /path/to/Py-Boost
bash experiments/bin/run_hyperbolic.sh
```

#### 6. Minimal pre-run checklist

1. Environment and dependencies are installed:
   - `pip install -e .`
   - Extra packages for `experiments` are installed (`numpy`, `pandas`, `scikit-learn`, `cupy`, `catboost`, `lightgbm`, `hydra-core`, `omegaconf`, `mlflow`, `iterstrat`, `ucimlrepo`, `scipy`, `huggingface-hub`, `pyarrow`).
2. `prepare_datasets.ipynb` has been executed successfully:
   - All required datasets are downloaded and stored at expected locations.
3. `experiments/config/datasets/default.yaml` is configured for the datasets you want to use (if you changed anything).
4. The desired scenario config in `experiments/config/*.yaml` is selected and, if needed, `skip_datasets` and `skip_first` are adjusted.
5. Run:
   - Either via `python -m experiments.exp --config-name <name>`,
   - Or via `bash experiments/bin/run_<something>.sh`.

