"""Blocked calibration and bounded comparison for the accepted review protocol."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


def blocked_case_order(cases, block_index):
    """Rotate methods within blocks so the full reference is not always first."""
    cases = tuple(cases)
    if not cases:
        raise ValueError('at least one case is required')
    offset = block_index % len(cases)
    ordered = cases[offset:] + cases[:offset]
    return ordered


def select_calibration(records, tolerance):
    """Freeze fastest eligible simple candidates; never use comparison splits."""
    selected = {}
    for dataset in sorted({row['dataset'] for row in records}):
        rows = [row for row in records if row['dataset'] == dataset]
        full = next(row for row in rows if row['case_id'] == 'full')
        def choose(prefix):
            candidates = [row for row in rows if row['case_id'].startswith(prefix)]
            eligible = [row for row in candidates
                        if row['f1_weighted'] >= full['f1_weighted'] - tolerance]
            # If none meets the guardrail, retain the best-quality candidate and say so.
            pool = eligible or candidates
            best = (min(pool, key=lambda row: (row['train_seconds'], row['case_id']))
                    if eligible else max(pool, key=lambda row: row['f1_weighted']))
            return best, bool(eligible)
        fixed, fixed_eligible = choose('fixed_')
        energy, energy_eligible = choose('energy_')
        selected[dataset] = {
            'fixed_budget': int(fixed['sketch_outputs']),
            'energy_fraction': float(energy['output_energy_fraction']),
            'fixed_calibration_eligible': fixed_eligible,
            'energy_calibration_eligible': energy_eligible,
            'calibration_split_seed': int(full['split_seed']),
            'calibration_model_seed': int(full['model_seed']),
        }
    return selected


def comparison_gate(records, quality_tolerance, minimum_time_saving):
    """Compare risk control against both frozen simple competitors by paired block."""
    import numpy as np
    output = []
    for dataset in sorted({row['dataset'] for row in records}):
        rows = [row for row in records if row['dataset'] == dataset]
        comparisons = []
        for competitor in ('fixed', 'energy'):
            deltas, ratios = [], []
            for seed in sorted({int(row['seed']) for row in rows}):
                block = {row['case_id']: row for row in rows if row['seed'] == seed}
                risk, simple = block['risk'], block[competitor]
                deltas.append(risk['f1_weighted'] - simple['f1_weighted'])
                ratios.append(risk['train_seconds'] / simple['train_seconds'])
            comparisons.append({
                'competitor': competitor, 'paired_blocks': len(deltas),
                'mean_f1_delta': float(np.mean(deltas)),
                'mean_time_ratio': float(np.mean(ratios)),
                'passes': bool(np.mean(deltas) >= -quality_tolerance
                               and np.mean(ratios) <= 1 - minimum_time_saving),
                'paired_f1_deltas': deltas, 'paired_time_ratios': ratios,
            })
        output.append({'dataset': dataset, 'comparisons': comparisons,
                       'passes': all(item['passes'] for item in comparisons)})
    return {'datasets': output, 'continue_risk_controller': any(item['passes'] for item in output),
            'quality_tolerance': quality_tolerance, 'minimum_time_saving': minimum_time_saving,
            'interpretation': 'pilot gate only; three whole trajectories, not independent windows'}


def run_comparison(config):
    import cupy as cp
    import numpy as np
    from experiments import benchmark_helena as benchmark
    from experiments.benchmark_datasets import load_benchmark_dataset, split_benchmark_dataset
    from experiments.benchmark_plan import ExperimentCase, PlanDefaults, PlannedRun, Scenario, SchedulerMode, Variant, full_case
    from experiments.reproducibility import paired_seed_map

    if config['phase'] == 'budget_comparison':
        fits = len(config['datasets']) * len(config['model_seeds']) * 4
        if fits > 24 or len(config['split_seeds']) != 3:
            raise ValueError('the first comparison is capped at 24 fits and three split blocks')
        if set(config['split_seeds']) & {42, 52, 62, 72, 82, 1140}:
            raise ValueError('the comparison requires new, non-calibration split seeds')

    previous_argv = sys.argv
    sys.argv = ['benchmark_helena']
    try:
        args = benchmark.parse_args()
    finally:
        sys.argv = previous_argv
    for key in ('dataset_seed', 'max_samples', 'ntrees', 'max_depth'):
        setattr(args, key, config[key])
    args.cache_dir = Path(config['cache_dir'])
    args.output_dir = Path('results/revision')
    args.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    args.split_seed_by_model = paired_seed_map(config['model_seeds'], config['split_seeds'])
    args.record_selection_indices = True
    args.profile_methods = False
    args.predict_repeats = 3
    args.gain_risk_threshold = 0.10
    args.split_top_k = 0  # The controller needs winners, not research-only candidate sorting.
    records = []
    shapes = {}
    for dataset_index, name in enumerate(config['datasets']):
        dataset = load_benchmark_dataset(name, args.max_samples, args.dataset_seed, args.cache_dir)
        shapes[name] = {'samples': dataset.features.shape[0], 'features': dataset.features.shape[1],
                        'outputs': dataset.n_outputs}
        args.output_min_fraction = min(50, dataset.n_outputs) / dataset.n_outputs
        defaults = PlanDefaults(row_subsample=1., sketch_outputs=75, min_warmup_rounds=5,
                                adaptive_budgets=(50, 75, dataset.n_outputs),
                                scheduler_control_interval=10, adaptive_min_dwell_rounds=10,
                                adaptive_cooldown_after_increase_rounds=20)
        cases = {'full': (full_case(), 0.90)}
        if config['phase'] == 'causal_forecast':
            cases = {'matched_trajectory': (ExperimentCase('matched_trajectory', Variant.HASBOOST_CURRENT_TOPK,
                          scheduler_mode=SchedulerMode.NEVER, row_subsample=1.,
                          sketch_outputs=25, collect_split_diagnostics=True), .90)}
        elif config['phase'] == 'fixed_memory':
            budget = config['selected_simple'][name]['fixed_budget']
            cases = {case_id: (ExperimentCase(case_id, variant,
                                scheduler_mode=SchedulerMode.FIXED_WARMUP,
                                row_subsample=1., sketch_outputs=budget, smoothing_alpha=.9), .90)
                     for case_id, variant in (
                         ('current', Variant.HASBOOST_CURRENT_TOPK),
                         ('ema', Variant.HASBOOST_EMA_ENERGY),
                         ('legacy', Variant.HASBOOST_SIGMOID_TOPK))}
        elif config['phase'] == 'calibration':
            cases.update({f'fixed_{budget}': (ExperimentCase(f'fixed_{budget}', Variant.HASBOOST_CURRENT_TOPK,
                                scheduler_mode=SchedulerMode.FIXED_WARMUP, row_subsample=1.,
                                sketch_outputs=budget), 0.90) for budget in config['fixed_candidates']})
            cases.update({f'energy_{fraction:.2f}': (ExperimentCase(f'energy_{fraction:.2f}', Variant.HASBOOST_ENERGY_BUDGET,
                                scheduler_mode=SchedulerMode.FIXED_WARMUP, row_subsample=1.), fraction)
                          for fraction in config['energy_candidates']})
        else:
            selected = config['selected_simple'][name]
            if selected['calibration_split_seed'] in config['split_seeds']:
                raise ValueError('calibration and comparison splits must be disjoint')
            cases.update({
                'fixed': (ExperimentCase('fixed', Variant.HASBOOST_CURRENT_TOPK,
                            scheduler_mode=SchedulerMode.FIXED_WARMUP, row_subsample=1.,
                            sketch_outputs=selected['fixed_budget']), 0.90),
                'energy': (ExperimentCase('energy', Variant.HASBOOST_ENERGY_BUDGET,
                            scheduler_mode=SchedulerMode.FIXED_WARMUP, row_subsample=1.), selected['energy_fraction']),
                'risk': (ExperimentCase('risk', Variant.HASBOOST_CURRENT_TOPK,
                            scheduler_mode=SchedulerMode.ADAPTIVE_BUDGET, row_subsample=1.,
                            sketch_outputs=75, adaptive_patience_down=2), 0.90),
            })
        resolved = {key: (case.resolve(defaults, dataset.n_outputs), fraction)
                    for key, (case, fraction) in cases.items()}
        for block_index, seed in enumerate(config['model_seeds']):
            split_seed = args.split_seed_by_model[seed]
            split = split_benchmark_dataset(dataset, split_seed, args.test_size)
            order = blocked_case_order(tuple(resolved), block_index + dataset_index)
            print(json.dumps({'BLOCK': {'dataset': name, 'model_seed': seed,
                                       'split_seed': split_seed, 'case_order': order}}), flush=True)
            for position, case_id in enumerate(order):
                case, fraction = resolved[case_id]
                case_args = argparse.Namespace(**vars(args))
                case_args.output_energy_fraction = fraction
                if config['phase'] == 'causal_forecast':
                    case_args.trajectory = 'gh'
                    case_args.trajectory_subset_rows = 1024
                    case_args.research_shadow_budgets = (50, 75)
                # Equal warm-up length, including the actual controller path.
                warm_args = argparse.Namespace(**vars(case_args))
                warm_args.ntrees = 26
                warm_args.record_selection_indices = False
                warm_args.trajectory = 'none'
                print(f'WARMUP {name}/{case_id}/model={seed}/split={split_seed}', flush=True)
                model = benchmark.build_model(case, warm_args, dataset.spec.task, seed)
                model.fit(split[0], split[2])
                benchmark._synchronize()
                del model
                benchmark._cleanup_gpu()
                run = PlannedRun(Scenario.ADAPTIVE_BUDGET, case, seed)
                record = benchmark.evaluate_run(run, case_args, dataset, *split)
                record.update(block_position=position, output_energy_fraction=fraction,
                              output_min_fraction=case_args.output_min_fraction,
                              warmup_trees=26, phase=config['phase'])
                records.append(record)
                print(json.dumps({'RESULT': record}), flush=True)
    if config['phase'] == 'calibration':
        decision = {'selected_simple': select_calibration(records, config['quality_tolerance'])}
    elif config['phase'] == 'budget_comparison':
        decision = comparison_gate(records, config['quality_tolerance'], config['minimum_time_saving'])
    elif config['phase'] == 'causal_forecast':
        from experiments.revision_forecast import analyze_forecasts
        decision = analyze_forecasts(records, args.output_dir / f'forecast_{args.run_id}.json')
    else:
        from experiments.revision_memory_analysis import memory_decision
        decision = memory_decision(records)
    report = {'metadata': {'phase': config['phase'], 'config': config, 'shapes': shapes,
                           'python': platform.python_version(), 'cupy': cp.__version__,
                           'numpy': np.__version__, 'gpu': benchmark._gpu_name(),
                           'measured_trainings': len(records), 'warmup_trainings': len(records)},
              'results': records, 'decision': decision}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'{config["phase"]}_{args.run_id}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'FINAL_SUMMARY': {'report': str(path.resolve()),
                                      'measured_trainings': len(records), 'decision': decision}}), flush=True)
