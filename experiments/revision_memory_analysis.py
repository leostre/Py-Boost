"""Whole-trajectory paired summaries for the controlled memory ablation."""

import numpy as np


def memory_decision(records):
    output = []
    for dataset in sorted({row['dataset'] for row in records}):
        rows = [row for row in records if row['dataset'] == dataset]
        for method in ('ema', 'legacy'):
            deltas, loss_deltas, ratios = [], [], []
            for seed in sorted({row['seed'] for row in rows}):
                block = {row['case_id']: row for row in rows if row['seed'] == seed}
                candidate, reference = block[method], block['current']
                deltas.append(candidate['f1_weighted'] - reference['f1_weighted'])
                loss_deltas.append(candidate['log_loss'] - reference['log_loss'])
                ratios.append(candidate['train_seconds'] / reference['train_seconds'])
            output.append({'dataset': dataset, 'method': method, 'paired_blocks': len(deltas),
                           'mean_f1_delta': float(np.mean(deltas)),
                           'mean_log_loss_delta': float(np.mean(loss_deltas)),
                           'mean_time_ratio': float(np.mean(ratios)),
                           'paired_f1_deltas': deltas, 'paired_log_loss_deltas': loss_deltas,
                           'paired_time_ratios': ratios,
                           'quality_consistent': all(value > 0 for value in deltas),
                           'log_loss_consistent': all(value < 0 for value in loss_deltas)})
    return {'comparisons': output,
            'interpretation': 'descriptive pilot at fixed budget; no dynamic memory-budget promotion'}
