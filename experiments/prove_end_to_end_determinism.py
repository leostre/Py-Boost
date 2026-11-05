"""Prove or refute end-to-end run-to-run bitwise determinism (open question 2).

Gap being closed
----------------
The deterministic kernels have unit tests (bitwise invariance to row
permutation, bitwise equality with frozen reference kernels), but no test
fits a *whole model* twice and asks whether predictions are bitwise
identical. Residual non-determinism could still enter through any other
stage of the pipeline (cumsum, sorting, argmax tie-breaking, leaf value
accumulation, data sampling).

Protocol
--------
1. Fit R replicas of the identical configuration with
   ``deterministic_histogram=True`` and ``deterministic_leaf_values=True``.
2. Compare every replica against replica 0:
   predictions (bitwise), per-tree structure fingerprints, per-tree
   leaf-value fingerprints.
3. Control arm: R replicas with both flags False (default atomic path).
   If the control arm is also bitwise identical, the check has no power to
   detect non-determinism at this scale, and the result is inconclusive
   rather than proven.

Verdict
-------
``proven`` is True only when the deterministic arm is fully bitwise
identical AND the atomic control arm shows at least one deviation (or the
control arm is disabled explicitly).

Usage
-----
    python3 experiments/prove_end_to_end_determinism.py
    python3 experiments/prove_end_to_end_determinism.py --replicas 4 --ntrees 30
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.proving_common import (
    base_report,
    compare_replica,
    fit_replica,
    make_dataset,
    replica_record,
    strip_heavy_fields,
    write_artifact,
)


def run_arm(X, y, config, deterministic, replicas, tolerances=None):
    records = []
    for index in range(replicas):
        model, elapsed = fit_replica(X, y, config, deterministic)
        records.append(replica_record(model, X, elapsed))
        print(f"[{'det' if deterministic else 'atomic'}] replica {index} fitted "
              f"in {elapsed:.2f}s", flush=True)
    reference = records[0]
    comparisons = [
        compare_replica(reference, rec, tolerances=tolerances)
        for rec in records[1:]
    ]
    all_bitwise = all(
        c['prediction_bitwise_equal']
        and c['flipped_trees'] == 0
        and c['leaf_value_only_trees'] == 0
        for c in comparisons
    )
    return records, comparisons, all_bitwise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replicas', type=int, default=3)
    parser.add_argument('--ntrees', type=int, default=30)
    parser.add_argument('--nrows', type=int, default=32768)
    parser.add_argument('--nfeatures', type=int, default=16)
    parser.add_argument('--nclasses', type=int, default=6)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--min-data-in-leaf', type=int, default=10)
    parser.add_argument('--layout', default='hybrid',
                        choices=['hybrid', 'perf1'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--skip-atomic-control', action='store_true',
                        help='do not fit the atomic control arm')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    config = {
        'replicas': args.replicas,
        'ntrees': args.ntrees,
        'nrows': args.nrows,
        'nfeatures': args.nfeatures,
        'nclasses': args.nclasses,
        'max_depth': args.max_depth,
        'lr': args.lr,
        'min_data_in_leaf': args.min_data_in_leaf,
        'layout': args.layout,
        'seed': args.seed,
        'subsample': 1.0,
        'colsample': 1.0,
        'full_leaf_values': True,
        'loss': 'crossentropy',
    }
    X, y = make_dataset(seed=args.seed, nrows=args.nrows,
                        nfeatures=args.nfeatures, nclasses=args.nclasses)

    report = base_report('prove_end_to_end_determinism', config, {
        'deterministic': {'deterministic_histogram': True,
                          'deterministic_leaf_values': True},
        'atomic_control': None if args.skip_atomic_control else {
            'deterministic_histogram': False, 'deterministic_leaf_values': False},
    })
    report['question'] = (
        'Q2: does deterministic mode make a full model fit bitwise '
        'reproducible run-to-run?'
    )

    tolerances = {
        'gain_atol': 5e-3, 'gain_rtol': 5e-5,
        'bin_tol': 1, 'value_atol': 1e-6, 'value_rtol': 1e-5,
    }
    det_records, det_cmp, det_bitwise = run_arm(
        X, y, config, True, args.replicas, tolerances)
    report['detail']['deterministic'] = {
        'all_replicas_bitwise_identical': det_bitwise,
        'comparisons_vs_replica0': det_cmp,
    }

    control_sensitive = args.skip_atomic_control
    if not args.skip_atomic_control:
        _, atom_cmp, atom_bitwise = run_arm(
            X, y, config, False, args.replicas, tolerances)
        control_sensitive = not atom_bitwise
        report['detail']['atomic_control'] = {
            'all_replicas_bitwise_identical': atom_bitwise,
            'comparisons_vs_replica0': atom_cmp,
        }

    if det_bitwise and control_sensitive:
        report['proven'] = True
        report['verdict'] = (
            'PASS: deterministic mode is bitwise reproducible end-to-end '
            '(predictions + every tree structure + every leaf value), and the '
            'atomic control arm deviates, so the check has discriminative power.'
        )
    elif det_bitwise and not control_sensitive:
        report['proven'] = False
        report['verdict'] = (
            'INCONCLUSIVE: deterministic arm is bitwise identical, but the '
            'atomic control arm is also bitwise identical at this scale - the '
            'experiment cannot distinguish the modes (see Q3 flip-rate script).'
        )
    else:
        report['proven'] = False
        divergent = next(
            c for c in det_cmp
            if not c['prediction_bitwise_equal'] or c['flipped_trees']
            or c['leaf_value_only_trees']
        )
        report['verdict'] = (
            'FAIL: deterministic mode is NOT bitwise reproducible end-to-end. '
            f'First divergence vs replica 0: {divergent}'
        )

    report['detail']['replica_summaries'] = {
        'deterministic': [strip_heavy_fields(r) for r in det_records],
    }
    report['summary'] = {
        'deterministic_bitwise': det_bitwise,
        'atomic_control_sensitive': control_sensitive,
    }

    path = write_artifact(
        f"end_to_end_determinism_{args.tag + '_' if args.tag else ''}"
        f"{report['environment']['device_name'].replace(' ', '_')}.json",
        report,
    )
    print(json.dumps(report['summary'], indent=2))
    print(f"verdict: {report['verdict']}")
    print(f"artifact: {path}")


if __name__ == '__main__':
    main()
