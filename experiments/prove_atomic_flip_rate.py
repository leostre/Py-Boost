"""Measure the atomic-mode structural flip rate (open question 3).

Gap being closed
----------------
The deterministic histogram/leaf-value machinery was built to eliminate
float ``atomicAdd`` non-associativity, but the null hypothesis was never
tested: how often does the default atomic path actually change a tree
between two identical runs? Without this number, the necessity of the
deterministic lever is unknown - it may be protecting against noise that
never flips a split decision at realistic scales.

Protocol
--------
1. Fit R replicas in default atomic mode (both deterministic flags False),
   identical seed, identical data, no row/column subsampling. The only
   difference between replicas is GPU scheduling order inside atomic
   kernels.
2. Compare each replica against replica 0 at three levels:
   - split decisions: nodes where (feature, bin, nan direction, child
     topology) changed, with a Wilson confidence interval on the per-split
     flip rate;
   - leaf values only: trees whose structure matches but values differ;
   - predictions: bitwise equality and max absolute deviation.
3. Control arm: R replicas with deterministic mode on. A discrepancy in
   this arm invalidates attribution of the differences to atomic reductions.

Tolerant classification
-----------------------
Strict fingerprint comparison counts any integer difference in
(feature, bin, NaN direction, child topology) as a flip. The element-wise
comparison additionally classifies every divergent split node:
  near_tie         - same feature and NaN direction, children equal, bin
                     differs by at most --bin-tol (adjacent thresholds)
  gain_equivalent  - gains agree within --gain-atol + --gain-rtol*|g| (the
                     repository's own refinement-audit tolerance), or one
                     side is a leaf and the other's split gain is within
                     --gain-atol of zero (a borderline split)
  hard_flip        - everything else
The verdict is decided by the HARD-flip rate, not the strict rate; strict
and near-tie/gain-equivalent counts are reported alongside. Per-node
element-wise records (exact values on both sides) are stored in the
artifact under each tree's ``elementwise`` entry.

Interpreting the verdict
------------------------
- ``ATOMIC FLIPS OBSERVED``: report the Wilson CI - this is the effect size
  the deterministic lever protects against.
- ``NO HARD FLIPS OBSERVED``: zero hard flips in this sample at the specified
  tolerances. Trees in the same fit and pairs sharing replica 0 are dependent,
  so the Wilson intervals are descriptive.

Usage
-----
    python3 experiments/prove_atomic_flip_rate.py
    python3 experiments/prove_atomic_flip_rate.py --replicas 8 --ntrees 30
    python3 experiments/prove_atomic_flip_rate.py --replay saved.json --gain-atol 1e-4
"""

import argparse
from hashlib import sha256
import json
from math import isfinite
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.proving_common import (
    DEFAULT_TOLERANCES,
    SCHEMA_VERSION,
    base_report,
    compare_replica,
    fit_replica,
    make_dataset,
    reclassify_splits,
    replica_record,
    strip_heavy_fields,
    wilson_ci,
    write_artifact,
)


def summarize_arm(comparisons):
    pairs = len(comparisons)
    flipped_trees = sum(c['flipped_trees'] for c in comparisons)
    total_trees = sum(c['total_trees'] for c in comparisons)
    leaf_value_only = sum(c['leaf_value_only_trees'] for c in comparisons)
    flipped_split_nodes = sum(c['flipped_split_nodes'] for c in comparisons)
    total_split_nodes = sum(c['total_split_nodes'] for c in comparisons)
    bitwise = sum(c['prediction_bitwise_equal'] for c in comparisons)
    diffs = [c['prediction_max_abs_diff'] for c in comparisons]
    diverged_replicas = sum(
        1 for c in comparisons
        if c['flipped_trees'] or c['leaf_value_only_trees']
        or not c['prediction_bitwise_equal']
    )
    tree_rate = wilson_ci(flipped_trees, total_trees)
    split_rate = wilson_ci(flipped_split_nodes, total_split_nodes)
    tolerant = {
        'hard_flip_trees': sum(c['tolerant']['hard_flip_trees'] for c in comparisons),
        'near_tie_trees': sum(c['tolerant']['near_tie_trees'] for c in comparisons),
        'gain_equivalent_trees': sum(
            c['tolerant']['gain_equivalent_trees'] for c in comparisons),
        'value_drift_trees': sum(
            c['tolerant']['value_drift_trees'] for c in comparisons),
        'values_allclose_trees': sum(
            c['tolerant']['values_allclose_trees'] for c in comparisons),
        'hard_flip_split_nodes': sum(
            c['tolerant']['hard_flip_split_nodes'] for c in comparisons),
        'total_split_nodes': sum(
            c['tolerant']['total_split_nodes'] for c in comparisons),
        'values_max_abs_diff_max': max(
            c['tolerant']['values_max_abs_diff_max'] for c in comparisons),
        'root_divergence_depths': [
            depth
            for c in comparisons
            for depth in c['tolerant']['root_divergence_depths']
        ],
    }
    hard_rate = wilson_ci(tolerant['hard_flip_trees'], total_trees)
    return {
        'replica_pairs': pairs,
        'replicas_with_any_divergence': diverged_replicas,
        'flipped_trees': flipped_trees,
        'total_trees': total_trees,
        'tolerant': tolerant,
        'hard_flip_tree_rate_wilson_ci': {
            'point': hard_rate[0], 'lower': hard_rate[1], 'upper': hard_rate[2]},
        'leaf_value_only_trees': leaf_value_only,
        'tree_flip_rate_wilson_ci': {
            'point': tree_rate[0], 'lower': tree_rate[1], 'upper': tree_rate[2]},
        'flipped_split_nodes': flipped_split_nodes,
        'total_split_nodes': total_split_nodes,
        'split_flip_rate_wilson_ci': {
            'point': split_rate[0], 'lower': split_rate[1], 'upper': split_rate[2]},
        'prediction_bitwise_equal_pairs': bitwise,
        'prediction_max_abs_diff_median': float(np.median(diffs)) if diffs else 0.0,
        'prediction_max_abs_diff_max': float(max(diffs)) if diffs else 0.0,
        'first_divergent_trees': [c['first_divergent_tree'] for c in comparisons],
    }


def run_arm(X, y, config, deterministic, replicas, tolerances):
    records = []
    for index in range(replicas):
        model, elapsed = fit_replica(X, y, config, deterministic)
        records.append(replica_record(model, X, elapsed))
        print(f"[{'det' if deterministic else 'atomic'}] replica {index} fitted "
              f"in {elapsed:.2f}s", flush=True)
    comparisons = [
        compare_replica(records[0], rec, tolerances=tolerances)
        for rec in records[1:]
    ]
    return records, comparisons


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replicas', type=int, default=8)
    parser.add_argument('--ntrees', type=int, default=30)
    parser.add_argument('--nrows', type=int, default=32768)
    parser.add_argument('--nfeatures', type=int, default=16)
    parser.add_argument('--nclasses', type=int, default=6)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--min-data-in-leaf', type=int, default=10)
    parser.add_argument('--layout', default='hybrid', choices=['hybrid', 'perf1'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--gain-atol', type=float)
    parser.add_argument('--gain-rtol', type=float)
    parser.add_argument('--bin-tol', type=int)
    parser.add_argument('--value-atol', type=float)
    parser.add_argument('--value-rtol', type=float)
    parser.add_argument('--tag', default='')
    parser.add_argument('--replay', type=Path,
                        help='reclassify all saved Q3 split comparisons with no GPU fit')
    return parser, parser.parse_args()


def collect(args, tolerances):
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
        'tolerances': tolerances,
        'subsample': 1.0,
        'colsample': 1.0,
        'full_leaf_values': True,
        'loss': 'crossentropy',
    }
    X, y = make_dataset(seed=args.seed, nrows=args.nrows,
                        nfeatures=args.nfeatures, nclasses=args.nclasses)

    report = base_report('prove_atomic_flip_rate', config, {
        'atomic': {'deterministic_histogram': False, 'deterministic_leaf_values': False},
        'deterministic_control': {'deterministic_histogram': True,
                                  'deterministic_leaf_values': True},
    }, X, y)
    report['question'] = (
        'Q3: how often does default atomic-mode GPU noise flip a split '
        'decision between identical seeded runs?'
    )

    atom_records, atom_cmp = run_arm(X, y, config, False, args.replicas, tolerances)
    det_records, det_cmp = run_arm(X, y, config, True, args.replicas, tolerances)
    report['detail']['replica_summaries'] = {
        'atomic': [strip_heavy_fields(r) for r in atom_records],
        'deterministic_control': [strip_heavy_fields(r) for r in det_records],
    }
    return report, atom_cmp, det_cmp


def replay_comparisons(report, tolerances):
    def replay_arm(name):
        pairs = report['detail'][name]['comparisons_vs_replica0']
        if len(pairs) != report['config']['replicas'] - 1 or not pairs:
            raise ValueError(f'incomplete replica pairs for {name}')
        for pair in pairs:
            counts = [len(tree['elementwise']) for tree in pair['trees']]
            if (len(counts) != pair['total_trees']
                    or sum(bool(count) for count in counts) != pair['flipped_trees']
                    or sum(counts) != pair['flipped_split_nodes']):
                raise ValueError(f'incomplete split details for {name}')
        return [reclassify_splits(pair, tolerances) for pair in pairs]

    return replay_arm('atomic'), replay_arm('deterministic_control')


def main():
    parser, args = parse_args()
    overrides = {
        key: getattr(args, key) for key in DEFAULT_TOLERANCES if getattr(args, key) is not None
    }
    if any(not isfinite(value) or value < 0 for value in overrides.values()):
        parser.error('comparison tolerances must be finite and non-negative')

    if args.replay is None:
        if args.replicas < 2 or args.ntrees < 1:
            parser.error('need at least two replicas and one tree')
        tolerances = {**DEFAULT_TOLERANCES, **overrides}
        report, atom_cmp, det_cmp = collect(args, tolerances)
    else:
        if 'value_atol' in overrides or 'value_rtol' in overrides:
            parser.error('replay supports split tolerances only; leaf-value results are preserved')
        try:
            contents = args.replay.read_bytes()
            report = json.loads(contents)
            if (not isinstance(report, dict) or report.get('schema_version') != SCHEMA_VERSION
                    or report.get('experiment') != 'prove_atomic_flip_rate'):
                raise ValueError(f'expected a Q3 artifact with schema_version={SCHEMA_VERSION}')
            tolerances = {**report['config']['tolerances'], **overrides}
            atom_cmp, det_cmp = replay_comparisons(report, tolerances)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f'cannot replay {args.replay}: {error}')
        report['config']['tolerances'] = tolerances
        report['replay_source_sha256'] = sha256(contents).hexdigest()

    atom_summary = summarize_arm(atom_cmp)
    det_summary = summarize_arm(det_cmp)
    report['detail']['atomic'] = {
        **atom_summary,
        'comparisons_vs_replica0': atom_cmp,
    }
    report['detail']['deterministic_control'] = {
        **det_summary,
        'comparisons_vs_replica0': det_cmp,
    }
    control_clean = (
        det_summary['flipped_trees'] == 0
        and det_summary['leaf_value_only_trees'] == 0
        and det_summary['replicas_with_any_divergence'] == 0
    )
    hard = atom_summary['tolerant']['hard_flip_trees']
    flips_observed = hard > 0
    strict_rate = atom_summary['tree_flip_rate_wilson_ci']
    hard_rate = atom_summary['hard_flip_tree_rate_wilson_ci']
    near_tie = atom_summary['tolerant']['near_tie_trees']
    gain_equiv = atom_summary['tolerant']['gain_equivalent_trees']

    if not control_clean:
        report['proven'] = False
        report['verdict'] = (
            'INVALID: the deterministic control arm diverged between identical '
            'seeded runs. The comparison cannot isolate the effect of atomic reductions.'
        )
    elif flips_observed:
        report['proven'] = True
        report['verdict'] = (
            'ATOMIC FLIPS OBSERVED: default atomic mode changes tree structure '
            f"between identical runs. STRICT tree flip rate {strict_rate['point']:.3e}, "
            f"Wilson 95% CI [{strict_rate['lower']:.3e}, {strict_rate['upper']:.3e}] "
            f"({atom_summary['flipped_trees']} of {atom_summary['total_trees']} "
            'trees). After element-wise tolerant classification '
            f"(bin_tol={tolerances['bin_tol']}, gain_atol={tolerances['gain_atol']}, "
            f"gain_rtol={tolerances['gain_rtol']}): near-tie trees {near_tie}, "
            f'gain-equivalent trees {gain_equiv}, HARD-FLIP trees {hard}, '
            f"hard-flip tree rate {hard_rate['point']:.3e}, Wilson 95% CI "
            f"[{hard_rate['lower']:.3e}, {hard_rate['upper']:.3e}]. "
            'Near-tie/gain-equivalent divergences are decision noise within '
            'audit tolerance; hard flips are the effect size the deterministic '
            'lever protects against.'
        )
    else:
        upper = atom_summary['hard_flip_tree_rate_wilson_ci']['upper']
        report['proven'] = True
        report['verdict'] = (
            'NO HARD FLIPS OBSERVED: no HARD structural '
            f'flips observed across {atom_summary["replica_pairs"]} replica '
            f'pairs and {atom_summary["tolerant"]["total_split_nodes"]} '
            'split-node comparisons at the configured tolerances. The Wilson '
            f'descriptive 95% upper limit on the hard-flip tree rate is {upper:.3e}.'
        )

    report['summary'] = {
        'atomic_tree_flip_rate_strict': strict_rate,
        'atomic_tree_flip_rate_hard': hard_rate,
        'atomic_near_tie_trees': near_tie,
        'atomic_gain_equivalent_trees': gain_equiv,
        'atomic_replicas_diverged': atom_summary['replicas_with_any_divergence'],
        'deterministic_control_clean': control_clean,
    }

    tag = args.tag or ('replay' if args.replay else '')
    path = write_artifact(
        f"atomic_flip_rate_{tag + '_' if tag else ''}"
        f"{report['environment']['device_name'].replace(' ', '_')}.json",
        report,
    )
    print(json.dumps(report['summary'], indent=2))
    print(f"verdict: {report['verdict']}")
    print(f"artifact: {path}")


if __name__ == '__main__':
    main()
