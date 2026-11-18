"""Shared fit, snapshot and comparison code for the Q2 and Q3 checks."""

from collections import Counter
from hashlib import sha256
from itertools import count
import json
from math import sqrt
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from experiments.reproducibility import array_fingerprint, source_fingerprint


DEFAULT_BIN_TOL = 1
DEFAULT_GAIN_ATOL = 5e-3
DEFAULT_GAIN_RTOL = 5e-5
DEFAULT_VALUE_ATOL = 1e-6
DEFAULT_VALUE_RTOL = 1e-5
DEFAULT_TOLERANCES = {
    'bin_tol': DEFAULT_BIN_TOL,
    'gain_atol': DEFAULT_GAIN_ATOL,
    'gain_rtol': DEFAULT_GAIN_RTOL,
    'value_atol': DEFAULT_VALUE_ATOL,
    'value_rtol': DEFAULT_VALUE_RTOL,
}
SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
SPLIT_FIELDS = ('feature', 'bin', 'nan_left', 'children')


def make_dataset(*, seed, nrows, nfeatures, nclasses):
    """Generate the balanced Gaussian classification workload for this harness."""
    if nclasses < 2 or nrows < nclasses or nfeatures < 1:
        raise ValueError('need at least two classes, one row per class and one feature')

    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(nrows, dtype=np.int32) % nclasses)
    centers = rng.standard_normal((nclasses, nfeatures), dtype=np.float32)
    X = rng.standard_normal((nrows, nfeatures), dtype=np.float32) + centers[y]
    return X, y


def fit_replica(X, y, config, deterministic):
    from py_boost import GradientBoosting

    model = GradientBoosting(
        loss=config['loss'], ntrees=config['ntrees'], max_depth=config['max_depth'],
        lr=config['lr'], min_data_in_leaf=config['min_data_in_leaf'],
        subsample=config['subsample'], colsample=config['colsample'],
        full_leaf_values=config['full_leaf_values'], min_gain_to_split=0,
        deterministic_histogram=deterministic, deterministic_leaf_values=deterministic,
        deterministic_histogram_layout=config['layout'], seed=config['seed'],
        debug=True, es=0,
    )
    start = perf_counter()
    model.fit(X, y)
    return model, perf_counter() - start


def _tree_record(tree):
    """Address nodes and leaf values by root-to-node path within each group."""
    if tree.gains is None or tree.leaves is None:
        raise ValueError('split comparison requires a tree fitted with debug=True')

    splits = {}
    values = []
    for group in range(tree.ngroups):
        nodes = [(0, '')]
        leaf_ids = []
        while nodes:
            node, path = nodes.pop()
            feature = int(tree.feats[group, node])
            if feature < 0:
                leaf_ids.append(tree.leaves[node, group])
                continue

            left, right = tree.split[group, node]
            splits[group, path] = {
                'feature': feature,
                'bin': int(tree.bin_splits[group, node]),
                'nan_left': bool(tree.nans[group, node]),
                'children': [bool(tree.feats[group, child] >= 0) for child in (left, right)],
                'gain': float(tree.gains[group, node]),
            }
            nodes.extend(((right, path + 'R'), (left, path + 'L')))

        outputs = tree.group_index == group
        values.append(tree.values[leaf_ids][:, outputs].ravel())

    leaf_values = np.concatenate(values)
    output_groups = tree.group_index.tolist()
    structure = [(group, path, *(node[key] for key in SPLIT_FIELDS))
                 for (group, path), node in sorted(splits.items())]
    return {
        'splits': splits,
        'values': leaf_values,
        'output_groups': output_groups,
        'structure_fingerprint': sha256(json.dumps(
            [output_groups, structure], separators=(',', ':')).encode('ascii')).hexdigest(),
        'leaf_value_fingerprint': array_fingerprint(leaf_values),
    }


def replica_record(model, X, elapsed):
    trees = [_tree_record(tree) for tree in model.models]
    postprocess = model.postprocess_fn
    try:
        model.postprocess_fn = model._default_postprocess_fn
        logits = model.predict(X)
    finally:
        model.postprocess_fn = postprocess
        model.to_cpu()

    return {
        'trees': trees,
        'logits': logits,
        'elapsed_seconds': elapsed,
        'prediction_fingerprint': array_fingerprint(logits),
        'structure_fingerprints': [tree['structure_fingerprint'] for tree in trees],
        'leaf_value_fingerprints': [tree['leaf_value_fingerprint'] for tree in trees],
    }


def strip_heavy_fields(record):
    return {key: value for key, value in record.items() if key not in {'trees', 'logits'}}


def _split_changes(reference, replica, paths):
    changes = []
    for group, path in sorted(paths):
        before = reference.get((group, path))
        after = replica.get((group, path))
        if before is not None and after is not None:
            if all(before[key] == after[key] for key in SPLIT_FIELDS):
                continue
        parent = group, path[:-1]
        changes.append({
            'group': group, 'path': path, 'before': before, 'after': after,
            'aligned': not path or (parent in reference and parent in replica),
        })
    return changes


def _split_class(change, tolerances):
    before, after = change['before'], change['after']
    atol, rtol = tolerances['gain_atol'], tolerances['gain_rtol']
    if before is None or after is None:
        split = after if before is None else before
        if change['aligned'] and abs(split['gain']) <= atol:
            return 'gain_equivalent'
    else:
        same_route = all(before[key] == after[key] for key in ('feature', 'nan_left', 'children'))
        bin_gap = abs(before['bin'] - after['bin'])
        if same_route and 0 < bin_gap <= tolerances['bin_tol']:
            return 'near_tie'
        if abs(before['gain'] - after['gain']) <= atol + rtol * abs(before['gain']):
            return 'gain_equivalent'
    return 'hard_flip'


def reclassify_splits(comparison, tolerances):
    """Apply a new split tolerance to stored decisions without fitting a model."""
    trees = []
    classes = Counter()
    hard_nodes = 0
    for tree in comparison['trees']:
        changes = [
            {**change, 'classification': _split_class(change, tolerances)}
            for change in tree['elementwise']
        ]
        node_classes = Counter(change['classification'] for change in changes)
        tree_class = next(
            (label for label in ('hard_flip', 'gain_equivalent', 'near_tie') if node_classes[label]),
            'identical',
        )
        classes[tree_class] += 1
        hard_nodes += node_classes['hard_flip']
        trees.append({**tree, 'elementwise': changes, 'structural_class': tree_class})

    return {**comparison, 'trees': trees, 'tolerant': {
        **comparison['tolerant'],
        'hard_flip_trees': classes['hard_flip'],
        'gain_equivalent_trees': classes['gain_equivalent'],
        'near_tie_trees': classes['near_tie'],
        'hard_flip_split_nodes': hard_nodes,
    }}


def _max_abs_diff(before, after):
    return float(np.max(np.abs(after.astype(np.float64) - before), initial=0.0))


def compare_replica(reference, replica, tolerances=None):
    tolerances = DEFAULT_TOLERANCES if tolerances is None else tolerances
    if len(reference['trees']) != len(replica['trees']):
        raise ValueError('replicas must have the same number of trees')
    if reference['logits'].shape != replica['logits'].shape:
        raise ValueError('replicas must predict the same rows and outputs')

    trees = []
    structural = []
    for index, (before, after) in enumerate(zip(reference['trees'], replica['trees'])):
        if before['output_groups'] != after['output_groups']:
            raise ValueError('replicas must use the same output groups')

        left, right = before['splits'], after['splits']
        paths = left.keys() | right.keys()
        changes = _split_changes(left, right, paths)
        tree = {'tree': index, 'split_nodes': len(paths), 'elementwise': changes}
        if changes:
            structural.append(tree)
        else:
            tree.update(
                values_bitwise_equal=before['leaf_value_fingerprint'] == after['leaf_value_fingerprint'],
                values_allclose=bool(np.allclose(
                    after['values'], before['values'],
                    atol=tolerances['value_atol'], rtol=tolerances['value_rtol'],
                )),
                values_max_abs_diff=_max_abs_diff(before['values'], after['values']),
            )
        trees.append(tree)

    matched = [tree for tree in trees if not tree['elementwise']]
    total_split_nodes = sum(tree['split_nodes'] for tree in trees)
    return reclassify_splits({
        'trees': trees,
        'total_trees': len(trees),
        'flipped_trees': len(structural),
        'leaf_value_only_trees': sum(not tree['values_bitwise_equal'] for tree in matched),
        'flipped_split_nodes': sum(len(tree['elementwise']) for tree in structural),
        'total_split_nodes': total_split_nodes,
        'first_divergent_tree': structural[0]['tree'] if structural else None,
        'prediction_bitwise_equal': reference['prediction_fingerprint'] == replica['prediction_fingerprint'],
        'prediction_max_abs_diff': _max_abs_diff(reference['logits'], replica['logits']),
        'tolerant': {
            'total_split_nodes': total_split_nodes,
            'values_allclose_trees': sum(tree['values_allclose'] for tree in matched),
            'value_drift_trees': sum(not tree['values_allclose'] for tree in matched),
            'values_max_abs_diff_max': max((tree['values_max_abs_diff'] for tree in matched), default=0.0),
            'root_divergence_depths': [
                min(len(change['path']) for change in tree['elementwise']) for tree in structural
            ],
        },
    }, tolerances)


def wilson_ci(successes, total):
    """Return a point estimate and descriptive 95% Wilson limits."""
    if total == 0:
        return 0.0, 0.0, 1.0
    point = successes / total
    z = 1.959963984540054
    z2 = z * z
    center = point + z2 / (2 * total)
    radius = z * sqrt(point * (1 - point) / total + z2 / (4 * total ** 2))
    denominator = 1 + z2 / total
    lower = 0.0 if successes == 0 else (center - radius) / denominator
    upper = 1.0 if successes == total else (center + radius) / denominator
    return point, lower, upper


def base_report(experiment, config, arms, X, y):
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {
        'experiment': experiment,
        'schema_version': SCHEMA_VERSION,
        'config': {**config, 'min_gain_to_split': 0},
        'arms': arms,
        'dataset_recipe': 'balanced_gaussian_v1',
        'dataset_fingerprint': array_fingerprint(X, y),
        'source_fingerprint': source_fingerprint(ROOT),
        'environment': {
            'device_name': props['name'].decode('utf-8'),
            'compute_capability': [props['major'], props['minor']],
            'cupy': cp.__version__,
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'numpy': np.__version__,
            'python': platform.python_version(),
        },
        'detail': {},
    }


def write_artifact(filename, report):
    if Path(filename).name != filename:
        raise ValueError('artifact file name must not contain a directory')

    folder = ROOT / 'results' / 'proving'
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    stem = Path(filename).stem
    for sequence in count(1):
        path = folder / (filename if sequence == 1 else f'{stem}_{sequence}.json')
        try:
            with path.open('x', encoding='utf-8') as file:
                file.write(payload)
        except FileExistsError:
            continue
        return path
