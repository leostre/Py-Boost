"""Source integrity and preregistered five-way development roles, no training."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import requests
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from experiments.a100_runner import ROOT, emit, sha256, write_report, read_dataset, make_partitions
from experiments.tiny_residual_study import calibration_partition


@dataclass(frozen=True)
class PublicSource:
    name: str
    samples: int
    features: int
    outputs: int
    files: tuple[str, ...]


PUBLIC = (
    PublicSource('scene', 2407, 294, 6, ('scene_train.bz2', 'scene_test.bz2')),
    PublicSource('bibtex', 7395, 1836, 159, ('bibtex.bz2',)),
)
BASE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/'


def indicator_with_complete_label_domain(rows, outputs):
    """Accept only a fully observed, unambiguous 0..C-1 or 1..C domain.

    A missing boundary label must not silently shift an incomplete domain.
    This loader works on the complete public file, before any subsampling.
    """
    values = sorted({float(v) for row in rows for v in row})
    if any(not np.isfinite(v) or v != int(v) for v in values):
        raise ValueError('noninteger or nonfinite public label ID')
    if values == list(range(outputs)):
        origin = 0
    elif values == list(range(1, outputs+1)):
        origin = 1
    else:
        raise ValueError('complete public label domain is missing or ambiguous')
    target = np.zeros((len(rows), outputs), np.float32)
    for i, row in enumerate(rows):
        if len(set(row)) != len(row):
            raise ValueError('duplicate label ID in one row')
        target[i, [int(v)-origin for v in row]] = 1
    return target, {str(int(v)): int(v)-origin for v in values}


def validate_roles(parts, count):
    names = ('train', 'calibration', 'control', 'selection', 'test')
    if set(parts) != set(names):
        raise ValueError('unknown development role set')
    joined = []
    for key in names:
        rows = np.asarray(parts[key])
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
            raise ValueError('invalid or empty role indices')
        if rows.min() < 0 or rows.max() >= count:
            raise ValueError('role index outside dataset')
        joined.extend(rows.tolist())
    if sorted(joined) != list(range(count)):
        raise ValueError('role overlap, omissions or duplicates')


def control_selection_partition(target, selection, task, outputs, seed):
    selection = np.asarray(selection, np.int64)
    y = np.eye(outputs)[target[selection].astype(int)] if task == 'multiclass' else target[selection]
    a, b = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.5,
        random_state=seed+200000).split(np.zeros((len(y), 1)), y))
    return selection[a], selection[b]


def array_profile(x, y, task):
    c = int(y.max()+1) if task == 'multiclass' else y.shape[1]
    if x.ndim != 2 or len(x) != len(y) or np.isinf(x).any():
        raise ValueError('invalid feature/target arrays')
    if task == 'multilabel' and (y.ndim != 2 or not np.isin(y, [0, 1]).all()):
        raise ValueError('nonbinary multilabel target')
    labels = np.eye(c, dtype=np.int8)[y.astype(int)] if task == 'multiclass' else y
    support = labels.sum(axis=0).astype(int)
    return dict(rows=len(x), features=x.shape[1], outputs=c, task=task,
        feature_nan_count=int(np.isnan(x).sum()), feature_zero_fraction=float(np.mean(x == 0)),
        feature_duplicate_rows=int(len(x)-len(np.unique(x, axis=0))),
        target_support=support.tolist(), label_density=float(labels.mean()),
        label_cardinality=float(labels.sum(axis=1).mean()),
        support_minimum=int(support.min()), support_maximum=int(support.max()))


def download_public(spec, output):
    directory = output/'public_sources'/spec.name
    directory.mkdir(parents=True, exist_ok=True)
    matrices, labels, sources = [], [], []
    for filename in spec.files:
        path = directory/filename
        with requests.get(BASE_URL+filename, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with path.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
        matrix, rows = load_svmlight_file(path, n_features=spec.features, multilabel=True,
                                         zero_based=False, dtype=np.float32)
        matrices.append(matrix)
        labels.extend(rows)
        sources.append(dict(url=BASE_URL+filename, sha256=sha256(path), rows=matrix.shape[0],
                            file=str(path)))
    x = sparse.vstack(matrices, format='csr').toarray().astype(np.float32)
    y, mapping = indicator_with_complete_label_domain(labels, spec.outputs)
    if x.shape != (spec.samples, spec.features) or y.shape != (spec.samples, spec.outputs):
        raise RuntimeError('public dataset shape differs from preregistration')
    path = output/'data'/(spec.name+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, target=y, source_indices=np.arange(len(x), dtype=np.int64))
    return x, y, np.arange(len(x), dtype=np.int64), dict(path=str(path), sha256=sha256(path), task='multilabel',
        sources=sources, label_id_mapping=mapping, features_shape=list(x.shape), target_shape=list(y.shape),
        split_policy='union of public files followed by new registered splits; not original published split')


def expanded_budget_data(config, report, output):
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_expanded_budget_data.py'],
                   cwd=ROOT, check=True, timeout=300)
    old_manifest = json.loads((ROOT/'experiments/a100_data_manifest.json').read_text())
    datasets, metadata, locks = {}, {}, {}
    for name in config['datasets']:
        public = next((s for s in PUBLIC if s.name == name), None)
        if public:
            x, y, indices, entry = download_public(public, output)
        else:
            x, y, indices, old_entry = read_dataset(name, config)
            entry = {**old_entry, 'path': str(Path(config['data_dir'])/(name+'.npz'))}
            locks[entry['path']] = entry['sha256']
        datasets[name] = (x, y, indices)
        metadata[name] = {**entry, 'profile': array_profile(x, y, entry['task'])}
        emit('expanded_dataset_ready', dataset=name, path=entry['path'], sha256=entry['sha256'],
             profile=metadata[name]['profile'])
    cases = []
    for name in config['datasets']:
        x, y, indices = datasets[name]
        task = metadata[name]['task']
        c = metadata[name]['profile']['outputs']
        for seed in config['seeds_by_dataset'][name]:
            old = name in ('yeast', 'mediamill', 'helena')
            if old:
                source = Path(config['source_dir'])/f'{name}_{seed}'/'splits.npz'
                development = Path(config['inference_parent_dir'])/f'{name}_{seed}'/'development_splits.npz'
                for path in (source, development):
                    locks[str(path)] = sha256(path)
                with np.load(source, allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['source_indices'], indices)
                    train, validation, test = (f[k].copy() for k in ('train', 'validation', 'test'))
                with np.load(development, allow_pickle=False) as f:
                    np.testing.assert_array_equal(train, f['train'])
                    calibration, previous_selection = f['calibration'].copy(), f['selection'].copy()
                np.testing.assert_array_equal(np.sort(np.r_[calibration, previous_selection]), np.sort(validation))
            else:
                train, validation, test = make_partitions(y, task, seed)
                a, b = calibration_partition(y[validation], task, c, seed)
                calibration, previous_selection = validation[a], validation[b]
            control, selection = control_selection_partition(y, previous_selection, task, c, seed)
            parts = dict(train=train, calibration=calibration, control=control, selection=selection, test=test)
            validate_roles(parts, len(x))
            directory = output/'splits'/f'{name}_{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/'splits.npz'
            np.savez(path, **parts, source_indices=indices)
            labels = np.eye(c, dtype=np.int8)[y.astype(int)] if task == 'multiclass' else y
            cases.append(dict(dataset=name, seed=seed, split_file=str(path), split_sha256=sha256(path),
                data_path=metadata[name]['path'], data_sha256=metadata[name]['sha256'], task=task, outputs=c,
                historical_outer_split=old, sizes={k: len(v) for k, v in parts.items()},
                supports={k: labels[v].sum(axis=0).astype(int).tolist() for k, v in parts.items()}))
    for path, expected in locks.items():
        if sha256(path) != expected:
            raise RuntimeError('old input changed while preparing new roles')
    manifest = dict(version=1, protocol='60/10/5/5/20; old outer and calibration preserved',
                    datasets=metadata, cases=cases, old_file_locks=locks, tree_fits=0,
                    source_sample_policy='no label/feature filtering; all source columns retained')
    path = output/'manifest.json'
    write_report(path, manifest)
    report.update(status='complete', tree_fits=0, gpu_runs=0, manifest_path=str(path),
                  manifest_sha256=sha256(path), cases=len(cases), datasets=list(metadata),
                  old_file_locks_verified=len(locks), elapsed_seconds=time.perf_counter()-started)
    emit('expanded_data_complete', **{k: report[k] for k in
        ('status', 'tree_fits', 'gpu_runs', 'manifest_path', 'manifest_sha256', 'cases', 'datasets', 'elapsed_seconds')})
