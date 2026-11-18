"""Inspectable identities for source snapshots, datasets, and exact splits."""

from hashlib import sha256
from pathlib import Path

import numpy as np


def array_fingerprint(*arrays):
    digest = sha256()
    for values in arrays:
        values = np.ascontiguousarray(values)
        digest.update(str(values.shape).encode('ascii'))
        digest.update(values.dtype.str.encode('ascii'))
        digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def source_fingerprint(root: Path):
    digest = sha256()
    for folder in ('py_boost', 'experiments'):
        for path in sorted((root / folder).rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode('utf-8'))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def paired_seed_map(model_seeds, split_seeds=None):
    model_seeds = tuple(model_seeds)
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError('model seeds must be unique')
    split_seeds = model_seeds if split_seeds is None else tuple(split_seeds)
    if len(split_seeds) != len(model_seeds):
        raise ValueError('one split seed is required for each model seed')
    return dict(zip(model_seeds, split_seeds))
