"""Dataset registry and effectful loaders for HASBoost benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from scipy import sparse
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Task(str, Enum):
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


@dataclass(frozen=True)
class DatasetSpec:
    """Stable identity and expected shape of one benchmark dataset."""

    name: str
    task: Task
    source: str
    data_id: Optional[int]
    expected_features: int
    expected_outputs: int
    default_max_samples: Optional[int]
    default_sketch_outputs: int


@dataclass(frozen=True)
class LoadedDataset:
    """Dense arrays and provenance required by the benchmark shell."""

    spec: DatasetSpec
    features: np.ndarray
    target: np.ndarray
    sample_indices: Optional[np.ndarray] = None

    @property
    def n_outputs(self) -> int:
        if self.spec.task is Task.MULTICLASS:
            return int(np.unique(self.target).size)
        return int(self.target.shape[1])


DATASETS: Dict[str, DatasetSpec] = {
    "helena": DatasetSpec(
        name="helena",
        task=Task.MULTICLASS,
        source="OpenML:41169",
        data_id=41169,
        expected_features=27,
        expected_outputs=100,
        default_max_samples=12_000,
        default_sketch_outputs=25,
    ),
    "birds": DatasetSpec(
        name="birds",
        task=Task.MULTILABEL,
        source="OpenML:40588",
        data_id=40588,
        expected_features=260,
        expected_outputs=19,
        default_max_samples=None,
        default_sketch_outputs=5,
    ),
    "yeast": DatasetSpec(
        name="yeast",
        task=Task.MULTILABEL,
        source="OpenML:40597",
        data_id=40597,
        expected_features=103,
        expected_outputs=14,
        default_max_samples=None,
        default_sketch_outputs=5,
    ),
    "mediamill": DatasetSpec(
        name="mediamill",
        task=Task.MULTILABEL,
        source="LIBSVM Tools MediaMill train/test (original MediaMill data)",
        data_id=None,
        expected_features=120,
        expected_outputs=101,
        default_max_samples=10_000,
        default_sketch_outputs=25,
    ),
}


def _numeric_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    try:
        return values.astype(np.float32)
    except (TypeError, ValueError):
        encoded = np.empty(values.shape, dtype=np.float32)
        for column in range(values.shape[1]):
            raw_column = values[:, column]
            try:
                encoded[:, column] = raw_column.astype(np.float32)
            except (TypeError, ValueError):
                _, codes = np.unique(raw_column.astype(str), return_inverse=True)
                encoded[:, column] = codes.astype(np.float32)
        return encoded


def _binary_targets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("multilabel target must be a two-dimensional matrix")
    try:
        numeric = values.astype(np.float32)
        unique = np.unique(numeric)
        if np.all(np.isin(unique, (0.0, 1.0))):
            return numeric
    except (TypeError, ValueError):
        pass

    encoded = np.empty(values.shape, dtype=np.float32)
    for column in range(values.shape[1]):
        classes, codes = np.unique(values[:, column].astype(str), return_inverse=True)
        if classes.size != 2:
            raise ValueError(
                f"multilabel target column {column} has {classes.size} values instead of 2"
            )
        encoded[:, column] = codes.astype(np.float32)
    return encoded


def _iterative_indices(
    target: np.ndarray,
    train_size,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise RuntimeError(
            "iterative-stratification is required for multilabel benchmark splits"
        ) from exc

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        test_size=(
            target.shape[0] - int(train_size)
            if isinstance(train_size, (int, np.integer))
            else 1 - float(train_size)
        ),
        random_state=seed,
    )
    dummy = np.zeros((target.shape[0], 1), dtype=np.uint8)
    return next(splitter.split(dummy, target))


def _subsample(
    features: np.ndarray,
    target: np.ndarray,
    task: Task,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not max_samples or max_samples >= features.shape[0]:
        return features, target
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    if task is Task.MULTICLASS:
        selected, _ = train_test_split(
            np.arange(features.shape[0]),
            train_size=max_samples,
            random_state=seed,
            stratify=target,
        )
    else:
        selected, _ = _iterative_indices(target, max_samples, seed)
    return features[selected], target[selected]


def _load_openml(spec: DatasetSpec, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    dataset = fetch_openml(
        data_id=spec.data_id,
        as_frame=False,
        data_home=str(cache_dir / "openml"),
    )
    features = _numeric_features(dataset.data)
    if spec.task is Task.MULTICLASS:
        target = LabelEncoder().fit_transform(np.asarray(dataset.target)).astype(np.int32)
    else:
        target = _binary_targets(np.asarray(dataset.target))
    return features, target


MEDIAMILL_BASE_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
    "multilabel/mediamill"
)
MEDIAMILL_FILES = ("train-exp1.svm.bz2", "test-exp1.svm.bz2")


def _download(url: str, destination: Path) -> None:
    """Stream one dataset file and publish it atomically after completion."""
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(destination)


def _labels_to_indicator(labels, n_outputs: int) -> sparse.csr_matrix:
    """Convert LIBSVM multi-label tuples into a fixed-width indicator matrix."""
    observed = [int(label) for row in labels for label in row]
    if not observed:
        raise ValueError("MediaMill label matrix is empty")
    minimum, maximum = min(observed), max(observed)
    if minimum >= 0 and maximum < n_outputs:
        offset = 0
    elif minimum >= 1 and maximum <= n_outputs:
        offset = 1
    else:
        raise ValueError(
            f"MediaMill labels span [{minimum}, {maximum}], expected 0- or 1-based "
            f"indices for {n_outputs} outputs"
        )

    rows = []
    columns = []
    for row_index, row_labels in enumerate(labels):
        rows.extend([row_index] * len(row_labels))
        columns.extend(int(label) - offset for label in row_labels)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(len(labels), n_outputs),
        dtype=np.float32,
    )


def _load_mediamill(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    dataset_dir = cache_dir / "mediamill"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    feature_parts = []
    target_parts = []
    for filename in MEDIAMILL_FILES:
        path = dataset_dir / filename
        if not path.exists():
            _download(f"{MEDIAMILL_BASE_URL}/{filename}", path)
        features, labels = load_svmlight_file(
            path,
            multilabel=True,
            n_features=DATASETS["mediamill"].expected_features,
            dtype=np.float32,
        )
        feature_parts.append(features)
        target_parts.append(
            _labels_to_indicator(labels, DATASETS["mediamill"].expected_outputs)
        )
    return sparse.vstack(feature_parts, format="csr"), sparse.vstack(
        target_parts, format="csr"
    )


def load_benchmark_dataset(
    name: str,
    max_samples: Optional[int] = None,
    dataset_seed: int = 42,
    cache_dir: Path = Path("data/benchmarks"),
) -> LoadedDataset:
    """Load, validate, subset, and densify one registered dataset."""
    try:
        spec = DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {name}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    if spec.data_id is not None:
        raw_features, raw_target = _load_openml(spec, cache_dir)
    elif name == "mediamill":
        raw_features, raw_target = _load_mediamill(cache_dir)
    else:
        raise ValueError(f"no loader configured for dataset: {name}")

    target = (
        np.asarray(raw_target)
        if spec.task is Task.MULTICLASS
        else _binary_targets(raw_target.toarray() if hasattr(raw_target, "toarray") else raw_target)
    )
    sample_indices = np.arange(raw_features.shape[0], dtype=np.int64)
    sample_limit = spec.default_max_samples if max_samples is None else max_samples

    # Sparse MediaMill features are subset before densification.
    if hasattr(raw_features, "shape") and sample_limit and sample_limit < raw_features.shape[0]:
        if spec.task is Task.MULTILABEL:
            selected, _ = _iterative_indices(target, sample_limit, dataset_seed)
        else:
            selected, _ = train_test_split(
                np.arange(raw_features.shape[0]),
                train_size=sample_limit,
                random_state=dataset_seed,
                stratify=target,
            )
        raw_features = raw_features[selected]
        target = target[selected]
        sample_indices = sample_indices[selected]
        sample_limit = None

    features = raw_features.toarray() if hasattr(raw_features, "toarray") else raw_features
    features = _numeric_features(features)
    features, target = _subsample(
        features,
        target,
        spec.task,
        sample_limit,
        dataset_seed,
    )

    if features.shape[1] != spec.expected_features:
        raise ValueError(
            f"{name} has {features.shape[1]} features; expected {spec.expected_features}"
        )
    n_outputs = int(np.unique(target).size) if spec.task is Task.MULTICLASS else target.shape[1]
    if n_outputs != spec.expected_outputs:
        raise ValueError(f"{name} has {n_outputs} outputs; expected {spec.expected_outputs}")
    return LoadedDataset(spec=spec, features=features, target=target, sample_indices=sample_indices)


def split_benchmark_indices(dataset: LoadedDataset, seed: int, test_size: float = 0.2):
    """Return subset-relative indices; callers can save the actual split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must satisfy 0 < test_size < 1")
    if dataset.spec.task is Task.MULTICLASS:
        return train_test_split(np.arange(dataset.features.shape[0]),
                                test_size=test_size, random_state=seed,
                                stratify=dataset.target)
    return _iterative_indices(dataset.target, 1 - test_size, seed)


def split_benchmark_dataset(
    dataset: LoadedDataset,
    seed: int,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible stratified split for one run seed."""
    train_indices, test_indices = split_benchmark_indices(dataset, seed, test_size)
    return (
        dataset.features[train_indices],
        dataset.features[test_indices],
        dataset.target[train_indices],
        dataset.target[test_indices],
    )
