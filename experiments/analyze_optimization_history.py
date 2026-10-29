"""Offline output-subspace diagnostics for saved HASBoost trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SpectralWindowResult:
    """Serializable diagnostics for one representation and history window."""

    representation: str
    window: int
    end_iteration: int
    effective_rank: float
    spike_count: int
    largest_eigenvalue: float
    largest_null_p95: float
    null_row_limit: Optional[int]
    mean_principal_angle_deg: Optional[float]
    max_principal_angle_deg: Optional[float]
    output_probabilities: Tuple[float, ...]


def output_covariance(history: np.ndarray) -> np.ndarray:
    """Compute C_out = sum_t Z_t.T @ Z_t / (n * W)."""

    if history.ndim != 3 or min(history.shape) < 1:
        raise ValueError('history must have shape (window, rows, outputs)')
    window, rows, _ = history.shape
    return np.einsum('tni,tnj->ij', history, history) / (window * rows)


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Return entropy effective rank for non-negative eigenvalues."""

    non_negative = np.maximum(np.asarray(eigenvalues, dtype=float), 0)
    total = non_negative.sum()
    if total <= 0:
        return 0.0
    probabilities = non_negative / total
    positive = probabilities > 0
    return float(np.exp(-(probabilities[positive] * np.log(probabilities[positive])).sum()))


def permutation_null_eigenvalues(
    history: np.ndarray,
    n_permutations: int,
    random_state: np.random.Generator,
) -> np.ndarray:
    """Destroy cross-output alignment by permuting rows per iteration and output."""

    if n_permutations < 1:
        raise ValueError('n_permutations must be positive')
    window, rows, outputs = history.shape
    null_eigenvalues = np.empty((n_permutations, outputs), dtype=float)
    permuted = np.empty_like(history)
    for permutation in range(n_permutations):
        for time_index in range(window):
            for output_index in range(outputs):
                order = random_state.permutation(rows)
                permuted[time_index, :, output_index] = history[
                    time_index, order, output_index
                ]
        null_eigenvalues[permutation] = np.linalg.eigvalsh(
            output_covariance(permuted)
        )[::-1]
    return null_eigenvalues


def rolling_window_end_positions(
    history_length: int,
    window: int,
    stride: int,
) -> Tuple[int, ...]:
    """Return deterministic window endpoints and always include the last one."""

    if history_length < 1:
        raise ValueError('history_length must be positive')
    if not 1 <= window <= history_length:
        raise ValueError('window must be within the available history')
    if stride < 1:
        raise ValueError('stride must be positive')
    positions = list(range(window - 1, history_length, stride))
    if positions[-1] != history_length - 1:
        positions.append(history_length - 1)
    return tuple(positions)


def _covariance_per_iteration(
    matrices: Mapping[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Compute one normalized output Gram matrix per saved iteration."""

    return {
        iteration: matrix.T @ matrix / matrix.shape[0]
        for iteration, matrix in matrices.items()
    }


def _null_covariance_per_iteration(
    matrices: Mapping[int, np.ndarray],
    n_permutations: int,
    random_state: np.random.Generator,
    max_rows: Optional[int],
) -> Dict[int, np.ndarray]:
    """Build reusable empirical-null covariances for overlapping windows."""

    if n_permutations < 1:
        raise ValueError('n_permutations must be positive')
    if max_rows is not None and max_rows < 2:
        raise ValueError('max_rows must be at least two or None')

    result: Dict[int, np.ndarray] = {}
    for iteration, matrix in matrices.items():
        if max_rows is not None and matrix.shape[0] > max_rows:
            selected = matrix[
                random_state.choice(matrix.shape[0], size=max_rows, replace=False)
            ]
        else:
            selected = matrix
        rows, outputs = selected.shape
        permuted = np.empty_like(selected)
        null_covariances = np.empty(
            (n_permutations, outputs, outputs),
            dtype=float,
        )
        for permutation in range(n_permutations):
            for output_index in range(outputs):
                order = random_state.permutation(rows)
                permuted[:, output_index] = selected[order, output_index]
            null_covariances[permutation] = permuted.T @ permuted / rows
        result[iteration] = null_covariances
    return result


def safe_output_probabilities(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    null_p95: np.ndarray,
    beta: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert excess spike energy into a uniform-protected output policy."""

    if not 0 <= beta <= 1:
        raise ValueError('beta must satisfy 0 <= beta <= 1')
    if tau < 0:
        raise ValueError('tau must be non-negative')
    eigenvalues = np.asarray(eigenvalues)
    null_p95 = np.asarray(null_p95)
    if eigenvalues.shape != null_p95.shape:
        raise ValueError('observed and null eigenvalue arrays must match')

    spikes = eigenvalues > null_p95
    excess = np.maximum(eigenvalues - null_p95, 0)
    importance = np.full(eigenvectors.shape[0], tau, dtype=float)
    if spikes.any():
        importance += (eigenvectors[:, spikes] ** 2) @ excess[spikes]
    if importance.sum() <= 0:
        importance[:] = 1.0
    spectral = importance / importance.sum()
    uniform = np.full_like(spectral, 1 / spectral.size)
    return (1 - beta) * uniform + beta * spectral, spikes


def principal_angles_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return principal angles between two column subspaces in degrees."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError('subspaces must be matrices with a shared ambient dimension')
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.asarray([], dtype=float)
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular_values, -1, 1)))


def _load_snapshots(path: Path) -> Tuple[Mapping[str, object], Dict[int, Dict[str, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
        snapshots: Dict[int, Dict[str, np.ndarray]] = {
            int(iteration): {} for iteration in manifest['iterations']
        }
        for key in archive.files:
            if key == 'manifest':
                continue
            field, iteration_text = key.rsplit('_', 1)
            snapshots[int(iteration_text)][field] = np.asarray(archive[key])
    return manifest, snapshots


def _representation(
    snapshot: Mapping[str, np.ndarray],
    name: str,
    regularization: float,
) -> np.ndarray:
    if name == 'g':
        return snapshot['gradient']
    if name == 'z':
        if 'normalized' in snapshot:
            return snapshot['normalized']
        return snapshot['gradient'] / np.sqrt(
            np.maximum(snapshot['hessian'], 0) + regularization
        )
    if name == 'newton_step':
        return snapshot['gradient'] / (snapshot['hessian'] + regularization)
    if name == 'gh_separate':
        gradient = snapshot['gradient']
        hessian = np.broadcast_to(snapshot['hessian'], gradient.shape)
        gradient_scale = np.sqrt(np.mean(gradient ** 2)) + np.finfo(float).eps
        hessian_scale = np.sqrt(np.mean(hessian ** 2)) + np.finfo(float).eps
        return np.concatenate(
            (gradient / gradient_scale, hessian / hessian_scale),
            axis=0,
        )
    raise ValueError(f'Unsupported representation: {name}')


def analyze_trajectory(
    trajectory_path: Path,
    representations: Sequence[str],
    windows: Sequence[int],
    n_null: int,
    seed: int,
    beta: float,
    tau: float,
    regularization: Optional[float] = None,
    stride: int = 1,
    null_max_rows: Optional[int] = None,
) -> List[SpectralWindowResult]:
    """Analyze all complete rolling windows in one saved trajectory."""

    manifest, snapshots = _load_snapshots(trajectory_path)
    regularization = float(
        manifest['regularization'] if regularization is None else regularization
    )
    iterations = sorted(snapshots)
    rng = np.random.default_rng(seed)
    results: List[SpectralWindowResult] = []

    for representation in representations:
        matrices = {
            iteration: _representation(
                snapshots[iteration], representation, regularization
            )
            for iteration in iterations
        }
        observed_covariances = _covariance_per_iteration(matrices)
        null_covariances = _null_covariance_per_iteration(
            matrices,
            n_null,
            rng,
            null_max_rows,
        )
        previous_subspace: Optional[np.ndarray] = None
        for window in windows:
            if window < 1:
                raise ValueError('window sizes must be positive')
            if window > len(iterations):
                continue
            previous_subspace = None
            for end_position in rolling_window_end_positions(
                len(iterations), window, stride
            ):
                selected_iterations = iterations[end_position - window + 1 : end_position + 1]
                covariance = sum(
                    observed_covariances[iteration]
                    for iteration in selected_iterations
                ) / window
                null_values = np.stack(
                    [
                        np.linalg.eigvalsh(
                            sum(
                                null_covariances[iteration][permutation]
                                for iteration in selected_iterations
                            ) / window
                        )[::-1]
                        for permutation in range(n_null)
                    ]
                )
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                null_p95 = np.quantile(null_values, 0.95, axis=0)
                probabilities, spikes = safe_output_probabilities(
                    eigenvalues,
                    eigenvectors,
                    null_p95,
                    beta,
                    tau,
                )
                subspace = eigenvectors[:, spikes]
                angles = (
                    np.asarray([], dtype=float)
                    if previous_subspace is None
                    else principal_angles_degrees(previous_subspace, subspace)
                )
                results.append(
                    SpectralWindowResult(
                        representation=representation,
                        window=window,
                        end_iteration=selected_iterations[-1],
                        effective_rank=effective_rank(eigenvalues),
                        spike_count=int(spikes.sum()),
                        largest_eigenvalue=float(eigenvalues[0]),
                        largest_null_p95=float(null_p95[0]),
                        null_row_limit=null_max_rows,
                        mean_principal_angle_deg=(
                            float(angles.mean()) if angles.size else None
                        ),
                        max_principal_angle_deg=(
                            float(angles.max()) if angles.size else None
                        ),
                        output_probabilities=tuple(float(value) for value in probabilities),
                    )
                )
                previous_subspace = subspace
    return results


def save_analysis(results: Iterable[SpectralWindowResult], output_dir: Path) -> Dict[str, Path]:
    """Save a detailed JSON file and a compact per-window CSV table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = [asdict(result) for result in results]
    json_path = output_dir / 'spectral_windows.json'
    csv_path = output_dir / 'spectral_windows.csv'
    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(materialized, stream, indent=2)
    scalar_rows = [
        {key: value for key, value in row.items() if key != 'output_probabilities'}
        for row in materialized
    ]
    if scalar_rows:
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
            writer.writeheader()
            writer.writerows(scalar_rows)
    return {'json': json_path, 'csv': csv_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory', type=Path)
    parser.add_argument(
        '--representations',
        nargs='+',
        choices=('g', 'z', 'newton_step', 'gh_separate'),
        default=['z'],
    )
    parser.add_argument('--windows', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--n-null', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=1e-6)
    parser.add_argument('--regularization', type=float)
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Analyze every Nth complete rolling window; the final window is always kept.',
    )
    parser.add_argument(
        '--null-max-rows',
        type=int,
        help='Optional deterministic row budget for the empirical null only.',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('results/spectral'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_trajectory(
        args.trajectory,
        args.representations,
        args.windows,
        args.n_null,
        args.seed,
        args.beta,
        args.tau,
        args.regularization,
        args.stride,
        args.null_max_rows,
    )
    paths = save_analysis(results, args.output_dir)
    print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, indent=2))


if __name__ == '__main__':
    main()
