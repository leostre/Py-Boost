"""Generate a controlled low-rank optimization trajectory for RMT diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_trajectory(
    rows: int,
    outputs: int,
    iterations: int,
    rank: int,
    signal_to_noise: float,
    change_iteration: int,
    seed: int,
) -> np.ndarray:
    """Return a trajectory with known rank and an optional subspace rotation."""

    if min(rows, outputs, iterations, rank) < 1:
        raise ValueError('all dimensions and rank must be positive')
    if rank > outputs:
        raise ValueError('rank cannot exceed the number of outputs')
    if signal_to_noise <= 0:
        raise ValueError('signal_to_noise must be positive')

    rng = np.random.default_rng(seed)
    basis_before, _ = np.linalg.qr(rng.normal(size=(outputs, rank)))
    basis_after, _ = np.linalg.qr(rng.normal(size=(outputs, rank)))
    noise_scale = 1 / np.sqrt(signal_to_noise)
    trajectory = np.empty((iterations, rows, outputs), dtype=np.float32)
    for iteration in range(iterations):
        factors = rng.normal(size=(rows, rank))
        basis = basis_before if iteration + 1 < change_iteration else basis_after
        signal = factors @ basis.T
        noise = rng.normal(scale=noise_scale, size=signal.shape)
        trajectory[iteration] = signal + noise
    return trajectory


def save_trajectory(
    trajectory: np.ndarray,
    path: Path,
    generator_parameters=None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = list(range(1, trajectory.shape[0] + 1))
    manifest = {
        'representation': 'z',
        'window': None,
        'regularization': 1.0,
        'every': 1,
        'dtype': str(trajectory.dtype),
        'iterations': iterations,
        'shapes': {'normalized': list(trajectory.shape[1:])},
        'synthetic': True,
        'generator_parameters': dict(generator_parameters or {}),
    }
    arrays = {'manifest': np.asarray(json.dumps(manifest))}
    arrays.update(
        {
            f'normalized_{iteration:04d}': trajectory[iteration - 1]
            for iteration in iterations
        }
    )
    np.savez_compressed(path, **arrays)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--outputs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--signal-to-noise', type=float, default=4.0)
    parser.add_argument('--change-iteration', type=int, default=17)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/spectral/synthetic_trajectory.npz'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = generate_trajectory(
        args.rows,
        args.outputs,
        args.iterations,
        args.rank,
        args.signal_to_noise,
        args.change_iteration,
        args.seed,
    )
    print(
        save_trajectory(
            trajectory,
            args.output,
            generator_parameters={
                'rows': args.rows,
                'outputs': args.outputs,
                'iterations': args.iterations,
                'rank': args.rank,
                'signal_to_noise': args.signal_to_noise,
                'change_iteration': args.change_iteration,
                'seed': args.seed,
            },
        ).resolve()
    )


if __name__ == '__main__':
    main()
