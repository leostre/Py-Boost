"""Optional callbacks for recording reproducible optimization trajectories."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

import numpy as np

SUPPORTED_REPRESENTATIONS = frozenset({'g', 'gh', 'z', 'ghz'})


def normalized_newton_signal(
    gradient: np.ndarray,
    hessian: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Compute Z = G / sqrt(max(H, 0) + regularization)."""

    if regularization <= 0:
        raise ValueError('regularization must be positive')
    return gradient / np.sqrt(np.maximum(hessian, 0) + regularization)


class OptimizationTrajectoryRecorder:
    """Copy selected full gradient/Hessian representations to host memory."""

    def __init__(
        self,
        representation: str = 'z',
        window: Optional[int] = None,
        regularization: float = 1.0,
        every: int = 1,
        dtype: Any = np.float32,
    ) -> None:
        if representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f'Unknown representation: {representation}. '
                f'Available values: {", ".join(sorted(SUPPORTED_REPRESENTATIONS))}'
            )
        if window is not None and window < 1:
            raise ValueError('window must be positive or None')
        if regularization <= 0:
            raise ValueError('regularization must be positive')
        if every < 1:
            raise ValueError('every must be positive')

        self.representation = representation
        self.window = window
        self.regularization = float(regularization)
        self.every = int(every)
        self.dtype = np.dtype(dtype)
        self.snapshots: Deque[Dict[str, Any]] = deque(maxlen=window)

    def before_train(self, build_info):
        self.snapshots.clear()

    @staticmethod
    def _to_host(array: Any) -> np.ndarray:
        getter = getattr(array, 'get', None)
        return np.asarray(getter() if callable(getter) else array)

    def before_iteration(self, build_info):
        iteration = int(build_info['num_iter']) + 1
        if (iteration - 1) % self.every:
            return

        train = build_info['data']['train']
        gradient_device = train.get('grad')
        hessian_device = train.get('hess')
        if gradient_device is None or hessian_device is None:
            return

        gradient = self._to_host(gradient_device).astype(self.dtype, copy=False)
        hessian = self._to_host(hessian_device).astype(self.dtype, copy=False)
        snapshot: Dict[str, Any] = {'iteration': iteration}
        if 'g' in self.representation:
            snapshot['gradient'] = gradient.copy()
        if 'h' in self.representation:
            snapshot['hessian'] = hessian.copy()
        if 'z' in self.representation:
            snapshot['normalized'] = normalized_newton_signal(
                gradient,
                hessian,
                self.regularization,
            ).astype(self.dtype, copy=False)
        self.snapshots.append(snapshot)

    def after_iteration(self, build_info):
        return False

    def after_train(self, build_info):
        return None

    def manifest(self) -> Dict[str, Any]:
        """Return JSON-compatible metadata without copying trajectory arrays."""

        snapshots: Iterable[Dict[str, Any]] = tuple(self.snapshots)
        shapes = {
            key: list(value.shape)
            for snapshot in snapshots
            for key, value in snapshot.items()
            if key != 'iteration'
        }
        return {
            'representation': self.representation,
            'window': self.window,
            'regularization': self.regularization,
            'every': self.every,
            'dtype': self.dtype.name,
            'iterations': [snapshot['iteration'] for snapshot in snapshots],
            'shapes': shapes,
        }

    def save(self, path: Path) -> Path:
        """Persist snapshots and a manifest in one compressed NPZ artifact."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, Any] = {
            'manifest': np.asarray(json.dumps(self.manifest())),
        }
        for snapshot in self.snapshots:
            iteration = snapshot['iteration']
            for key, value in snapshot.items():
                if key != 'iteration':
                    arrays[f'{key}_{iteration:04d}'] = value
        np.savez_compressed(path, **arrays)
        return path
