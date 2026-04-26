import gc
from typing import Optional

import cupy as cp


class GPUTimer:
    """
    Simple CUDA event-based timer.

    Returns elapsed time in milliseconds in the ``time`` attribute.
    """

    def __init__(self) -> None:
        self.time: Optional[float] = None
        self.start_event: Optional[cp.cuda.Event] = None
        self.end_event: Optional[cp.cuda.Event] = None
        self._enabled: bool = True

    def __enter__(self) -> "GPUTimer":
        try:
            self.start_event = cp.cuda.Event()
            self.end_event = cp.cuda.Event()
            self.start_event.record()
            self._enabled = True
        except Exception:
            # If CUDA is already in a bad state (e.g. illegal address from a prior kernel),
            # timer creation itself can fail and mask the real error. Degrade to no-op.
            self._enabled = False
            self.start_event = None
            self.end_event = None
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if not self._enabled:
            return
        assert self.end_event is not None and self.start_event is not None
        try:
            self.end_event.record()
            self.end_event.synchronize()
            self.time = cp.cuda.get_elapsed_time(self.start_event, self.end_event)
        except Exception:
            # Keep timer best-effort only.
            self.time = None


def initialize_gpu_settings() -> None:
    """
    Configure CuPy memory allocator limits for experiments.

    This is extracted from several experiment scripts to centralize GPU setup.
    """
    if cp.is_available():
        try:
            cp.cuda.set_allocator(None)
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=512 * 1024**2)
        except Exception:
            # Best-effort; experiments should not crash if this fails.
            pass


def nuclear_cleanup() -> None:
    """
    Aggressively free Python and GPU memory between experiment runs.
    """
    gc.collect()

    if cp.is_available():
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass

    # Some experiments also use PyTorch; clear its CUDA cache when available.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Torch is optional; ignore if unavailable.
        pass


def _sklearn_feature_names_for_predict(model):
    """
    If the fitted estimator recorded feature names (e.g. LGBM on DataFrame),
    return them so batched predict can use matching pandas inputs and avoid
    sklearn's 'X does not have valid feature names' warning.
    """
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and len(estimators) > 0:
        sub = estimators[0]
        fn = getattr(sub, "feature_names_in_", None)
        if fn is not None and len(fn) > 0:
            return list(fn)
    return None


def safe_predict(model, X_test, batch_size: int = 1000):
    """
    Run predictions in batches to reduce peak memory usage.

    When the underlying estimator was fit with feature names (common for
    ``MultiOutputClassifier`` + ``LGBMClassifier``), batches are wrapped in a
    ``pandas.DataFrame`` with those column names so predict matches training.
    """
    import numpy as np
    import pandas as pd

    names = _sklearn_feature_names_for_predict(model)

    if isinstance(X_test, pd.DataFrame):
        n = len(X_test)
        predictions = []
        for i in range(0, n, batch_size):
            batch = X_test.iloc[i : min(i + batch_size, n)]
            pred_batch = model.predict(batch)
            predictions.append(pred_batch)
        return np.concatenate(predictions)

    X_arr = np.asarray(X_test)
    n = len(X_arr)
    predictions = []
    for i in range(0, n, batch_size):
        batch = X_arr[i : min(i + batch_size, n)]
        if names is not None and batch.ndim == 2 and batch.shape[1] == len(names):
            batch = pd.DataFrame(batch, columns=names)
        pred_batch = model.predict(batch)
        predictions.append(pred_batch)

    return np.concatenate(predictions)

