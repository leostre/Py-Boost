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

    def __enter__(self) -> "GPUTimer":
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()
        self.start_event.record()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        assert self.end_event is not None and self.start_event is not None
        self.end_event.record()
        self.end_event.synchronize()
        self.time = cp.cuda.get_elapsed_time(self.start_event, self.end_event)


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


def safe_predict(model, X_test, batch_size: int = 1000):
    """
    Run predictions in batches to reduce peak memory usage.
    """
    predictions = []
    n = len(X_test)
    for i in range(0, n, batch_size):
        batch = X_test[i : min(i + batch_size, n)]
        pred_batch = model.predict(batch)
        predictions.append(pred_batch)
    import numpy as np

    return np.concatenate(predictions)

