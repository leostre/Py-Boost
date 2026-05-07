import functools
import inspect
import types
from typing import Dict, Iterable

from experiments.core.gpu import GPUTimer


def time_patch_methods(model, method_names: Iterable[str]) -> Dict[str, dict]:
    """
    Monkey‑patch selected methods on a model (and its params) to record
    cumulative GPU time and call counts.

    Returns a dict mapping method name to ``{\"total_time\", \"call_count\"}``.
    """
    timing_data = {}

    def create_timed_method(original_method, method_name):
        if method_name not in timing_data:
            timing_data[method_name] = {"total_time": 0.0, "call_count": 0}

        underlying = getattr(original_method, "__func__", original_method)

        @functools.wraps(underlying)
        def timed_wrapper(*args, **kwargs):
            with GPUTimer() as timer:
                result = original_method(*args, **kwargs)
            timing_data[method_name]["total_time"] += timer.time
            timing_data[method_name]["call_count"] += 1
            return result

        # Marker to make patching idempotent across folds/runs.
        timed_wrapper._is_timed_wrapper = True
        return timed_wrapper

    def patch_method(obj, method_name, original_method):
        # Skip methods that are already wrapped to avoid recursive wrapping.
        maybe_func = getattr(original_method, "__func__", original_method)
        if getattr(maybe_func, "_is_timed_wrapper", False):
            return

        timed_method = create_timed_method(original_method, method_name)
        if inspect.isclass(obj):
            setattr(obj, method_name, timed_method)
        else:
            if hasattr(original_method, "__self__") and original_method.__self__ is not None:
                setattr(obj, method_name, types.MethodType(timed_method, original_method.__self__))
            else:
                setattr(obj, method_name, types.MethodType(timed_method, obj))

    objects_to_check = [model]
    if hasattr(model, "params"):
        objects_to_check.append(model.params)

    for obj in objects_to_check:
        for attr_name in dir(obj):
            try:
                attr = getattr(obj, attr_name)
                for method_name in method_names:
                    if hasattr(attr, method_name):
                        method = getattr(attr, method_name)
                        if callable(method):
                            patch_method(attr, method_name, method)
            except Exception:
                continue

    return timing_data


def log_timing_data(timing_data, mlflow, fold: int) -> None:
    """
    Log timing statistics for patched methods into MLflow.
    """
    for method_name, data in timing_data.items():
        if data["call_count"] > 0:
            mlflow.log_metric(
                f"{method_name}_total_time_fold_{fold}",
                data["total_time"],
                step=fold,
            )
            mlflow.log_metric(
                f"{method_name}_calls_fold_{fold}",
                data["call_count"],
                step=fold,
            )
            mlflow.log_metric(
                f"{method_name}_avg_time_fold_{fold}",
                data["total_time"] / data["call_count"],
                step=fold,
            )

