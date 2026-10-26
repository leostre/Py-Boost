"""Small measurement contracts; no model fitting, CUDA, or test-label access."""
import json
import os
from pathlib import Path

from experiments.model_bank_core import traffic_schedule


def request_plan(tasks, count, traffic, seed=20261021):
    """Prefix of the OLD trace, not a freshly shuffled shorter trace."""
    if isinstance(count, bool) or not isinstance(count, int) or not tasks <= count <= 128:
        raise ValueError('short trace must cover every task and stay within old128')
    return traffic_schedule(tasks, 128, traffic, seed)[:count]


def worker_plan():
    return [(mode, family) for mode in ('fit_with_cache', 'fit_preprocessors')
            for family in ('tabpfn_auto', 'tabpfn_1')]


def gpu_owner(before, after, namespace_pids):
    """Never certify a foreign PID from total device usage or an ambiguous diff."""
    direct = set(after).intersection(namespace_pids)
    if len(direct) == 1:
        return dict(pid=next(iter(direct)), verified=True, method='namespace_pid_match')
    added = set(after) - set(before)
    if len(added) == 1:
        return dict(pid=next(iter(added)), verified=False,
                    method='unique_new_pid_after_actual_inference_not_namespace_verified')
    return dict(pid=None, verified=False, method='ambiguous_or_no_observed_pid')


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def interrupted_result(path, mode, family, status, **extra):
    """Preserve all completed traces AND the last request checkpoint on failure."""
    path = Path(path)
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    result.update(status=status, mode=mode, family=family, **extra)
    return result
