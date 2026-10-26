"""Pure frozen-cohort decisions and memory accounting; no labels or fitting."""
from dataclasses import dataclass


BASE_ARRAYS = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
EXTRA_ARRAYS = ('indptr', 'columns', 'values', 'decision_shifts')


def cohort_choices(record, prefix_record, fraction=.75):
    """Exactly predeclared families; failed development choices remain labelled."""
    result = {'full': {**record['full'], 'state': 'accepted'}}
    for family in ('rank_residual', 'joint_static'):
        choices = [r for r in record['decisions'] if r['fraction'] == fraction
                   and r['family'] == family and r['cap_percent'] == 5]
        if len(choices) != 1:
            raise ValueError('missing or ambiguous frozen compact decision')
        result[family] = choices[0]
    choices = [r for r in prefix_record['choices'] if r['fraction'] == fraction
               and r['selection_scope'] == 'all_depths']
    if len(choices) != 1:
        raise ValueError('missing or ambiguous frozen native prefix decision')
    result['native_prefix'] = choices[0]
    if any(r.get('candidate') is None or not r.get('artifact') for r in result.values()):
        raise ValueError('cohort includes missing model; do not replace silently')
    return result


def executor_name(outputs):
    if not 1 <= outputs <= 1024:
        raise ValueError('no validated GPU executor for this output count')
    return 'groups' if outputs <= 128 else 'unified_wide'


@dataclass(frozen=True)
class MemoryItem:
    name: str
    pointer: int
    size: int


def owned_bytes(items):
    """Reject overlapping live arrays, not just identical Python wrappers."""
    if any(i.pointer < 0 or i.size < 0 for i in items):
        raise ValueError('negative address or size')
    nonempty = sorted((i for i in items if i.size), key=lambda i: i.pointer)
    for left, right in zip(nonempty, nonempty[1:]):
        if left.pointer+left.size > right.pointer:
            raise ValueError('overlapping resident ownership: '+left.name+' / '+right.name)
    return sum(i.size for i in items)


def gpu_items(runtime):
    result = []
    for name in (*BASE_ARRAYS, *EXTRA_ARRAYS):
        array = getattr(runtime, '_'+name)
        result.append(MemoryItem(name, int(array.data.ptr), int(array.nbytes)))
    return result


def cpu_items(snapshot):
    return [MemoryItem(n, int(a.ctypes.data), int(a.nbytes)) for n, a in
            [(n, getattr(snapshot.base, n)) for n in BASE_ARRAYS]
            + [(n, getattr(snapshot, n)) for n in EXTRA_ARRAYS]]


def capacity_prefix(rows, budget, shared=0):
    """Technical fixed-order resident frontier, NOT arbitrary subset selection."""
    if budget < 0 or shared < 0:
        raise ValueError('negative memory budget')
    total, accepted = shared, 0
    for row in rows:
        size = row['resident_bytes']
        if size < 0:
            raise ValueError('negative resident size')
        if not row['development_pass'] or total+size > budget:
            break
        total += size
        accepted += 1
    return dict(count=accepted, bytes=total, model_versions=True)


def latency_summary(milliseconds):
    import numpy as np
    values = np.asarray(milliseconds, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid request timings')
    return dict(p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), requests=len(values),
                qps=float(1000*len(values)/sum(values)) if sum(values) else None)


def task_order(records):
    """Round-robin seed position: first six slots cover all six datasets."""
    groups = {}
    for record in records:
        groups.setdefault(record['dataset'], []).append(record)
    ordered = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        ordered.extend(g[i] for g in groups.values() if i < len(g))
    return ordered


def traffic_schedule(tasks, count, kind, seed):
    import numpy as np
    if tasks < 1 or count < tasks or kind not in ('uniform','skew'):
        raise ValueError('invalid traffic specification')
    rng = np.random.default_rng(seed)
    # Every stored task is exercised even under skew; this is a closed-loop trace.
    first = list(range(tasks))
    if kind == 'uniform':
        tail = np.arange(count-tasks) % tasks
        rng.shuffle(tail)
    else:
        weights = 1/np.arange(1,tasks+1,dtype=float)
        tail = rng.choice(tasks, count-tasks, p=weights/weights.sum())
    return first+list(map(int,tail))
