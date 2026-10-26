"""Integer byte matching for a fixed topology and fixed FP32 decoder family."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Allocation:
    storage: str
    rank: int
    nnz: int
    matching_bytes: int
    slack_bytes: int


def representation_bytes(fixed_bytes, leaves, outputs, rank, nnz, storage):
    if (any(type(v) is not int or v < 0 for v in
            (fixed_bytes, leaves, outputs, rank, nnz))
            or not leaves or not outputs or not 1 <= rank <= outputs
            or nnz > leaves * outputs or storage not in ('coo', 'csr')):
        raise ValueError('invalid byte-accounting domain')
    residual = 12 * nnz if storage == 'coo' else (8 * nnz + 4 * (leaves + 1) if nnz else 0)
    return fixed_bytes + 4 * (leaves + outputs) * rank + residual


def matched_allocations(fixed_bytes, leaves, outputs, anchor_rank, maximum_nnz, storage):
    """All lower ranks that fill the anchor bytes within one sparse entry.

    No padding, hidden dense fallback, or cap-bound underfilled pair is allowed.
    COO leaves <12 bytes unused; CSR leaves <8 bytes unused. The CSR pointer is
    charged once, only when there is a nonempty residual.
    """
    target = representation_bytes(fixed_bytes, leaves, outputs, anchor_rank, 0, storage)
    if type(maximum_nnz) is not int or not 0 <= maximum_nnz <= leaves * outputs:
        raise ValueError('invalid residual cap')
    entry_bytes = 12 if storage == 'coo' else 8
    pointer_bytes = 0 if storage == 'coo' else 4 * (leaves + 1)
    result = []
    for rank in range(1, anchor_rank):
        base = representation_bytes(fixed_bytes, leaves, outputs, rank, 0, storage)
        count = (target - base - pointer_bytes) // entry_bytes
        if not 1 <= count <= maximum_nnz:
            continue
        cost = representation_bytes(fixed_bytes, leaves, outputs, rank, count, storage)
        result.append(Allocation(storage, rank, count, cost, target - cost))
    return result


def choose_family(rows, anchor, reference_quality, percent, reference_parameters):
    """Validation-only selection, with the strong zero-residual control included."""
    from experiments.shared_leaf_basis import quality_guard
    from experiments.uniform_threshold_core import choice_key
    candidates = [anchor, *[r for r in rows if r['nnz'] * 100 <= percent * reference_parameters]]
    eligible = [r for r in candidates if quality_guard(r['validation'], reference_quality)]
    return min(eligible or candidates, key=choice_key), bool(eligible)
