"""Array-only diagnostics of saved increments; no training or file access."""
from dataclasses import dataclass
from enum import Enum
import numpy as np

from abl_protected_core import append_protected, block_cost
from abl_quality_core import full_output_base
from experiments.shared_leaf_basis import compress
from experiments.sparse_leaf_core import (
    contrast, training_weights, coordinate_scores, score_order, make_sparse_model)
from experiments.uniform_threshold_core import ThresholdModel


class StopCause(str, Enum):
    ACCEPTED = 'accepted'
    MEMORY = 'memory'
    QUALITY = 'local_quality'


@dataclass(frozen=True)
class ProjectionCandidate:
    percent: int
    increment: ThresholdModel
    combined: object
    raw: dict
    cost: dict
    train_weighted_mse: float


def guard_details(candidate, reference, tolerance=.05):
    failures = []
    if candidate['loss'] > (1+tolerance)*reference['loss']:
        failures.append('loss')
    for key in ('f1_macro', 'f1_weighted'):
        if candidate[key] < (1-tolerance)*reference[key]:
            failures.append(key)
    return dict(passed=not failures, failures=failures,
        relative_loss=(candidate['loss']/reference['loss'] if reference['loss'] else None),
        relative_macro=(candidate['f1_macro']/reference['f1_macro'] if reference['f1_macro'] else None),
        relative_weighted=(candidate['f1_weighted']/reference['f1_weighted'] if reference['f1_weighted'] else None))


def stop_cause(rows, budget, reference):
    feasible = [r for r in rows if r['cost_bytes'] <= budget]
    if not feasible:
        return StopCause.MEMORY
    if any(guard_details(r['validation'], reference)['passed'] for r in feasible):
        return StopCause.ACCEPTED
    return StopCause.QUALITY


def task_quality(y, raw, task, temperature=1., shifts=None):
    if task == 'multiclass':
        from budget_quality_core import quality
        q = quality(y, raw, temperature, shifts)
        return {k: q[k] for k in ('loss', 'f1_macro', 'f1_weighted')}
    if task != 'multilabel' or temperature != 1.:
        raise ValueError('independent sigmoid outputs require temperature one')
    from experiments.shared_leaf_basis import probabilities
    y, raw = np.asarray(y), np.asarray(raw, dtype=float)
    shift = np.zeros(raw.shape[1]) if shifts is None else np.asarray(shifts)
    if y.shape != raw.shape or shift.shape != (raw.shape[1],):
        raise ValueError('label/score/shift domains differ')
    pred = raw-shift >= 0
    support = y.sum(0)
    denominator = support+pred.sum(0)
    f1 = np.divide(2*np.sum(pred*y, axis=0), denominator,
        out=np.zeros(raw.shape[1]), where=denominator > 0)
    p = probabilities(raw, task)
    return dict(loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        f1_macro=float(f1.mean()),
        f1_weighted=float(np.dot(support, f1)/max(float(support.sum()), 1.)))


def fidelity(error, target, task, weights=None):
    e, t = contrast(error, task), contrast(target, task)
    centered = t-t.mean(0)
    mse = float(np.mean(e*e))
    energy = float(np.mean(centered*centered))
    return dict(mse=mse, rmse=float(np.sqrt(mse)), max_abs=float(np.max(np.abs(e))),
        relative_centered_mse=mse/energy if energy > 0 else None,
        weighted_mse=None if weights is None else float(np.sum(weights*e*e)/e.shape[1]))


def accumulation(previous_error, local_error, measured_total, task):
    """The exact additive identity includes the shared float32 bias rounding."""
    old, local, total = (contrast(x, task) for x in (previous_error, local_error, measured_total))
    if old.shape != local.shape or total.shape != old.shape:
        raise ValueError('accumulated errors must share object/output axes')
    reconstruction = float(np.max(np.abs(total-old-local)))
    local_energy = float(np.mean(local*local))
    old_energy = float(np.mean(old*old))
    combined = float(np.mean(total*total))
    denominator = old_energy+local_energy
    cosine_denominator = np.linalg.norm(old)*np.linalg.norm(local)
    return dict(identity_max_abs=reconstruction, previous_mse=old_energy,
        local_mse=local_energy, cumulative_mse=combined,
        cross_term=float(2*np.mean(old*local)),
        amplification=combined/denominator if denominator > 0 else None,
        cosine=float(np.sum(old*local)/cosine_denominator) if cosine_denominator else None)


def projection_candidates(checkpoint, increment, designs, protected, labels,
                          basis, mean, teacher, rank, percentages=(0, 1, 2, 5)):
    """Same H1 projection/residual algebra, including candidates over the cap.

    The zero-residual comparison has identical storage for every basis. Hybrid
    comparisons share the original cap; actual nnz/cost are retained explicitly.
    """
    dense = full_output_base(increment)
    c = len(dense.bias)
    base = compress(dense, mean, basis[:, :rank])
    zero = np.zeros(c, np.float32)
    pure = ThresholdModel(base, np.empty(0, np.int32), np.empty(0, np.int32),
        np.empty(0, np.float32), zero)
    residual = (contrast(dense.coefficients, dense.task)-
        contrast(base.coefficients.astype(float)@base.basis.T, dense.task)).astype(np.float32)
    error = contrast(teacher['train']-protected['train']-
        pure.raw_from_design(designs['train']), dense.task)
    weights = training_weights(labels['train'], teacher['train'], dense.task, 'balanced_mse', .5)
    order = score_order(coordinate_scores(designs['train'], residual, error, 'balanced_mse', weights), residual)
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    seen = set()
    for percent in percentages:
        count = min(len(order), percent*increment.parameter_count//100)
        if count in seen:
            continue
        seen.add(count)
        sparse = make_sparse_model(base, residual, order, count, frequency)
        inc = ThresholdModel(sparse.base, sparse.rows, sparse.columns, sparse.values, zero)
        model = append_protected(checkpoint, inc)
        rounding = model.bias.astype(float)-checkpoint.bias.astype(float)-inc.base.bias.astype(float)
        raw = {k: protected[k]+inc.raw_from_design(d)+rounding for k, d in designs.items()}
        mse = float(np.sum(weights*contrast(teacher['train']-raw['train'], dense.task)**2)/c)
        yield ProjectionCandidate(percent, inc, model, raw, block_cost(model), mse)
