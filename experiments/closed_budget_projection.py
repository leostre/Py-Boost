"""Bounded compact proposals using training geometry and a control-only guard."""
from dataclasses import asdict, replace
import numpy as np

from experiments.closed_budget_core import Topology, bounded_allocations, independent_snapshot
from experiments.shared_leaf_basis import functional_basis, compress, quality_guard
from experiments.sparse_leaf_study import base_at_rank
from experiments.sparse_leaf_core import (contrast, leaf_design, training_weights, coordinate_scores,
    score_order, make_sparse_model, Cost)
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants, choice_key


def compact_proposal(dense, train_features, train_labels, calibration_features, calibration_labels,
                     control_features, control_labels, budget, reference_parameters, config, *, fixed=False,
                     enforce_local_quality=True, prefer_local_quality=False):
    """No test or final-selection inputs exist at this decision boundary.

    Scalar per-leaf visitation weights give the same orthogonal A=VB projection.
    Sparse scores are label-balanced, not the previous Hessian-diagonal solver.
    """
    m, c = dense.coefficients.shape
    topology = Topology(dense.n_trees, len(dense.nodes), m)
    designs = {k: leaf_design(dense.leaf_indices(features), m) for k, features in
               [('train', train_features), ('calibration', calibration_features), ('control', control_features)]}
    contributions = designs['train']@dense.coefficients
    mean, basis, singular = functional_basis(contributions, dense.task)
    full = compress(dense, mean, basis)
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    latents = {k: d@full.coefficients for k, d in designs.items()}
    teacher = contrast(contributions+dense.bias, dense.task)
    weights = training_weights(train_labels, teacher, dense.task, 'balanced_mse', config['margin_scale'])
    allocations = bounded_allocations(topology, c, budget, reference_parameters, basis.shape[1], fixed=fixed)
    full_variants = threshold_variants(designs['calibration']@dense.coefficients+dense.bias, calibration_labels,
        designs['control']@dense.coefficients+dense.bias, control_labels, dense.task,
        config['decision_grid'], config['decision_shrinkage'])
    full_choice = min(full_variants, key=lambda v: (-v['validation']['f1_macro'], v['validation']['loss'],
                                                 v['threshold_fraction']))
    reference = full_choice['validation']
    rows, models, seen = [], {}, set()
    for rank in sorted({a.rank for a in allocations}):
        base = base_at_rank(dense, mean, basis, rank, full)
        residual = (contrast(dense.coefficients, dense.task)-
                    contrast(base.coefficients.astype(float)@base.basis.T, dense.task)).astype(np.float32)
        error = teacher-contrast(latents['train'][:, :rank]@base.basis.T+base.bias, dense.task)
        order = score_order(coordinate_scores(designs['train'], residual, error, 'balanced_mse', weights), residual)
        for allocation in (a for a in allocations if a.rank == rank):
            count = min(allocation.nnz, len(order))
            if (rank, count) in seen:
                continue
            seen.add((rank, count))
            leaf = make_sparse_model(base, residual, order, count, frequency)
            if not budget.allows(Cost(leaf.parameter_count+c, leaf.payload_bytes+4*c)):
                raise RuntimeError('compact allocation exceeded actual array budget')
            variants = threshold_variants(leaf.raw_from_design(designs['calibration']), calibration_labels,
                leaf.raw_from_design(designs['control']), control_labels, dense.task,
                config['decision_grid'], config['decision_shrinkage'])
            for variant in variants:
                cid = f"r{rank}-s{len(leaf.values)}-t{variant['threshold_fraction']}"
                row = dict(candidate_id=cid, rank=rank, nnz=len(leaf.values),
                           parameters=leaf.parameter_count+c, payload_bytes=leaf.payload_bytes+4*c,
                           cap_percent=allocation.cap_percent, **variant)
                rows.append(row)
                models[cid] = (leaf, np.array(variant['decision_shifts'], np.float32))
    eligible = [r for r in rows if quality_guard(r['validation'], reference)]
    diagnostic = min(rows, key=choice_key) if rows else None
    viable = eligible if enforce_local_quality else ((eligible or rows) if prefer_local_quality else rows)
    if not viable:
        return None, dict(state='no_memory_feasible' if not rows else 'local_quality_rejected',
            proposal_control=reference, diagnostic=diagnostic, candidates=rows, allocations=[asdict(a) for a in allocations])
    chosen = min(viable, key=choice_key)
    leaf, shifts = models[chosen['candidate_id']]
    model = ThresholdModel(leaf.base, leaf.rows, leaf.columns, leaf.values, shifts)
    model = independent_snapshot(model)
    return model, dict(state='accepted', proposal_control=reference, chosen=chosen, candidates=rows,
                       local_quality_enforced=bool(enforce_local_quality),
                       local_quality_pass=quality_guard(chosen['validation'], reference),
                       singular_values=singular.tolist(), allocations=[asdict(a) for a in allocations])
