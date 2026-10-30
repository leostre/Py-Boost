"""Training adaptation of pinned cumulative_budget_projection.

Differences: parent-relative residual cap and calibration-aware local fidelity.
The original native_source file remains unchanged.
"""
from dataclasses import replace,asdict
import numpy as np

from experiments.cumulative_budget_core import (serving_cost,Quality,
    CandidateValue,Accept,Reject,decide_candidate)
from experiments.closed_budget_core import Topology,independent_snapshot
from experiments.shared_leaf_basis import functional_basis,compress
from experiments.sparse_leaf_study import base_at_rank
from experiments.sparse_leaf_core import (contrast,leaf_design,training_weights,
    coordinate_scores,score_order,make_sparse_model)
from experiments.uniform_threshold_core import ThresholdModel
from abl_training_core import training_allocations,calibrated_local_variants as threshold_variants


def training_proposal(dense,development,budget,error_state,config,*,fixed=False,criterion='local',basis_estimate=None):
    """No final-selection or test arguments exist at this boundary."""
    if set(development)!={'train','calibration','control'}:
        raise ValueError('projection accepts only train/calibration/control roles')
    m,c=dense.coefficients.shape
    topology=Topology(dense.n_trees,len(dense.nodes),m)
    designs={k:leaf_design(dense.leaf_indices(pair[0]),m) for k,pair in development.items()}
    contributions=designs['train']@dense.coefficients
    before_train=contributions+dense.bias
    if error_state.values.shape!=before_train.shape:
        raise ValueError('cumulative error object/output axis mismatch')
    if basis_estimate is None:
        mean,basis,singular=functional_basis(contributions,dense.task)
    else:
        mean=contributions.mean(axis=0)
        basis,singular=basis_estimate.basis,basis_estimate.spectrum
        expected=c-(dense.task=='multiclass')
        if (basis.shape!=(c,expected) or not np.isfinite(basis).all()
                or not np.allclose(basis.T@basis,np.eye(expected),atol=2e-5)
                or (dense.task=='multiclass' and not np.allclose(basis.sum(axis=0),0,atol=2e-5))):
            raise ValueError('invalid supplied gradient basis')
    full=compress(dense,mean,basis)
    frequency=np.asarray(designs['train'].mean(axis=0)).ravel()
    latents={k:d@full.coefficients for k,d in designs.items()}
    teacher=contrast(before_train,dense.task)
    weights=training_weights(development['train'][1],teacher,dense.task,'balanced_mse',config['margin_scale'])
    parent_parameters=m*(c-int(dense.task=="multiclass"))+2*c
    allocations=training_allocations(topology,c,basis.shape[1],budget,parent_parameters,fixed=fixed)
    full_variants=threshold_variants(designs['calibration']@dense.coefficients+dense.bias,
        development['calibration'][1],designs['control']@dense.coefficients+dense.bias,
        development['control'][1],dense.task,config['decision_grid'],config['decision_shrinkage'])
    reference=min(full_variants,key=lambda r:(-r['validation']['f1_macro'],r['validation']['loss'],r['threshold_fraction']))
    candidates,values,models=[],[],[]
    for rank in sorted({a.rank for a in allocations}):
        base=base_at_rank(dense,mean,basis,rank,full)
        residual=(contrast(dense.coefficients,dense.task)-
                  contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
        local_error=teacher-contrast(latents['train'][:,:rank]@base.basis.T+base.bias,dense.task)
        order=score_order(coordinate_scores(designs['train'],residual,local_error,'balanced_mse',weights),residual)
        seen=set()
        for allocation in (a for a in allocations if a.rank==rank):
            count=min(allocation.nnz,len(order))
            if count in seen: continue
            seen.add(count)
            leaf=make_sparse_model(base,residual,order,count,frequency)
            raw_train=leaf.raw_from_design(designs['train'])
            error=error_state.candidate_error(before_train,raw_train)
            cumulative_error=float(np.sum(weights*error**2)/c)
            local_mse=float(np.sum(weights*contrast(before_train-raw_train,dense.task)**2)/c)
            actual_cost=serving_cost(topology,c,rank,len(leaf.values))
            bare=ThresholdModel(leaf.base,leaf.rows,leaf.columns,leaf.values,np.zeros(c,np.float32))
            prepared=bare.prepare_inference()
            if (actual_cost.coo_bytes!=bare.payload_bytes or actual_cost.csr_bytes!=prepared.resident_bytes
                    or actual_cost.maximum_bytes>budget):
                raise RuntimeError('absolute cost differs from real model arrays')
            del prepared,bare
            variants=threshold_variants(leaf.raw_from_design(designs['calibration']),development['calibration'][1],
                leaf.raw_from_design(designs['control']),development['control'][1],dense.task,
                config['decision_grid'],config['decision_shrinkage'])
            for variant in variants:
                index=len(candidates)
                cid=f'r{rank}-s{len(leaf.values)}-t{variant["threshold_fraction"]}'
                row=dict(candidate_id=cid,rank=rank,nnz=len(leaf.values),cap_percent=allocation.cap_percent,
                    parameters=actual_cost.parameters,payload_bytes=actual_cost.coo_bytes,
                    csr_bytes=actual_cost.csr_bytes,max_serving_bytes=actual_cost.maximum_bytes,
                    cumulative_weighted_mse=cumulative_error,local_weighted_mse=local_mse,**variant)
                candidates.append(row)
                values.append(CandidateValue(index,cid,Quality.from_row(variant['validation']),cumulative_error,actual_cost))
                models.append((leaf,np.array(variant['decision_shifts'],np.float32)))
    decision=decide_candidate(values,Quality.from_row(reference['validation']),budget,criterion)
    information=dict(state='accepted' if isinstance(decision,Accept) else decision.reason,
        proposal_control=reference['validation'],candidates=candidates,criterion=criterion,
        allocations=[asdict(a) for a in allocations],training_error_state_bytes=error_state.values.nbytes,
        singular_values=singular.tolist() if basis_estimate is None else None,
        gradient_gram_sqrt_eigenvalues=singular.tolist() if basis_estimate is not None else None,
        basis_source='functional' if basis_estimate is None else basis_estimate.diagnostics)
    if isinstance(decision,Reject):
        return None,information,error_state
    leaf,shifts=models[decision.index]
    model=independent_snapshot(ThresholdModel(leaf.base,leaf.rows,leaf.columns,leaf.values,shifts))
    information['chosen']=candidates[decision.index]
    new_error=error_state.advance(before_train,model.raw_from_design(designs['train']))
    information['cumulative_error_rms']=float(np.sqrt(np.mean(new_error.values**2)))
    return model,information,new_error
