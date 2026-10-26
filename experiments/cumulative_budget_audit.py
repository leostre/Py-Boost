"""Audit saved actual feedback and immutable choices, then score preserved test."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT,emit,sha256,write_report
from experiments.cumulative_budget_core import (serving_cost,CandidateValue,Quality,Accept,Reject,decide_candidate)
from experiments.closed_budget_core import Topology
from experiments.shared_leaf_basis import LeafEnsemble,quality_guard
from experiments.uniform_threshold_core import ThresholdModel,choice_key
from experiments.output_calibration_core import calibrated_quality
from experiments.sparse_leaf_core import contrast,training_weights
from experiments.history_basis_study import raw
from experiments.leaf_budget_runtime import load_expanded,load_roles


def locked(path,digest,locks):
    p=Path(path)
    if sha256(p)!=digest: raise RuntimeError('frozen input changed: '+str(p))
    locks[str(p)]=digest
    return p


def quality_matches(actual,expected):
    if any(abs(actual[k]-expected[k])>1e-7 for k in actual):
        raise RuntimeError('frozen development quality does not replay')


def cohort_keys(records):
    keys=[(r['dataset'],r['seed']) for r in records]
    if len(set(keys))!=len(keys): raise RuntimeError('duplicate development case identity')
    return set(keys)


def record_tree_cost(record):
    return (sum(c['fit']['trees'] for t in record['trajectories'] for c in t['cycles'] if 'fit' in c)
            +sum(f['trees'] for f in record['full_fits']))


def verify_reused_records(source,locks):
    cohort_keys(source['records'])
    reused=[r for r in source['records'] if r.get('reused') is True]
    cost=sum(record_tree_cost(r) for r in reused)
    if cost!=source.get('reused_tree_fits',0): raise RuntimeError('reused training price changed')
    if reused:
        path=locked(source['reused_records_manifest'],source['reused_records_sha256'],locks)
        parent=json.loads(path.read_text(encoding='utf-8'))
        cohort_keys(parent['records'])
        for record in reused:
            original=next((r for r in parent['records'] if r['dataset']==record['dataset'] and r['seed']==record['seed']),None)
            if {k:v for k,v in record.items() if k!='reused'}!=original:
                raise RuntimeError('reused immutable record changed')
    new=sum(record_tree_cost(r) for r in source['records'] if r.get('reused') is not True)
    if new!=source['tree_fits']: raise RuntimeError('new training price changed')
    return new,cost


def audit_cumulative(config,output):
    from experiments.cumulative_budget_study import initial_prior,model_bytes
    from experiments.closed_budget_study import combine_actual
    locks={}
    p=locked(config['cumulative_development_manifest'],config['cumulative_development_manifest_sha256'],locks)
    source=json.loads(p.read_text(encoding='utf-8'))
    rp=locked(config['cumulative_roles_manifest'],config['cumulative_roles_manifest_sha256'],locks)
    roles_source=json.loads(rp.read_text(encoding='utf-8'))
    if cohort_keys(source['records'])!=cohort_keys(roles_source['cases']) or not source['records']:
        raise RuntimeError('incomplete development cohort')
    new_tree_fits,reused_tree_fits=verify_reused_records(source,locks)
    if source.get('source_controls_manifest'):
        control_path=locked(source['source_controls_manifest'],source['source_controls_sha256'],locks)
        controls=json.loads(control_path.read_text(encoding='utf-8'))
        cohort_keys(controls['records'])
        for record in source['records']:
            old=next(r for r in controls['records'] if r['dataset']==record['dataset'] and r['seed']==record['seed'])
            old_methods={d['method'] for d in old['decisions']}
            if (record['full']!=old['full'] or record['reference_validation']!=old['reference_validation']
                    or [d for d in record['decisions'] if d['method'] in old_methods]!=old['decisions']):
                raise RuntimeError('reused baseline controls changed')
    total_trees=0
    checkpoints=0
    maximum_error=0.
    casting_drift=0.
    history_observations=0
    for record in source['records']:
        case=next(c for c in roles_source['cases'] if c['dataset']==record['dataset'] and c['seed']==record['seed'])
        data=roles_source['datasets'][case['dataset']]
        locked(data['path'],data['sha256'],locks)
        x,y=load_expanded(case['dataset'],roles_source)
        roles=load_roles(case,len(x))
        locked(case['split_file'],case['split_sha256'],locks)
        original=load_roles(dict(split_file=case['source_split_file'],split_sha256=case['source_split_sha256']),len(x))
        locked(case['source_split_file'],case['source_split_sha256'],locks)
        for name in ('calibration','test'):
            np.testing.assert_array_equal(roles[name],original[name])
        train_x,train_y=x[roles['train']],y[roles['train']]
        prior=initial_prior(train_y,case['outputs'],case['task'])
        for trajectory in record['trajectories']:
            history=None
            if trajectory['method'].startswith('gh_'):
                from experiments.absolute_history_core import AbsoluteHistory
                history=AbsoluteHistory(trajectory['method'][3:],case['outputs'],case['task'],config['history_beta'])
            checkpoint=None
            cumulative=np.zeros((len(train_x),case['outputs']))
            casting=np.zeros_like(cumulative)
            shadow=np.broadcast_to(prior,cumulative.shape).astype(float).copy()
            accepted=0
            for cycle in trajectory['cycles']:
                if 'fit' not in cycle: continue
                fit=cycle['fit']
                total_trees+=fit['trees']
                ip=locked(fit['artifact'],fit['sha256'],locks)
                increment=ThresholdModel.load(ip)
                init=locked(ip.parent/'initial.npz',fit['initial_sha256'],locks)
                worker=locked(ip.parent.parent/'worker_data.npz',fit['data_sha256'],locks)
                with np.load(worker,allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['train_x'],train_x)
                    np.testing.assert_array_equal(f['train_y'],train_y)
                    np.testing.assert_array_equal(f['validation_x'],x[roles['calibration']])
                    np.testing.assert_array_equal(f['validation_y'],y[roles['calibration']])
                expected=np.broadcast_to(prior,cumulative.shape).copy() if checkpoint is None else raw(checkpoint,train_x).astype(np.float32)
                with np.load(init,allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['train'],expected)
                    expected_cal=(np.broadcast_to(prior,(len(roles['calibration']),case['outputs'])).copy()
                                  if checkpoint is None else raw(checkpoint,x[roles['calibration']]))
                    np.testing.assert_array_equal(f['validation'],expected_cal)
                proposal=combine_actual(checkpoint,increment,prior)
                before=raw(proposal,train_x)
                increment_raw=raw(increment,train_x)
                previous_raw=(np.broadcast_to(prior,cumulative.shape) if checkpoint is None
                              else raw(checkpoint,train_x))
                # combine_actual materializes FP32 tree values. Separate this
                # conversion drift from the ideal mathematical concatenation.
                casting+=contrast(before-previous_raw-increment_raw,case['task'])
                casting_drift=max(casting_drift,float(np.max(np.abs(casting))))
                shadow+=increment_raw
                topology=Topology(proposal.base.n_trees,len(proposal.base.nodes),len(proposal.base.coefficients))
                information=cycle['projection']
                estimate=None
                if history is not None:
                    rank=max(1,(case['outputs']-(case['task']=='multiclass'))//2)
                    estimate=history.observe(train_y,before,config['lambda_l2'],rank)
                    if set(estimate.diagnostics)!=set(information['basis_source']):
                        raise RuntimeError('gradient basis diagnostic fields changed')
                    for key,value in estimate.diagnostics.items():
                        recorded=information['basis_source'][key]
                        if isinstance(value,float):
                            if recorded is None or abs(value-recorded)>1e-9:
                                raise RuntimeError('gradient basis history does not replay: '+key)
                        elif value!=recorded:
                            raise RuntimeError('gradient basis state does not replay: '+key)
                    np.testing.assert_allclose(estimate.spectrum,information['gradient_gram_sqrt_eigenvalues'],atol=1e-10,rtol=1e-9)
                    history_observations+=1
                candidates=[]
                for index,row in enumerate(information['candidates']):
                    cost=serving_cost(topology,case['outputs'],row['rank'],row['nnz'])
                    if 'reserved_horizon' in information:
                        from experiments.closed_budget_core import extend_bound
                        future=extend_bound(topology,information['reserved_horizon']-topology.trees,3)
                        if asdict(future)!=information['allocation_topology']:
                            raise RuntimeError('future topology reserve changed')
                        reserved=serving_cost(future,case['outputs'],row['rank'],row['nnz']).maximum_bytes
                        if reserved!=row['reserved_max_serving_bytes'] or reserved>case['absolute_budget_bytes']:
                            raise RuntimeError('candidate future reserve does not replay')
                    if (row['parameters']!=cost.parameters or row['payload_bytes']!=cost.coo_bytes
                            or row['csr_bytes']!=cost.csr_bytes or cost.maximum_bytes>case['absolute_budget_bytes']):
                        raise RuntimeError('frozen candidate cost invalid')
                    candidates.append(CandidateValue(index,row['candidate_id'],Quality.from_row(row['validation']),
                                                     row['cumulative_weighted_mse'],cost))
                decision=decide_candidate(candidates,Quality.from_row(information['proposal_control']),
                                           case['absolute_budget_bytes'],information['criterion'])
                if isinstance(decision,Reject):
                    if cycle.get('artifact') or information['state']!=decision.reason:
                        raise RuntimeError('rejected proposal retained a model')
                    continue
                chosen=information['candidates'][decision.index]
                if chosen!=information['chosen']: raise RuntimeError('frozen ranking does not replay')
                mp=locked(cycle['artifact'],cycle['sha256'],locks)
                model=ThresholdModel.load(mp)
                if estimate is not None:
                    np.testing.assert_allclose(model.base.basis,estimate.basis[:,:chosen['rank']],atol=2e-6,rtol=1e-6)
                for name in ('nodes','node_offsets','leaf_offsets'):
                    np.testing.assert_array_equal(getattr(model.base,name),getattr(proposal.base,name))
                if model_bytes(model)>case['absolute_budget_bytes'] or model.parameter_count!=chosen['parameters']:
                    raise RuntimeError('accepted model violated absolute cost')
                np.testing.assert_array_equal(model.decision_shifts,np.array(chosen['decision_shifts'],np.float32))
                q=calibrated_quality(y[roles['control']],raw(model,x[roles['control']]),case['task'],model.decision_shifts)
                quality_matches(q,chosen['validation'])
                cumulative+=contrast(before-raw(model,train_x),case['task'])
                direct=contrast(shadow-raw(model,train_x),case['task'])+casting
                error_path=mp.parent/'training_error.npz'
                locks[str(error_path)]=sha256(error_path)
                with np.load(error_path,allow_pickle=False) as f:
                    error=float(np.max(np.abs(f['error']-cumulative)))
                    direct_error=float(np.max(np.abs(f['error']-direct)))
                    if max(error,direct_error)>1e-10: raise RuntimeError('cumulative-error identity does not replay')
                    assert int(f['updates'])==accepted+1
                maximum_error=max(maximum_error,error,direct_error)
                weights=training_weights(train_y,contrast(before,case['task']),case['task'],'balanced_mse',config['margin_scale'])
                value=float(np.sum(weights*cumulative**2)/case['outputs'])
                if abs(value-chosen['cumulative_weighted_mse'])>1e-10:
                    raise RuntimeError('chosen cumulative surrogate does not replay')
                checkpoint=model
                checkpoints+=1
                accepted+=1
            if accepted!=trajectory['accepted_checkpoints']: raise RuntimeError('missing trajectory snapshot')
        for fit in record['full_fits']:
            total_trees+=fit['trees']
            locked(fit['artifact'],fit['sha256'],locks)
        full=ThresholdModel.load(locked(record['full']['artifact'],record['full']['sha256'],locks))
        reference=calibrated_quality(y[roles['selection']],raw(full,x[roles['selection']]),case['task'],full.decision_shifts)
        quality_matches(reference,record['reference_validation'])
        for d in record['decisions']:
            if d['candidate'] is None: continue
            model=ThresholdModel.load(locked(d['artifact'],d['sha256'],locks))
            quality=calibrated_quality(y[roles['selection']],raw(model,x[roles['selection']]),case['task'],model.decision_shifts)
            quality_matches(quality,d['candidate']['validation'])
            np.testing.assert_array_equal(model.decision_shifts,np.array(d['candidate']['decision_shifts'],np.float32))
            if model_bytes(model)>case['absolute_budget_bytes']: raise RuntimeError('final model over budget')
            allowed=[r for r in d['candidates'] if quality_guard(r['validation'],record['reference_validation'])]
            # Replay ranking from the validated frozen numeric table, not re-rounded scores.
            chosen=min(allowed or d['candidates'],key=choice_key)
            if chosen!=d['candidate'] or (d['state']=='accepted')!=bool(allowed):
                raise RuntimeError('final frozen selection changed')
        emit('cumulative_case_audited',dataset=case['dataset'],seed=case['seed'],test_evaluated=False)
    if total_trees!=new_tree_fits+reused_tree_fits or new_tree_fits>config['cumulative_pilot_tree_cap']:
        raise RuntimeError('tree ledger failed independent replay')
    result=dict(status='audit_passed',file_locks=locks,checkpoints=checkpoints,trees=total_trees,
        source_new_tree_fits=new_tree_fits,source_reused_tree_fits=reused_tree_fits,
        gradient_history_observations_replayed=history_observations,
        cumulative_error_max_abs_error=maximum_error,test_evaluated=False,new_fits=0,
        accumulated_fp32_materialization_drift_max_abs=casting_drift,
        ranking_policy='validated frozen numeric rows; never reselect under BLAS rounding',
        limitation='unselected candidate qualities replayed by frozen tables, not by refitting all projections')
    write_report(output/'pretest_audit.json',result)
    emit('cumulative_pretest_audit_passed',checkpoints=checkpoints,trees=total_trees,
         cumulative_error_max_abs_error=maximum_error,files=len(locks),test_evaluated=False,
         accumulated_fp32_materialization_drift_max_abs=casting_drift,
         seal_sha256=sha256(output/'pretest_audit.json'))
    return source,roles_source,result


def cumulative_final_test(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_cumulative_budget_audit.py'],
                   cwd=ROOT,check=True,timeout=600)
    with threadpool_limits(limits=1):
        source,roles_source,audit=audit_cumulative(config,output)
        results=[]
        for record in source['records']:
            case=next(c for c in roles_source['cases'] if c['dataset']==record['dataset'] and c['seed']==record['seed'])
            x,y=load_expanded(case['dataset'],roles_source)
            roles=load_roles(case,len(x))
            full=ThresholdModel.load(record['full']['artifact'])
            reference=calibrated_quality(y[roles['test']],raw(full,x[roles['test']]),case['task'],full.decision_shifts)
            for decision in record['decisions']:
                row=dict(dataset=case['dataset'],seed=case['seed'],method=decision['method'],state=decision['state'],
                         reference=reference,budget_bytes=case['absolute_budget_bytes'])
                if decision['candidate'] is None:
                    row.update(test_quality_pass=False,joint_success=False,quality=None)
                else:
                    model=ThresholdModel.load(decision['artifact'])
                    q=calibrated_quality(y[roles['test']],raw(model,x[roles['test']]),case['task'],model.decision_shifts)
                    memory=max(model.payload_bytes,model.prepare_inference().resident_bytes)
                    passed=quality_guard(q,reference)
                    row.update(quality=q,test_quality_pass=passed,coo_bytes=model.payload_bytes,
                        csr_bytes=model.prepare_inference().resident_bytes,max_serving_bytes=memory,
                        parameters=model.parameter_count,trees=model.base.n_trees,rank=model.base.coefficients.shape[1],
                        nnz=len(model.values),leaves=len(model.base.coefficients),artifact=decision['artifact'],sha256=decision['sha256'],
                        loss_ratio=q['loss']/reference['loss'],macro_retained=q['f1_macro']/reference['f1_macro'],
                        weighted_retained=q['f1_weighted']/reference['f1_weighted'],
                        joint_success=bool(decision['state']=='accepted' and passed and memory<=case['absolute_budget_bytes']))
                results.append(row)
                emit('cumulative_frozen_test_result',**row)
        for p,digest in audit['file_locks'].items():
            if sha256(p)!=digest: raise RuntimeError('input changed during test')
    methods=sorted({r['method'] for r in results})
    summaries=[dict(method=m,cases=sum(r['method']==m for r in results),
                    successes=sum(r['joint_success'] for r in results if r['method']==m)) for m in methods]
    report.update(status='complete',results=results,summaries=summaries,audit=audit,new_fits=0,tree_fits=0,
        test_evaluated=True,test_is_reused_exploratory=True,pretest_audit_sha256=sha256(output/'pretest_audit.json'))
    emit('cumulative_frozen_test_complete',summaries=summaries,new_fits=0,
         cumulative_error_max_abs_error=audit['cumulative_error_max_abs_error'])
