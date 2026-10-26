"""Off-policy diagnostic: equal controller revelations, NOT online optimization."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.cumulative_budget_audit import locked
from experiments.cumulative_budget_core import serving_cost
from experiments.closed_budget_core import Topology
from experiments.closed_budget_study import combine_actual
from experiments.cumulative_budget_study import initial_prior
from experiments.absolute_history_core import diagonal_derivatives
from experiments.gradient_basis_core import gradient_signal
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.uniform_threshold_core import ThresholdModel
from experiments.history_basis_study import raw
from experiments.shared_leaf_basis import quality_guard
from experiments.output_calibration_core import calibrated_quality
from experiments.held_budget_event_core import spectral_view,audit_decision,hold_budget


def held_budget_event_study(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_held_budget_event.py'],cwd=ROOT,check=True,timeout=300)
    locks={}
    p=locked(config['event_development_manifest'],config['event_development_sha256'],locks)
    rp=locked(config['event_roles_manifest'],config['event_roles_sha256'],locks)
    source=json.loads(p.read_text(encoding='utf-8'))
    manifest=json.loads(rp.read_text(encoding='utf-8'))
    results,excluded=[],[]
    with threadpool_limits(limits=1):
        for record in source['records']:
            trajectory=next(t for t in record['trajectories'] if t['method']=='cumulative_reserve')
            cycles=[c for c in trajectory['cycles'] if c.get('artifact')]
            if len(cycles)!=5:
                excluded.append(dict(dataset=record['dataset'],seed=record['seed'],accepted_states=len(cycles),
                    reason='fewer_than_five_saved_states;cannot_compare_three_revelations'))
                continue
            case=next(c for c in manifest['cases'] if c['dataset']==record['dataset'] and c['seed']==record['seed'])
            entry=manifest['datasets'][case['dataset']]
            locked(entry['path'],entry['sha256'],locks)
            locked(case['split_file'],case['split_sha256'],locks)
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            tx,ty=x[roles['train']],y[roles['train']]
            cx,cy=x[roles['control']],y[roles['control']]
            prior=initial_prior(ty,case['outputs'],case['task'])
            actual_checkpoint=None
            states=[]
            for cycle in cycles:
                fit=cycle['fit']
                inc=ThresholdModel.load(locked(fit['artifact'],fit['sha256'],locks))
                proposal=combine_actual(actual_checkpoint,inc,prior)
                z=raw(proposal,tx)
                g,h=diagonal_derivatives(ty,z,case['task'])
                signal=gradient_signal(g,h,config['lambda_l2'],case['task'])
                state,noise=spectral_view(signal,case['seed']+95000)
                states.append(dict(proposal=proposal,state=state,noise=noise,cycle=cycle))
                actual_checkpoint=ThresholdModel.load(locked(cycle['artifact'],cycle['sha256'],locks))
            for policy in ('periodic','event'):
                confirmed,confirmed_state=None,None
                middle=False
                rows=[]
                for index,state in enumerate(states):
                    dense=state['proposal'].base
                    topology=Topology(dense.n_trees,len(dense.nodes),len(dense.coefficients))
                    drift=0. if confirmed_state is None else float(np.linalg.norm(state['state']-confirmed_state,'fro'))
                    ratio=drift/max(state['noise'],1e-12)
                    blocked=(confirmed is not None and serving_cost(topology,case['outputs'],
                        confirmed.base.basis.shape[1],len(confirmed.values)).maximum_bytes>case['absolute_budget_bytes'])
                    # Action is fixed before any control-oracle label score.
                    audit,reason=audit_decision(index,policy,middle,ratio,blocked)
                    row=dict(trees=(index+1)*40,audit=audit,reason=reason,ratio=ratio,temporal_drift=drift,
                        simultaneous_noise=state['noise'],held_budget_blocked=bool(blocked),oracle_unsafe=False,
                        unsafe_applied=False,safe_applied=False)
                    if confirmed is not None and not blocked:
                        held=hold_budget(dense,tx,ty,confirmed,config['margin_scale'])
                        if max(held.payload_bytes,held.prepare_inference().resident_bytes)>case['absolute_budget_bytes']:
                            raise RuntimeError('held projection violated hard budget')
                        # Deliberately unshifted logits: diagnose representation,
                        # not threshold retuning. Does not replace final quality.
                        zero=np.zeros(case['outputs'],np.float32)
                        reference=calibrated_quality(cy,raw(state['proposal'],cx),case['task'],zero)
                        q=calibrated_quality(cy,raw(held,cx),case['task'],zero)
                        unsafe=not quality_guard(q,reference)
                        row.update(oracle_unsafe=unsafe,oracle_quality=q,oracle_reference=reference,
                            unsafe_applied=bool(unsafe and not audit),safe_applied=bool(not unsafe and not audit))
                    if audit:
                        cycle=state['cycle']
                        confirmed=ThresholdModel.load(cycle['artifact'])
                        confirmed_state=state['state'].copy()
                        if index in (1,2,3): middle=True
                    rows.append(row)
                if sum(r['audit'] for r in rows)!=3: raise RuntimeError('unequal warning audit budget')
                results.append(dict(dataset=case['dataset'],seed=case['seed'],policy=policy,rows=rows,
                    decision_audits=3,unsafe_applied=sum(r['unsafe_applied'] for r in rows),
                    safe_applied=sum(r['safe_applied'] for r in rows),
                    blocked_applications=sum(r['held_budget_blocked'] and not r['audit'] for r in rows),
                    audit_iterations=[r['trees'] for r in rows if r['audit']]))
            emit('held_budget_warning_case',dataset=case['dataset'],seed=case['seed'],
                periodic_unsafe=results[-2]['unsafe_applied'],event_unsafe=results[-1]['unsafe_applied'],new_trees=0)
    groups=[]
    for name in sorted({r['dataset'] for r in results}):
        a=[r for r in results if r['dataset']==name and r['policy']=='periodic']
        b=[r for r in results if r['dataset']==name and r['policy']=='event']
        old=sum(r['unsafe_applied'] for r in a)
        new=sum(r['unsafe_applied'] for r in b)
        safe_old=sum(r['safe_applied'] for r in a)
        safe_new=sum(r['safe_applied'] for r in b)
        blocked_old=sum(r['blocked_applications'] for r in a)
        blocked_new=sum(r['blocked_applications'] for r in b)
        passed=bool(old>0 and new<=.8*old and safe_new>=.9*safe_old and blocked_new<=blocked_old)
        groups.append(dict(dataset=name,cases=len(a),periodic_unsafe=old,event_unsafe=new,
            periodic_safe=safe_old,event_safe=safe_new,periodic_blocked=blocked_old,event_blocked=blocked_new,online_gate=passed))
    if any(sha256(path)!=value for path,value in locks.items()): raise RuntimeError('warning sources changed')
    report.update(status='complete',results=results,groups=groups,excluded=excluded,input_locks=locks,new_fits=0,new_trees=0,
        online_gate_passed=sum(g['online_gate'] for g in groups)>=2,
        interpretation='OFFPOLICY warning diagnostic only; source gradients came from full controller every40trees',
        oracle_definition='unshifted control loss/macro/weighted relative to current proposal,5percent;not final calibrated test',
        equal_budget='3causal controller revelations per policy; all oracle calculations charged as offline diagnostics',
        no_inference_speed_claim=True,test_labels_used=False)
    emit('held_budget_warning_complete',groups=groups,excluded_cases=len(excluded),online_gate=report['online_gate_passed'],new_trees=0)
