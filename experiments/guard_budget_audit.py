"""Independent replay of guard state, feedback, costs, then sealed test scoring."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.guard_budget_study import locked_json, same_arrays
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants
from experiments.output_calibration_core import calibrated_quality, fit_decision_shifts, labels_from_raw
from experiments.joint_budget_audit import guard, array_cost, raw, verify_model, csv_table
from experiments.closed_budget_study import combine_actual, empty_model
from experiments.parallel_leaf_core import prefix
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.label_budget_diagnostics import compare_labels


def independent_guard_step(policy, block, confirmed, failures, local_pass, available=True):
    """Reference oracle deliberately does not import the production reducer."""
    if not available:
        return 'rollback', confirmed, failures, 'resource'
    if local_pass:
        return 'confirmed', block, 0, None
    if (policy == 'warmup2' and block < 2) or (policy == 'confirm2' and failures == 0):
        return 'provisional', confirmed, failures+1, None
    if policy not in ('strict', 'warmup2', 'confirm2'):
        raise ValueError('unknown audit guard')
    return 'rollback', confirmed, failures, 'quality'


def best_row(rows, reference):
    eligible = [r for r in rows if guard(r['validation'], reference)]
    return (min(eligible or rows, key=lambda r:(-r['validation']['f1_macro'],r['validation']['loss'],
                r['payload_bytes'],r['parameters'],r['candidate_id'])) if rows else None), bool(eligible)


def equal_quality(actual, expected):
    if set(actual) != set(expected) or any(abs(actual[k]-expected[k]) > 1e-7 for k in actual):
        raise RuntimeError('independent guard metric replay mismatch')


def independent_depth(topology, remaining, outputs, rank, nnz, budget):
    t0,i0,m0 = topology
    for depth in (5,4,3):
        t,i,m = t0+remaining,i0+remaining*(2**depth-1),m0+remaining*2**depth
        s = min(nnz,m*outputs)
        p = (m+outputs)*rank+s+2*outputs
        b = 16*i+16*(t+1)+4*(m+outputs)*rank+12*s+8*outputs
        if p <= budget['parameters'] and b <= budget['payload_bytes']:
            return depth
    return None


def check_candidate_cost(row, proposal, budget, reference_parameters):
    m,c=proposal.base.coefficients.shape
    t,i,r,s=proposal.base.n_trees,len(proposal.base.nodes),row['rank'],row['nnz']
    p,b=(m+c)*r+s+2*c,16*i+16*(t+1)+4*(m+c)*r+12*s+8*c
    if (p,b)!=(row['parameters'],row['payload_bytes']):
        raise RuntimeError('independent guard candidate array formula differs')
    if p>budget['parameters'] or b>budget['payload_bytes'] or s*100>5*reference_parameters:
        raise RuntimeError('guard candidate exceeds immutable budget')


def verify_final_choice(record, snapshots, x, y, roles, reference, root, config, stats):
    rows=[]
    for block, model in snapshots:
        variants=threshold_variants(raw(model,x[roles['calibration']]),y[roles['calibration']],
            raw(model,x[roles['selection']]),y[roles['selection']],model.base.task,
            config['decision_grid'],config['decision_shrinkage'])
        for v in variants:
            rows.append(dict(candidate_id=f"block{block}-t{v['threshold_fraction']}",block=block,
                parameters=model.parameter_count,payload_bytes=model.payload_bytes,rank=model.base.basis.shape[1],
                nnz=len(model.values),trees=model.base.n_trees,leaves=len(model.base.coefficients),
                internal_nodes=len(model.base.nodes),**v))
    choice=record['choice']
    if not rows:
        if choice['candidate'] is not None or choice['state']!='no_accepted_checkpoint':
            raise RuntimeError('nonexistent final checkpoint reported as a model')
        return None
    if len(rows)!=len(choice['candidates']):
        raise RuntimeError('rollback did not remove the correct final candidates')
    for actual,expected in zip(rows,choice['candidates']):
        equal_quality(actual['validation'],expected['validation'])
        if {k:v for k,v in actual.items() if k!='validation'} != {k:v for k,v in expected.items() if k!='validation'}:
            raise RuntimeError('final guard candidate metadata differs')
    # Metrics above are independently checked numerically. Replay the frozen
    # decision at its original precision rather than making a new local choice:
    # BLAS/platform rounding may change loss by one ULP without changing labels.
    chosen,accepted=best_row(choice['candidates'],reference)
    if chosen!=choice['candidate'] or choice['state']!=('accepted' if accepted else 'no_quality_feasible'):
        raise RuntimeError('guard final selection differs: '+json.dumps(dict(
            actual_id=chosen['candidate_id'],expected_id=choice['candidate']['candidate_id'],
            actual_quality=chosen['validation'],expected_quality=choice['candidate']['validation'],
            expected_state=choice['state'],actual_state='accepted' if accepted else 'no_quality_feasible')))
    cache={}
    verified=verify_model(choice,root,x,y,roles,config,cache,stats)
    source=next(m for block,m in snapshots if block==chosen['block'])
    same_arrays(replace(source,decision_shifts=np.array(chosen['decision_shifts'],np.float32)),verified['model'])
    return dict(artifact=str(root/choice['artifact']),sha256=choice['sha256'],state=choice['state'],
                parameters=verified['parameters'],payload_bytes=verified['payload_bytes'],csr_bytes=verified['csr_bytes'],
                candidate=chosen)


def verify_trajectory(entry, full_record, static_root, x, y, roles, config, stats):
    from experiments.closed_budget_projection import compact_proposal
    from experiments.sparse_leaf_core import Budget
    record,root=entry['trajectory'],Path(entry['artifact_root'])
    policy,allocation=entry['guard_policy'],entry['allocation_policy']
    if entry.get('reuse_reason')=='all_pass_equivalence':
        # Replay the actual strict source first; its all-pass proof is checked
        # below with independently recomputed control metrics.
        policy='strict'
    if record.get('guard_policy') is not None and record['guard_policy']!=policy:
        raise RuntimeError('physical guard policy differs without an equivalence proof')
    case_id=f"{record['dataset']}_{record['seed']}"
    directory=root/case_id/record['method']
    stored=json.loads((directory/'trajectory.json').read_text())
    if stored!=record or record['test_evaluated']:
        raise RuntimeError('trajectory differs from its sealed source manifest')
    budget=record['budget'];reference=full_record['reference_cost']
    if (budget['parameters'],budget['payload_bytes'])!=(int(reference['parameters']*.75),int(reference['payload_bytes']*.75)):
        raise RuntimeError('guard comparison changed the joint 75percent cap')
    full_fit=next(f for f in full_record['fits'] if f['depth']==5)
    dense=LeafEnsemble.load(static_root/full_fit['artifact'])
    prior=dense.bias.copy();c=len(prior);ceiling=c-1 if dense.task=='multiclass' else c
    fixed_depth=independent_depth((0,0,0),200,c,max(1,ceiling//2),5*reference['parameters']//100,budget)
    if fixed_depth!=record['fixed_depth']:
        raise RuntimeError('fixed-depth reservation changed')
    worker_path=directory/'worker_data.npz'
    with np.load(worker_path,allow_pickle=False) as f:
        for key,expected in [('train_x',x[roles['train']]),('train_y',y[roles['train']]),
                             ('validation_x',x[roles['calibration']]),('validation_y',y[roles['calibration']])]:
            np.testing.assert_array_equal(f[key],expected)
    checkpoint=None;snapshots=[];confirmed=-1;failures=0;all_pass=True
    built=0;stopped=False
    for block,cycle in enumerate(record['cycles']):
        if stopped or cycle['block']!=block:
            raise RuntimeError('events after stopping or unordered blocks')
        topology=(0,0,0) if checkpoint is None else (checkpoint.base.n_trees,len(checkpoint.base.nodes),len(checkpoint.base.coefficients))
        minimum=max(1,int(ceiling*.25 if checkpoint is None else checkpoint.base.basis.shape[1]*.75))
        depth=fixed_depth if allocation=='periodic_fixed' else independent_depth(topology,200-topology[0],c,minimum,0,budget)
        if 'fit' not in cycle:
            if depth is not None or cycle['state']!='no_depth_with_final_horizon_reserve':
                raise RuntimeError('spurious structural-budget stop')
            snapshots=[(b,m) for b,m in snapshots if b<=confirmed]
            all_pass=False;stopped=True
            break
        if cycle['proposed_depth']!=depth or cycle['accepted_trees_before']!=topology[0]:
            raise RuntimeError('guard path changed its allocation rule')
        cycle_dir=directory/f'block{block}';fit=cycle['fit']
        if fit['seed']!=record['seed']+40000+40*block or fit['trees']!=40 or fit['lr']!=config['lr']:
            raise RuntimeError('native continuation command/RNG contract changed')
        for path,expected in [(worker_path,fit['data_sha256']),(cycle_dir/'initial.npz',cycle['initial_sha256']),
                              (cycle_dir/'increment.npz',fit['sha256'])]:
            if sha256(path)!=expected:
                raise RuntimeError('continuation input or increment changed')
        if fit['initial_sha256']!=cycle['initial_sha256']:
            raise RuntimeError('fit consumed a different initial checkpoint')
        expected_train=(np.broadcast_to(prior,(len(roles['train']),c)).copy() if checkpoint is None else
                        raw(checkpoint,x[roles['train']]).astype(np.float32))
        expected_cal=(np.broadcast_to(prior,(len(roles['calibration']),c)).copy() if checkpoint is None else
                      raw(checkpoint,x[roles['calibration']]))
        with np.load(cycle_dir/'initial.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['train'],expected_train)
            np.testing.assert_array_equal(f['validation'],expected_cal)
        increment=ThresholdModel.load(cycle_dir/'increment.npz');built+=40
        proposal=combine_actual(checkpoint,increment,prior)
        if proposal.payload_bytes!=cycle['proposal_full_bytes']:
            raise RuntimeError('temporary proposal cost metadata differs')
        if block==0:
            control_fit=next(f for f in full_record['fits'] if f['depth']==depth)
            control=prefix(empty_model(LeafEnsemble.load(static_root/control_fit['artifact'])),40)
            probes=x[roles['control'][:min(32,len(roles['control']))]]
            np.testing.assert_allclose(proposal.predict(probes),control.predict(probes),atol=2e-5,rtol=0)
        variants=threshold_variants(raw(proposal,x[roles['calibration']]),y[roles['calibration']],
            raw(proposal,x[roles['control']]),y[roles['control']],dense.task,
            config['decision_grid'],config['decision_shrinkage'])
        local=min(variants,key=lambda v:(-v['validation']['f1_macro'],v['validation']['loss'],v['threshold_fraction']))['validation']
        projection=cycle['projection'];equal_quality(local,projection['proposal_control'])
        for row in projection['candidates']:
            check_candidate_cost(row,proposal,budget,reference['parameters'])
        chosen,feasible=best_row(projection['candidates'],projection['proposal_control'])
        local_pass=chosen is not None and guard(chosen['validation'],projection['proposal_control'])
        action,next_confirmed,next_failures,reason=independent_guard_step(policy,block,confirmed,failures,
                                                                       local_pass,chosen is not None)
        before=dict(kind='active',next_block=block,consecutive_failures=failures,last_confirmed_block=confirmed)
        after=(dict(kind='stopped',retained_through=confirmed,evaluated_block=block,reason=reason) if action=='rollback'
               else dict(kind='active',next_block=block+1,consecutive_failures=next_failures,last_confirmed_block=next_confirmed))
        if 'guard_action' in cycle and (cycle['guard_action']!=action or cycle['guard_before']!=before or cycle['guard_after']!=after):
            raise RuntimeError('independent guard transition/rollback replay differs')
        if action=='rollback':
            all_pass=False;stopped=True
            if cycle['state']=='accepted':
                raise RuntimeError('forbidden compact checkpoint accepted')
            snapshots=[(b,m) for b,m in snapshots if b<=confirmed]
            retained=snapshots[-1][1].base.n_trees if snapshots else 0
            if cycle['rollback_to_trees']!=retained:
                raise RuntimeError('rollback returned to a provisional model')
            if chosen is not None:
                # Old rejected snapshots were deliberately not exported. Rebuild
                # only this frozen diagnostic, then verify its control metric.
                diagnostic,info=compact_proposal(proposal.base,x[roles['train']],y[roles['train']],
                    x[roles['calibration']],y[roles['calibration']],x[roles['control']],y[roles['control']],
                    Budget(**budget),reference['parameters'],config,fixed=allocation=='periodic_fixed',
                    enforce_local_quality=False,prefer_local_quality=True)
                if diagnostic is None or info['chosen']['candidate_id']!=chosen['candidate_id']:
                    raise RuntimeError('rejected diagnostic does not reproduce')
                equal_quality(calibrated_quality(y[roles['control']],raw(diagnostic,x[roles['control']]),
                    dense.task,diagnostic.decision_shifts),chosen['validation'])
        else:
            if cycle['state']!='accepted' or chosen!=projection['chosen']:
                raise RuntimeError('guard accepted a different local candidate')
            path=root/cycle['artifact']
            if sha256(path)!=cycle['sha256']:
                raise RuntimeError('accepted compact snapshot changed')
            model=ThresholdModel.load(path);p,b=array_cost(path)
            if (p,b)!=(chosen['parameters'],chosen['payload_bytes']) or (p,b)!=(cycle['parameters'],cycle['payload_bytes']):
                raise RuntimeError('accepted actual-array cost mismatch')
            if p>budget['parameters'] or b>budget['payload_bytes']:
                raise RuntimeError('even provisional state must satisfy both hard caps')
            fitted=fit_decision_shifts(raw(model,x[roles['calibration']]),y[roles['calibration']],dense.task,config['decision_grid'])
            np.testing.assert_array_equal(model.decision_shifts,(fitted*chosen['threshold_fraction']).astype(np.float32))
            np.testing.assert_array_equal(model.decision_shifts,np.array(chosen['decision_shifts'],np.float32))
            equal_quality(calibrated_quality(y[roles['control']],raw(model,x[roles['control']]),dense.task,
                                             model.decision_shifts),chosen['validation'])
            checkpoint=model;snapshots.append((block,model));confirmed,failures=next_confirmed,next_failures
            all_pass &= local_pass
            stats['accepted_checkpoints']+=1
        stats['feedback_blocks']+=1
    if built!=record['new_built_trees'] or built//40!=record['native_fit_jobs']:
        raise RuntimeError('continuation work ledger omitted rejected trees')
    selected=verify_final_choice(record,snapshots,x,y,roles,full_record['validation_reference'],root,config,stats)
    proof=all_pass and len(record['cycles'])==5 and built==200
    if entry.get('reuse_reason')=='all_pass_equivalence' and not proof:
        raise RuntimeError('unjustified all-pass counterfactual reuse')
    return dict(selected=selected,all_pass_proof=proof,retained_blocks=[b for b,_ in snapshots],
                stopped=stopped,feedback_blocks=len(record['cycles']))


def seal_inputs(data, static, static_path, entries, manifests, prefix_path):
    sealed={}
    def add(path, expected=None):
        path=Path(path);actual=sha256(path)
        if expected is not None and actual!=expected:
            raise RuntimeError('sealed guard source changed: '+str(path))
        if str(path) in sealed and sealed[str(path)]!=actual:
            raise RuntimeError('one physical input has conflicting fingerprints')
        sealed[str(path)]=actual
    add(static_path)
    if prefix_path is not None:
        add(prefix_path)
    for path,expected in manifests:
        add(path,expected)
    for case in data['cases']:
        add(case['split_file'],case['split_sha256'])
    for source in data['datasets'].values():
        add(source['path'],source['sha256'])
    for record in static['records']:
        for fit in record['fits']:
            add(static_path.parent/fit['artifact'],fit['sha256'])
        for choice in [record['full'],*record['decisions']]:
            if choice.get('candidate') is not None:
                add(static_path.parent/choice['artifact'],choice['sha256'])
    for entry in entries:
        r=entry['trajectory'];root=Path(entry['artifact_root'])
        directory=root/f"{r['dataset']}_{r['seed']}"/r['method']
        add(directory/'trajectory.json');add(directory/'worker_data.npz')
        for cycle in r['cycles']:
            if 'fit' in cycle:
                cycle_dir=directory/f"block{cycle['block']}"
                add(cycle_dir/'initial.npz',cycle['initial_sha256'])
                add(cycle_dir/'increment.npz',cycle['fit']['sha256'])
            if cycle['state']=='accepted':
                add(root/cycle['artifact'],cycle['sha256'])
        if r['choice'].get('candidate') is not None:
            add(root/r['choice']['artifact'],r['choice']['sha256'])
    return sealed


def paired_prefixes(entries):
    pairs=[]
    for entry in entries:
        if entry['guard_policy']=='strict':
            continue
        r=entry['trajectory']
        strict=next(e['trajectory'] for e in entries if e['guard_policy']=='strict' and
            e['allocation_policy']==entry['allocation_policy'] and e['trajectory']['dataset']==r['dataset'] and
            e['trajectory']['seed']==r['seed'])
        checked=0
        for old,new in zip(strict['cycles'],r['cycles']):
            if 'fit' not in old or 'fit' not in new:
                break
            if old['initial_sha256']!=new['initial_sha256'] or old['fit']['sha256']!=new['fit']['sha256']:
                raise RuntimeError('guard variants diverged before their first different decision')
            checked+=1
            if old['state']!='accepted' or new['state']!='accepted':
                break
            if old['sha256']!=new['sha256']:
                raise RuntimeError('feasible compact candidate changed under a guard-only variant')
        pairs.append(dict(dataset=r['dataset'],seed=r['seed'],allocation=entry['allocation_policy'],
                          guard=entry['guard_policy'],identical_prefix_blocks=checked,reused=entry['reused']))
    return pairs


def guard_budget_audit(config, report, output):
    started=time.perf_counter()
    audit_only=config.get('audit_only',True);expected_seal=None
    if not audit_only:
        parent,parent_path=locked_json(config['pretest_audit_report'],config['pretest_audit_report_sha256'])
        if parent['status']!='guard_audit_complete' or parent['test_evaluated']:
            raise RuntimeError('all-case pretest guard audit must complete first')
        expected_seal=parent['sealed_test_inputs_sha256']
        if sha256(parent_path.parent/'sealed_test_inputs.json')!=expected_seal:
            raise RuntimeError('pretest guard input seal changed')
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_guard_budget_audit.py',
        'tests/test_local_guard_core.py','tests/test_guard_reuse_core.py','tests/test_guard_budget_study.py',
        'tests/test_closed_budget_projection.py'],cwd=ROOT,check=True,timeout=600)
    data=expanded_manifest(config)
    static,static_path=locked_json(config['joint_development_manifest'],config['joint_development_manifest_sha256'])
    manifests=[(Path(config['expanded_manifest']),config['expanded_manifest_sha256'])]
    entries=[];reference_config=None
    if {p['policy'] for p in config['guard_sources']}!={'strict','warmup2','confirm2'} or len(config['guard_sources'])!=3:
        raise RuntimeError('all three frozen guard sources are required')
    fixed_keys=('datasets','guard_seeds_by_dataset','guard_allocation_policies','joint_development_manifest_sha256',
        'expanded_manifest_sha256','guard_baseline_manifest_sha256','closed_max_blocks','own_pool_limit_bytes',
        'lr','ntrees','max_depth','lambda_l2','decision_grid','decision_shrinkage','margin_scale')
    for source in config['guard_sources']:
        part,path=locked_json(source['path'],source['sha256']);manifests.append((path,source['sha256']))
        if part['test_evaluated'] or part['static_manifest_sha256']!=sha256(static_path):
            raise RuntimeError('guard source used different data or static control')
        if len(part['cases'])!=18 or len(part['records'])!=36 or part['config']['guard_gate']:
            raise RuntimeError('guard source is not the complete registered comparison')
        if reference_config is None:
            reference_config=part['config']
        for key in fixed_keys:
            if part['config'][key]!=reference_config[key]:
                raise RuntimeError('guard-only comparison changed '+key)
        for entry in part['records']:
            if entry['guard_policy']!=source['policy']:
                raise RuntimeError('wrong guard source policy')
            entries.append(entry)
    keys=[(e['trajectory']['dataset'],e['trajectory']['seed'],e['allocation_policy'],e['guard_policy']) for e in entries]
    expected={(name,seed,allocation,policy) for name,seeds in reference_config['guard_seeds_by_dataset'].items()
              for seed in seeds for allocation in ('periodic_fixed','joint_adaptive')
              for policy in ('strict','warmup2','confirm2')}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise RuntimeError('duplicate or incomplete paired guard coverage')
    prefixes,prefix_path=locked_json(config['prefix_development_manifest'],config['prefix_development_manifest_sha256'])
    manifests.append((prefix_path,sha256(prefix_path)))
    sealed=seal_inputs(data,static,static_path,entries,manifests,prefix_path)
    for r in prefixes['records']:
        for d in r['choices']:
            if d.get('candidate') is not None:
                path=prefix_path.parent/d['artifact']
                if sha256(path)!=d['sha256']:
                    raise RuntimeError('frozen prefix candidate changed')
                sealed[str(path)]=d['sha256']
    write_report(output/'sealed_test_inputs.json',sealed);seal=sha256(output/'sealed_test_inputs.json')
    if expected_seal is not None and seal!=expected_seal:
        raise RuntimeError('shared guard test inputs differ from successful audit')
    emit('guard_inputs_sealed',paths=len(sealed),sha256=seal,test_started=False)
    stats=dict(models=0,maximum_selection_replay_error=0.,accepted_checkpoints=0,feedback_blocks=0,
               unique_trajectories=0,all_pass_equivalence_replays=0)
    jobs=[];audits=[];checked={}
    with threadpool_limits(limits=1):
        for name in reference_config['datasets']:
            x,y=load_expanded(name,data)
            for seed in reference_config['guard_seeds_by_dataset'][name]:
                check_deadline(config,300)
                case=next(c for c in data['cases'] if c['dataset']==name and c['seed']==seed)
                roles=load_roles(case,len(x));cache={}
                full_record=next(r for r in static['records'] if r['dataset']==name and r['seed']==seed)
                full=verify_model(full_record['full'],static_path.parent,x,y,roles,config,cache,stats)
                c=len(full['model'].base.bias);reference=full_record['reference_cost']
                if (full['parameters']-c,full['payload_bytes']-4*c)!=(reference['parameters'],reference['payload_bytes']):
                    raise RuntimeError('full exact denominator changed')
                for entry in (e for e in entries if e['trajectory']['dataset']==name and e['trajectory']['seed']==seed):
                    r=entry['trajectory'];key=(entry['artifact_root'],name,seed,r['method'])
                    if key in checked:
                        if checked[key]['record']!=r:
                            raise RuntimeError('one physical trajectory has conflicting metadata')
                        verified=checked[key]['result']
                    else:
                        verified=verify_trajectory(entry,full_record,static_path.parent,x,y,roles,config,stats)
                        checked[key]=dict(record=r,result=verified);stats['unique_trajectories']+=1
                    if entry.get('reuse_reason')=='all_pass_equivalence':
                        if not verified['all_pass_proof']:
                            raise RuntimeError('unproved policy replay')
                        stats['all_pass_equivalence_replays']+=1
                    jobs.append(dict(dataset=name,seed=seed,method=entry['allocation_policy']+'__'+entry['guard_policy'],
                        allocation_policy=entry['allocation_policy'],guard_policy=entry['guard_policy'],
                        reused=entry['reused'],reuse_reason=entry.get('reuse_reason'),selected=verified['selected'],
                        state=r['choice']['state'],budget=r['budget'],new_trees=0 if entry['reused'] else r['new_built_trees'],
                        local_rejections=sum(cycle['state']!='accepted' for cycle in r['cycles']),
                        provisional=sum(cycle.get('guard_action')=='provisional' for cycle in r['cycles'])))
                controls=[('static_joint',next(d for d in full_record['decisions'] if d['fraction']==.75 and
                          d['family']=='joint_static' and d['cap_percent']==5),static_path.parent),
                          ('static_rank_residual',next(d for d in full_record['decisions'] if d['fraction']==.75 and
                          d['family']=='rank_residual' and d['cap_percent']==5),static_path.parent)]
                p=next(r for r in prefixes['records'] if r['dataset']==name and r['seed']==seed)
                controls.append(('native_prefix',next(d for d in p['choices'] if d['fraction']==.75 and
                                 d['selection_scope']=='all_depths'),prefix_path.parent))
                for method,d,root in controls:
                    verified=verify_model(d,root,x,y,roles,config,cache,stats) if d.get('candidate') is not None else None
                    selected=(dict(artifact=str(root/d['artifact']),sha256=d['sha256'],state=d['state'],
                        parameters=verified['parameters'],payload_bytes=verified['payload_bytes'],csr_bytes=verified['csr_bytes'],
                        candidate=d['candidate']) if verified is not None else None)
                    jobs.append(dict(dataset=name,seed=seed,method=method,allocation_policy=None,guard_policy=None,
                        reused=True,reuse_reason='frozen_static_control',selected=selected,state=d['state'],
                        budget=d['budget'],new_trees=0,local_rejections=0,provisional=0))
                audits.append(dict(dataset=name,seed=seed))
                emit('guard_case_audited',dataset=name,seed=seed,test_scored=False)
                del cache;gc.collect()
            del x,y;gc.collect()
        paired=paired_prefixes(entries)
        report.update(status='guard_audit_complete',test_evaluated=False,audit=stats,audit_cases=audits,
            selected_jobs=jobs,paired_prefixes=paired,sealed_test_inputs_sha256=seal,input_paths_verified=len(sealed),
            tree_fits=0,elapsed_seconds=time.perf_counter()-started)
        if not audit_only:
            emit('guard_all_cases_verified_before_test',cases=len(audits),jobs=len(jobs))
            score_guard_jobs(config,report,output,data,static,static_path,jobs)
    for path,expected_hash in sealed.items():
        if sha256(path)!=expected_hash:
            raise RuntimeError('guard sealed input changed during verification')
    report['elapsed_seconds']=time.perf_counter()-started
    emit('guard_audit_complete',cases=len(audits),test_scored=not audit_only,**stats)


def score_guard_jobs(config,report,output,data,static,static_path,jobs):
    """Called only after the successful parent seal AND all current case audits."""
    rows=[];label_rows=[]
    for case in report['audit_cases']:
        name,seed=case['dataset'],case['seed']
        x,y=load_expanded(name,data)
        info=next(c for c in data['cases'] if c['dataset']==name and c['seed']==seed)
        roles=load_roles(info,len(x))
        reference=next(r for r in static['records'] if r['dataset']==name and r['seed']==seed)
        full=ThresholdModel.load(static_path.parent/reference['full']['artifact'])
        full_raw=raw(full,x[roles['test']]);task=full.base.task;c=len(full.base.bias)
        full_q=calibrated_quality(y[roles['test']],full_raw,task,full.decision_shifts)
        full_labels=labels_from_raw(full_raw,task,full.decision_shifts)
        cache={}
        for job in (r for r in jobs if r['dataset']==name and r['seed']==seed):
            row=dict(dataset=name,seed=seed,method=job['method'],allocation_policy=job['allocation_policy'],
                guard_policy=job['guard_policy'],state=job['state'],reused=job['reused'],reuse_reason=job['reuse_reason'],
                new_trees=job['new_trees'],local_rejections=job['local_rejections'],provisional=job['provisional'],
                joint_success=False,test_quality_pass=False,parameters=None,bytes=None,csr_bytes=None,
                memory_fraction=None,coefficient_fraction=None,trees=None,leaves=None,rank=None,nnz=None,
                loss=None,macro_f1=None,weighted_f1=None,loss_ratio=None,macro_retained=None,weighted_retained=None,
                reference_loss=full_q['loss'],reference_macro_f1=full_q['f1_macro'],reference_weighted_f1=full_q['f1_weighted'],
                rare_labels=None,rare_full_macro_fixed_domain=None,rare_macro_fixed_domain=None,rare_macro_retained=None,
                label_f1_losses=None,label_f1_gains=None,full_macro_domain=None,macro_domain=None)
            selected=job['selected']
            if selected is not None:
                artifact=selected['artifact']
                if artifact not in cache:
                    if sha256(artifact)!=selected['sha256']:
                        raise RuntimeError('guard test candidate changed')
                    model=ThresholdModel.load(artifact);z=raw(model,x[roles['test']])
                    q=calibrated_quality(y[roles['test']],z,task,model.decision_shifts)
                    labels=labels_from_raw(z,task,model.decision_shifts)
                    details,diagnostic=compare_labels(y[roles['test']],full_labels,labels,y[roles['train']],task,c)
                    if max(abs(q['f1_macro']-diagnostic['independent_macro']),
                           abs(q['f1_weighted']-diagnostic['independent_weighted']))>1e-12:
                        raise RuntimeError('guard independent per-label F1 count differs')
                    diagnostic.pop('independent_macro');diagnostic.pop('independent_weighted')
                    cache[artifact]=(model,q,details,diagnostic)
                model,q,details,diagnostic=cache[artifact]
                if model.parameter_count>job['budget']['parameters'] or model.payload_bytes>job['budget']['payload_bytes']:
                    raise RuntimeError('test artifact exceeded hard joint cost')
                row.update(diagnostic)
                row.update(test_quality_pass=guard(q,full_q),joint_success=job['state']=='accepted' and guard(q,full_q),
                    parameters=model.parameter_count,bytes=model.payload_bytes,csr_bytes=selected['csr_bytes'],
                    memory_fraction=model.payload_bytes/reference['reference_cost']['payload_bytes'],
                    coefficient_fraction=model.parameter_count/reference['reference_cost']['parameters'],
                    trees=model.base.n_trees,leaves=len(model.base.coefficients),rank=model.base.coefficients.shape[1],
                    nnz=len(model.values),loss=q['loss'],macro_f1=q['f1_macro'],weighted_f1=q['f1_weighted'],
                    loss_ratio=q['loss']/full_q['loss'],macro_retained=q['f1_macro']/full_q['f1_macro'] if full_q['f1_macro'] else None,
                    weighted_retained=q['f1_weighted']/full_q['f1_weighted'] if full_q['f1_weighted'] else None)
                label_rows.extend(dict(dataset=name,seed=seed,method=job['method'],**r) for r in details)
            rows.append(row)
        csv_table(output/'comparisons.csv',rows);csv_table(output/'per_label.csv',label_rows)
        emit('guard_test_case_complete',dataset=name,seed=seed,methods=sum(r['dataset']==name and r['seed']==seed for r in rows))
        del cache,x,y;gc.collect()
    summaries=[]
    for name,method in sorted({(r['dataset'],r['method']) for r in rows}):
        group=[r for r in rows if r['dataset']==name and r['method']==method]
        summaries.append(dict(dataset=name,method=method,cases=len(group),
            development_acceptances=sum(r['state']=='accepted' for r in group),
            joint_successes=sum(r['joint_success'] for r in group),test_passes=sum(r['test_quality_pass'] for r in group),
            missing=sum(r['bytes'] is None for r in group),new_trees=sum(r['new_trees'] for r in group)))
    paired=[]
    for row in (r for r in rows if r['guard_policy'] in ('warmup2','confirm2')):
        strict=next(r for r in rows if r['dataset']==row['dataset'] and r['seed']==row['seed'] and
                    r['allocation_policy']==row['allocation_policy'] and r['guard_policy']=='strict')
        paired.append(dict(dataset=row['dataset'],seed=row['seed'],allocation=row['allocation_policy'],
            guard=row['guard_policy'],strict_success=strict['joint_success'],soft_success=row['joint_success'],
            rescued=not strict['joint_success'] and row['joint_success'],
            lost=strict['joint_success'] and not row['joint_success'],
            strict_state=strict['state'],soft_state=row['state'],
            delta_macro_f1=(row['macro_f1']-strict['macro_f1']) if row['macro_f1'] is not None and strict['macro_f1'] is not None else None))
    csv_table(output/'summaries.csv',summaries);csv_table(output/'paired_guard.csv',paired)
    report.update(status='complete',test_evaluated=True,records=rows,summaries=summaries,paired_guard=paired)
