"""Fixed post-hoc transfer study, separate from true compact-training claims."""
from pathlib import Path
from dataclasses import asdict
import gc,json,math,subprocess,sys
from time import perf_counter
import numpy as np
from budget_quality_worker import digest,locked,write,event,native_load
from budget_quality_core import quality
from abl_quality_core import (array_ledger,full_output_base,prepare_rank,build_representation,
    RepresentationPlan,fingerprint,fitted_control)
from abl_quality_worker import evaluate_candidate,save_logits,check_native_environment
from abl_stability_core import select_stable,NumericalTolerance
from calibration_pilot_core import make_plans,Scheme,POLICIES
from experiments.parallel_leaf_core import prefix
from experiments.sparse_leaf_core import leaf_design,training_weights
from experiments.shared_leaf_basis import functional_basis
from experiments.output_calibration_core import calibrated_quality,fit_decision_shifts


def data_before_test(record):
    case,source=record['case'],record['dataset']
    if digest(case['split_file'])!=case['split_sha256'] or digest(source['path'])!=source['sha256']:
        raise RuntimeError('transfer source changed')
    with np.load(case['split_file'],allow_pickle=False) as f:
        roles={k:f[k].copy() for k in ('train','calibration','control','selection','test')}
    joined=np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(joined),np.arange(len(joined))):raise RuntimeError('invalid transfer roles')
    dev=np.r_[roles['calibration'],roles['control'],roles['selection']]
    with np.load(source['path'],allow_pickle=False) as f:
        x=f['features'];y=f['target']
        result=dict(train_x=x[roles['train']].astype(np.float32),train_y=y[roles['train']].copy(),
            dev_x=x[dev].astype(np.float32),dev_y=y[dev].copy(),roles=roles,
            sizes=[len(roles[k]) for k in ('calibration','control','selection')])
    if case['task']=='multiclass' and set(result['train_y'])!=set(range(case['outputs'])):
        raise RuntimeError('training output domain incomplete')
    return result


def evaluated_rows(uid,z,data,cfg,task):
    if task=='multiclass':
        plan=next(p for p in make_plans(data['dev_y'],data['sizes'],0,(1.,),cfg['calibration_pilot']['seed'])
            if p.scheme is Scheme.POOLED)
        return evaluate_candidate(uid,z,data['dev_y'],[plan],cfg)
    end=sum(data['sizes'][:2]);fitted=fit_decision_shifts(z[:end],data['dev_y'][:end],task,cfg['decision_grid'])
    validation={}
    for p in POLICIES:
        # No multiclass temperature fit is transferred to independent sigmoid
        # outputs. Alias names exist solely for the common frozen selector.
        shifts=(fitted*p.shift_fraction).astype(np.float32)
        validation[p.name]=calibrated_quality(data['dev_y'][end:],z[end:],task,shifts)
    return [dict(uid=uid,plan=0,scheme='pooled',repeat=0,validation=validation,
        refit=dict(temperature=1.,raw_shifts=fitted.tolist(),temperature_shifts=fitted.tolist()),
        temperature_scope='fixed1 for multilabel; thresholds only')]


def fit_state(model,design,y):
    full=full_output_base(model);contribution=np.asarray(design@full.coefficients.astype(float))
    mean,basis,singular=functional_basis(contribution,full.task);teacher=contribution+full.bias
    return dict(mean=mean,basis=basis,singular=singular,teacher=teacher,
        frequency=np.asarray(design.mean(axis=0)).ravel(),
        weights=training_weights(y,teacher,full.task,'balanced_mse',.5))


def development_case(cfg,manifest,case,output,ledger):
    cid=f'{case["dataset"]}_{case["seed"]}';directory=output/'development'/cid;directory.mkdir(parents=True,exist_ok=True)
    record=dict(case=case,dataset=manifest['datasets'][case['dataset']]);data=data_before_test(record);task=case['task']
    source_ref=cfg['bank_sources']['validation'];source=locked(source_ref['path'],source_ref['sha256'])
    if task=='multilabel':
        old=next(r for r in source['records'] if r['dataset']==case['dataset'] and r['seed']==case['seed'])
        if old['split_sha256']!=case['split_sha256']:raise RuntimeError('saved full roles differ')
        item=old['full']
        if digest(item['artifact'])!=item['sha256']:raise RuntimeError('saved multilabel full changed')
        teacher=native_load(item['artifact'])
        if teacher.base.n_trees!=200 or teacher.base.task!=task:raise RuntimeError('expected saved200-tree multilabel control')
        sources=[(t,prefix(teacher,t)) for t in (40,80,160,200)]
        origin=dict(kind='reused_full200',artifact=item['artifact'],sha256=item['sha256'])
    else:
        if ledger['reserved_rounds']+800>4800:raise RuntimeError('transfer new-tree cap exhausted')
        ledger['reserved_rounds']+=800;ledger['attempts'].append(dict(case=cid,state='reserved',rounds=800))
        write(output/'training_ledger.json',ledger)
        request=dict(record=record,depth=5,minimum_leaf=50,rounds=800,
            training_config={k:cfg[k] for k in ('lr','lambda_l2','ntrees','own_pool_limit_bytes')})
        request_path=directory/'training_request.json';write(request_path,request)
        native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
        trajectory_dir=output/'new_full_trajectories'/cid
        subprocess.run([native,str(Path(__file__).resolve().parent/'abl_width_worker.py'),str(request_path),str(trajectory_dir)],
            check=True,timeout=3600)
        trajectory_path=trajectory_dir/'trajectory.json';trajectory=json.loads(trajectory_path.read_text())
        if trajectory['status']!='complete':raise RuntimeError('new transfer teacher incomplete')
        ledger['attempts'][-1].update(state='complete',fit=trajectory['fit']);write(output/'training_ledger.json',ledger)
        sources=[]
        for item in trajectory['models']:
            if digest(item['artifact'])!=item['sha256']:raise RuntimeError('new teacher changed')
            sources.append((item['rounds'],native_load(item['artifact'])))
        origin=dict(kind='new_full800_depth5_leaf50',path=str(trajectory_path),sha256=digest(trajectory_path))
    metas={};rows=[];profiles=[]
    def add(uid,model,meta):
        path=output/'models'/cid/(uid.replace('/','-')+'.npz');path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
        z=model.prepare_inference().raw(data['dev_x']);ledger=array_ledger(model)
        metas[uid]=dict(uid=uid,source=dict(artifact=str(path),sha256=digest(path)),
            cost_bytes=ledger['cost_bytes'],ledger=ledger,rank=model.base.basis.shape[1] or None,nnz=len(model.values),
            trees=model.base.n_trees,leaves=len(model.base.coefficients),nodes=len(model.base.nodes),
            development_cache=save_logits(directory/(uid.replace('/','-')+'-logits.npz'),z),**meta)
        rows.extend(evaluated_rows(uid,z,data,cfg,task))
    count=0
    for trees,teacher in sources:
        parent=f'pyboost/t{trees}';add(parent,teacher,dict(family='pyboost',identity=f't{trees}'))
        if trees not in (160,200,400,800):continue
        started=perf_counter();design=leaf_design(teacher.base.leaf_indices(data['train_x']),len(teacher.base.coefficients))
        state=fit_state(teacher,design,data['train_y']);ceiling=min(case['outputs']-int(task=='multiclass'),len(teacher.base.coefficients),len(data['train_y']))
        levels=sorted({max(1,math.ceil(f*ceiling)) for f in (.25,.5,.75,1.)})
        spectrum=state['singular']**2
        profiles.append(dict(parent_uid=parent,training_rows=len(data['train_y']),features=data['train_x'].shape[1],outputs=case['outputs'],
            effective_rank90=int(np.searchsorted(np.cumsum(spectrum),.9*spectrum.sum())+1),
            numerical_rank=int(np.sum(spectrum>1e-12*max(spectrum[0],1e-30))),source_parameters=teacher.parameter_count))
        seen={}
        for rank in levels:
            pure,residual,order=prepare_rank(teacher,design,state,rank)
            for fraction in (0.,.01,.02,.05):
                model=build_representation(pure,residual,order,state,
                    RepresentationPlan(rank,fraction,math.floor(fraction*teacher.parameter_count)))
                count+=1;key=fingerprint(model)
                if key in seen:continue
                uid=f'compact/t{trees}-r{rank}-p{int(100*fraction):02d}';seen[key]=uid
                add(uid,model,dict(family='compact',identity=uid,parent_uid=parent,
                    requested_rank=rank,residual_fraction=fraction,parent_parameters=teacher.parameter_count))
            del pure,residual,order
        profiles[-1]['construction_seconds']=perf_counter()-started
        del state,design;gc.collect()
    reference=metas['pyboost/t200']['cost_bytes']-4;choices=[]
    modes=('raw','temperature_shifts') if task=='multiclass' else ('raw','shifts')
    for mode in modes:
        for objective in ('loss','f1_macro'):
            for group in ('native','ab','abl','deployed'):
                for ratio in (.25,.5,.75,1.,1.5,2.,4.):
                    cap=math.floor(reference*ratio)+4
                    chosen=select_stable(rows,metas,group,mode,objective,cap,NumericalTolerance(1e-6,1e-12))
                    choices.append(dict(plan=0,mode=mode,objective=objective,group=group,budget_ratio=ratio,budget_bytes=cap,chosen=chosen))
    full_choices=[]
    for mode in modes:
        for objective in ('loss','f1_macro'):
            cap=max(m['cost_bytes'] for m in metas.values())+1
            full_choices.append(dict(plan=0,mode=mode,objective=objective,group='native',budget_bytes=cap,
                chosen=select_stable(rows,metas,'native',mode,objective,cap,NumericalTolerance(1e-6,1e-12))))
    result=dict(case_id=cid,record=record,task=task,metas=metas,rows=rows,choices=choices,source=origin,
        reference_b0_bytes=reference,requested_representations=count,training_profiles=profiles,full_reference_choices=full_choices,
        selection_scope='pooled calibration/control fit; original selection only; no test',test_accessed=False)
    path=directory/'case.json';write(path,result)
    event('abl_transfer_case_development_complete',case=cid,candidates=len(metas),requested_representations=count,path=str(path),sha256=digest(path))
    return dict(case_id=cid,path=str(path),sha256=digest(path))


def development(cfg,output):
    check_native_environment(cfg)
    source=locked(cfg['abl_transfer_data_completed']['path'],cfg['abl_transfer_data_completed']['sha256'])
    manifest=locked(source['manifest']['path'],source['manifest']['sha256'])
    if len(manifest['cases'])!=15:raise RuntimeError('fifteen preregistered transfer cases required')
    ledger=dict(reserved_rounds=0,attempts=[]);entries=[]
    for case in manifest['cases']:
        entries.append(development_case(cfg,manifest,case,output,ledger))
        write(output/'development_progress.json',dict(cases=entries,training_ledger=ledger,test_accessed=False))
    if ledger['reserved_rounds']!=4800 or any(a['state']!='complete' for a in ledger['attempts']):
        raise RuntimeError('transfer training ledger incomplete')
    from audit_abl_transfer import audit_development
    audit_path=output/'development_audit.json';write(audit_path,audit_development(entries,ledger))
    seal=output/'pretest_choices.json';write(seal,dict(status='transfer_choices_sealed_before_test',cases=entries,
        source_manifest=source['manifest'],test_used_for_selection=False,training_ledger=ledger,development_audit_sha256=digest(audit_path)))
    report=output/'report.json';write(report,dict(status='transfer_development_complete',cases=entries,
        training_ledger=ledger,pretest_seal_sha256=digest(seal),test_accessed=False,
        scope='posthoc representation transfer; not evidence of new multilabel compact-gradient training'))
    event('abl_transfer_development_complete',path=str(report),sha256=digest(report),cases=len(entries),new_rounds=4800)


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2])
    native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    if '--native' not in sys.argv:
        subprocess.run([native,__file__,sys.argv[1],sys.argv[2],'--native'],check=True,timeout=21600);return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):development(cfg,out)


if __name__=='__main__':main()
