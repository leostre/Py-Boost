"""Stage II: coefficient-only reserve experiments; no tree learner is imported."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys,gc,hashlib
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from unified_serving_core import coalesce
from unified_training import evaluate,serialize,quality_row
from unified_frontier_core import make_folds,sufficient,metrics
from unified_recalibration import audit_case
from adaptive_basis_core import risk_weights,risk_basis
from adaptive_controller_core import extend_basis
from experiments.sparse_leaf_core import leaf_design
from experiments.shared_leaf_basis import reference_class_model,probabilities
from window_reserve_core import quota,support,coordinate_scores,fit_support,fit_extra_rank,loss_gradient


def replace_residual(model,index,keys,values):
    block=model.blocks[index];c=len(model.bias);keys=np.asarray(keys,np.int64);values=np.asarray(values,np.float32)
    nonzero=values!=0;rows,cols=np.divmod(keys[nonzero],c)
    updated=replace(block,rows=rows.astype(np.int32),columns=cols.astype(np.int32),values=values[nonzero])
    return replace(model,blocks=model.blocks[:index]+(updated,)+model.blocks[index+1:])


def topology_signature(model):
    return hashlib.sha256(b''.join(a.tobytes() for b in model.blocks for a in (b.base.nodes,b.base.node_offsets,b.base.leaf_offsets))).hexdigest()


def rank_variant(model,index,basis,extra_values):
    old=model.blocks[index];values=np.column_stack((old.base.coefficients,extra_values)).astype(np.float32)
    base=replace(old.base,coefficients=values,basis=basis.astype(np.float32))
    full=basis.shape[1]==len(model.bias)-int(model.task=='multiclass')
    if full:
        base=replace(base,coefficients=(values.astype(float)@basis.astype(float).T).astype(np.float32),basis=np.empty((0,0),np.float32),representation='dense')
        if model.task=='multiclass':base=reference_class_model(base)
    block=replace(old,base=base)
    return replace(model,blocks=model.blocks[:index]+(block,)+model.blocks[index+1:])


def run(cfg,out):
    started=perf_counter();out.mkdir(parents=True,exist_ok=True);spec=cfg['window_reserve'];assert spec['inputs_ready']
    stage1=locked(spec['banks']['path'],spec['banks']['sha256']);assert stage1['status']=='saved_banks_complete'
    from audit_saved_bank import audit
    stage1_audit=audit(stage1,out/'stage_I_independent_audit.json')
    environment=check_native_environment(cfg);fresh=locked(spec['fresh']['path'],spec['fresh']['sha256']);wide=locked(spec['wide']['path'],spec['wide']['sha256'])
    refs=[r for r in fresh['cases'] if any(r['case'].startswith(name+'_') for name in ('emotions','medical','satimage'))]
    refs += [r for r in wide['cases'] if r['case'] in ('mediamill_1104','bibtex_2304')]
    entries=[];quota_rows=[];solver_records=[];fit_count=0
    write(out/'protocol_seal.json',dict(banks=spec['banks'],windows=[8,40],fractions=[.01,.02,.05],random_seeds=[17,29,43],
        objective='sum uncalibrated training logistic/softmax loss +0.5*lambda*coefficient_increment_norm2',maxiter=60,
        tree_training=False,base_coefficients_fixed=True,support_score_scope='train only',rank_extra_scope='train-gradient basis extension',
        scope='posthoc fixed-tree reserve; does not replay or validate online training trajectory'))
    for ref in refs:
        source=locked(ref['path'],ref['sha256']);cid=source['case'];record=source['record'];task=source['task'];data=data_before_test(record)
        source_model=next(m for m in source['models'] if m['family']=='fixed_risk');assert digest(source_model['artifact'])==source_model['sha256']
        model=coalesce(load_protected(source_model['artifact']));indices=[i for i,b in enumerate(model.blocks) if b.base.representation=='low_rank']
        assert len(indices)==1;index=indices[0];block=model.blocks[index];assert len(block.values)==0
        c=len(model.bias);r=block.base.basis.shape[1];old_topology=topology_signature(model)
        directory=out/cid;directory.mkdir(exist_ok=True);folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028)
        models=[];raw_train=model.raw(data['train_x']);visits=block.base.leaf_indices(data['train_x']);design=leaf_design(visits,len(block.base.coefficients));del visits
        assert np.all(np.asarray(design.sum(0)).ravel()>0),'saved leaves must have train support'
        scores=coordinate_scores(design,raw_train,data['train_y'],task,cfg['lambda_l2']);_,importance=risk_weights(data['train_y'],raw_train,task)
        balanced=scores*importance[None,:];cache={};rank_cache={};matches=[]
        def store(candidate,uid,family,**meta):
            path=directory/uid;path.mkdir(parents=True,exist_ok=True);raw=candidate.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,path)
            m=serialize(candidate,raw,ev,path,uid,dict(family=family,**meta));models.append(m);return m
        baseline=store(model,'baseline','basis',source=source_model)
        if cid.startswith(('mediamill_','bibtex_')):full_source=next(m for m in source['models'] if m['family']=='native_full')
        else:full_source=next(m for m in source['models'] if m['uid']=='ordinary/d3/t200')
        assert digest(full_source['artifact'])==full_source['sha256'];store(coalesce(load_protected(full_source['artifact'])),'full_reference','ordinary',source=full_source)
        # Extension preserves the old span; extra directions come from train gradients.
        _,gradient,_=loss_gradient(raw_train,data['train_y'],task);rw,cw=risk_weights(data['train_y'],raw_train,task)
        extension=extend_basis(block.base.basis,risk_basis(-gradient,task,rw,cw)[0]);extension[:,:r]=block.base.basis
        windows=[40] if cid.startswith(('mediamill_','bibtex_')) else [8,40]
        for fraction in (.01,.02,.05):
            maximum=quota(scores.size,fraction);dummy=replace_residual(model,index,np.arange(maximum),np.ones(maximum,np.float32));ceiling=block_cost(dummy)['cost_bytes'];del dummy
            row=dict(case=cid,outputs=c,rank=r,trees=block.base.n_trees,leaves=len(scores),fraction=fraction,
                independent_floor=sum(quota(int(l*c),fraction) for l in np.diff(block.base.leaf_offsets)),global_slots=maximum,byte_ceiling=ceiling)
            quota_rows.append(row)
            for window in windows:
                group=[]
                methods=[('floor',0),('carry',0),('uniform',0),('score',0),('balanced',0)]+[('random',seed) for seed in (17,29,43)]
                for kind,seed in methods:
                    keys,allocation=support(balanced if kind=='balanced' else scores,block.base.leaf_offsets,fraction,window,'score' if kind=='balanced' else kind,seed)
                    signature=hashlib.sha256(keys.tobytes()).hexdigest();uid=f'p{fraction:g}-w{window}-{kind}-{seed}'
                    reused=signature in cache
                    if reused:values,fit,old_uid=cache[signature]
                    else:
                        values,fit=fit_support(design,keys,raw_train,data['train_y'],task,cfg['lambda_l2'],spec['maxiter']);fit_count+=1
                        old_uid=uid;cache[signature]=(values,fit,uid);solver_records.append(dict(case=cid,uid=uid,fit=fit))
                    candidate=replace_residual(model,index,keys,values)
                    assert topology_signature(candidate)==old_topology
                    np.testing.assert_array_equal(candidate.blocks[index].base.coefficients,block.base.coefficients)
                    np.testing.assert_array_equal(candidate.blocks[index].base.basis,block.base.basis)
                    # Independent design replay of the float32 deployed correction.
                    delta=np.zeros_like(raw_train)
                    for output in np.unique(keys%c):
                        mask=keys%c==output;delta[:,output]=np.asarray(design[:,keys[mask]//c]@values[mask]).ravel()
                    assert np.max(np.abs(candidate.raw(data['train_x'])-raw_train-delta))<2e-5
                    item=store(candidate,uid,kind,fraction=fraction,window=window,random_seed=seed,allocated_slots=len(keys),effective_nnz=len(candidate.blocks[index].values),
                        byte_ceiling=ceiling,allocation=allocation,fit=fit,coefficient_fit_reused_from=old_uid if reused else None)
                    assert item['cost_bytes']<=ceiling
                    group.append(dict(uid=uid,kind=kind,allocated=len(keys),effective=item['nnz'],cost_bytes=item['cost_bytes']))
                same=[g for g in group if g['kind']!='floor'];matches.append(dict(fraction=fraction,window=window,
                    equal_allocated_slots=len({g['allocated'] for g in same})==1,equal_actual_bytes=len({g['cost_bytes'] for g in same})==1,models=group))
            # Largest affordable added rank under the same residual byte ceiling.
            affordable=[]
            for p in sorted(set([1,2,4,8,16,extension.shape[1]-r])):
                if not 1<=p<=extension.shape[1]-r:continue
                b=extension[:,:r+p];trial=rank_variant(model,index,b,np.zeros((len(scores),p),np.float32))
                if block_cost(trial)['cost_bytes']<=ceiling:affordable.append(p)
            if affordable:
                p=max(affordable);b=extension[:,:r+p]
                if p not in rank_cache:
                    values,fit=fit_extra_rank(design,b[:,r:].astype(float),raw_train,data['train_y'],task,cfg['lambda_l2'],spec['maxiter']);rank_cache[p]=(values,fit);fit_count+=1
                    solver_records.append(dict(case=cid,uid=f'rankplus{p}',fit=fit))
                values,fit=rank_cache[p];candidate=rank_variant(model,index,b,values)
                predicted=raw_train+np.asarray(design@values.astype(float))@b[:,r:].astype(float).T
                assert np.max(np.abs(probabilities(candidate.raw(data['train_x']),task)-probabilities(predicted,task)))<2e-5
                assert topology_signature(candidate)==old_topology
                item=store(candidate,f'p{fraction:g}-rankplus{p}','rank',fraction=fraction,rank=r+p,byte_ceiling=ceiling,fit=fit)
                assert item['cost_bytes']<=ceiling
            else:row['rank_control']='no_affordable_positive_rank_extension'
            event('window_reserve_fraction',case=cid,fraction=fraction,models=len(models),byte_ceiling=ceiling)
        case=dict(case=cid,task=task,record=record,models=models,source_basis=source_model,matches=matches,topology_signature=old_topology,
            scope='same frozen base and topology for reserve methods; rank control changes decoder and extra coefficients only')
        case['audit']=audit_case(case,data);file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit']))
        write(out/'progress.json',dict(cases=entries,quota_rows=quota_rows,unique_coefficient_fits=fit_count));event('window_reserve_case',case=cid,models=len(models),audit=case['audit'])
        del model,design,data;gc.collect();memory_guard()
    # All supports, fits, calibration and the entire evaluated catalogue are sealed.
    write(out/'pretest_seal.json',dict(cases=entries,quota_rows=quota_rows,test_labels_used=False,new_trees=0))
    points=[]
    for ref in entries:
        case=locked(ref['path'],ref['sha256']);record=case['record'];cid=case['case'];directory=out/cid
        with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(record['dataset']['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        target=directory/'test_targets.npz';np.savez_compressed(target,y=y,rows=ids)
        for i,m in enumerate(case['models']):
            assert digest(m['artifact'])==m['sha256'];z=load_protected(m['artifact']).raw(x);control=quality_row(m['evaluations'],case['task'])['refit_control']
            pred,loss,brier=sufficient(y,z,case['task'],control['temperature'],control['shifts']);q=metrics(y,pred,loss,brier,case['task'],z.shape[1])
            file=directory/f'test_prediction{i:03d}.npz';np.savez_compressed(file,raw=z,prediction=pred,loss=loss,brier=brier)
            points.append(dict(case=cid,task=case['task'],uid=m['uid'],family=m['family'],cost_bytes=m['cost_bytes'],quality=q,
                development=quality_row(m['evaluations'],case['task'])['quality'],control=control,outputs=z.shape[1],
                raw=dict(path=str(file),sha256=digest(file)),targets=dict(path=str(target),sha256=digest(target))))
        write(out/'test_points.json',points);event('window_reserve_test',case=cid,models=len(case['models']))
    result=dict(status='window_reserve_complete',cases=entries,points=points,quota_rows=quota_rows,solver_records=solver_records,unique_coefficient_fits=fit_count,
        new_trees=0,environment=environment,stage_I_independent_audit=stage1_audit,elapsed_seconds=perf_counter()-started,seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),
        limitations=['saved full ensemble state, not historical per-window raw state','support/values train-only; D calibrates outputs equally',
        'coefficient-only convex refit, not new tree or online training','zero optimized coefficients removed; exact byte matches explicitly audited',
        'rank control may leave unused bytes; it is a common ceiling, not always exact byte equality','old tests exploratory; no unseen dataset added'])
    write(out/'report.json',result);event('window_reserve_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),models=len(points),coefficient_fits=fit_count,new_trees=0)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
