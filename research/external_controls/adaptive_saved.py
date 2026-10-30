"""A1 saved-topology comparison: no tree training, no test-label API."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from abl_budget_revision_core import decode_model,hessian,pack
from abl_budget_revision import group_model
from unified_training_core import prepare_training_group
from experiments.shared_leaf_basis import functional_basis
from experiments.sparse_leaf_core import contrast
from unified_training import evaluate,serialize,quality_row
from unified_frontier_core import make_folds
from unified_recalibration import audit_case
from adaptive_basis_core import risk_weights,risk_basis,all_support_order


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();environment=check_native_environment(cfg);spec=cfg['adaptive_saved']
    source=locked(spec['source']['path'],spec['source']['sha256']);entries=[]
    event('adaptive_saved_configuration',config=spec,new_trees=0)
    for cid in spec['cases']:
        ref=next(r for r in source['cases'] if r['case']==cid);old=locked(ref['path'],ref['sha256'])
        full=next(m for m in old['models'] if m['family']=='ordinary' and m['block']==3)
        assert digest(full['artifact'])==full['sha256'];teacher=load_protected(full['artifact'])
        record=old['record'];data=data_before_test(record);task=old['task'];directory=out/cid;directory.mkdir(exist_ok=True)
        features=dict(train=data['train_x'],development=data['dev_x']);labels={'train':data['train_y']}
        folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028)
        view,_=decode_model(teacher);view=replace(view,values=contrast(view.values,task))
        design=view.design(data['train_x']);contribution=np.asarray(design@view.values);raw=teacher.raw(data['train_x']);hh=hessian(raw,task)
        row_weight,col_weight=risk_weights(data['train_y'],raw,task)
        ordinary_basis=functional_basis(contribution,task)[1];weighted,diagnostic=risk_basis(contribution,task,row_weight,col_weight)
        models=[];parity=[];pairs=[]
        for rank in spec['ranks']['multiclass' if task=='multiclass' else ('wide' if cid.startswith('bibtex') else 'multilabel')]:
            groups={}
            for name,basis in [('functional',ordinary_basis),('risk_basis',weighted),('risk_residual',weighted)]:
                group=prepare_training_group(view,basis[:,:rank],features,labels,raw,hh)
                group.order=all_support_order(group,data['train_y'],raw,'balanced_margin' if name=='risk_residual' else 'balanced_mse')
                groups[name]=group
            for fraction in spec['residual_fractions']:
                count=min(int(np.floor(fraction*view.values.size)),*(len(g.order) for g in groups.values()))
                same=[]
                for name,group in groups.items():
                    t=perf_counter();leaf,offset=group_model(group,group.order[:count]);model=pack([leaf],view.bias+offset)
                    uid=f'{name}/r{rank}-s{count}';p=directory/name/f'r{rank}-s{count}';p.mkdir(parents=True,exist_ok=True)
                    z=model.raw(data['dev_x']);ev=evaluate(z,data['dev_y'],task,folds,cfg,p)
                    m=serialize(model,z,ev,p,uid,dict(family=name,rank=rank,leaf_cap=8,residual_fraction=fraction,
                        requested_support=count,build_and_calibration_seconds=perf_counter()-t))
                    same.append(m);models.append(m)
                    group_raw=leaf.raw_from_design(group.designs['train'])+view.bias+offset
                    err=float(np.max(np.abs(model.raw(data['train_x'])-group_raw)));assert err<2e-5
                    parity.append(dict(uid=uid,error=err))
                costs=[m['cost_bytes'] for m in same]
                assert len(set(costs))==1,('matched count failed exact byte parity',cid,rank,count,costs)
                pairs.append(dict(rank=rank,fraction=fraction,support=count,bytes=costs[0],uids=[m['uid'] for m in same]))
                event('adaptive_saved_matched',case=cid,rank=rank,support=count,bytes=costs[0],
                    qualities={m['family']:quality_row(m['evaluations'],task)['quality'] for m in same})
            del groups;gc.collect()
        models.append(dict(full,uid='full_reference',family='ordinary'))
        case=dict(case=cid,task=task,record=record,models=models,reference_bytes=old['reference_bytes'],
            reference=old['reference'],matched=pairs,training_geometry=diagnostic,parity=parity)
        case['audit']=audit_case(case,data);file=directory/'case.json';write(file,case)
        entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit'],models=len(models)))
        write(out/'progress.json',dict(cases=entries));event('adaptive_saved_case_complete',case=cid,models=len(models),audit=case['audit'])
        del design,contribution;gc.collect();memory_guard()
    result=dict(status='adaptive_saved_complete',cases=entries,new_tree_fits=0,test_evaluated=False,environment=environment,
        config=spec,elapsed_seconds=perf_counter()-started,limitations=['same old training/D only; test not consulted',
        'risk weights are separable heuristics, not exact predictive curvature or an F1 theorem',
        'same topology/rank and actual bytes per comparison; all prescribed supports including harmful ones retained'])
    write(out/'report.json',result);event('adaptive_saved_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
