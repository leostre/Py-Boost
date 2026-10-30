"""Bounded2percent/W40 solver check; same supports, ranks, trees, data and loss."""
from pathlib import Path
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from experiments.sparse_leaf_core import leaf_design,contrast
from adaptive_basis_core import risk_basis,risk_weights
from adaptive_controller_core import extend_basis
from unified_training import serialize,evaluate,quality_row
from unified_frontier_core import make_folds,sufficient,metrics
from unified_recalibration import audit_case
from window_reserve_core import loss_gradient
from window_reserve_study import replace_residual,rank_variant,topology_signature
from reserve_precision_core import refine_support,refine_rank


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();spec=cfg['reserve_precision'];source=locked(spec['source']['path'],spec['source']['sha256'])
    assert source['status']=='window_reserve_complete';environment=check_native_environment(cfg);entries=[];records=[]
    for ref in source['cases']:
        case=locked(ref['path'],ref['sha256']);cid=case['case'];data=data_before_test(case['record']);task=case['task'];directory=out/cid;directory.mkdir(exist_ok=True)
        bmeta=next(m for m in case['models'] if m['uid']=='baseline');baseline=load_protected(bmeta['artifact']);index=next(i for i,b in enumerate(baseline.blocks) if b.base.representation=='low_rank')
        block=baseline.blocks[index];c=len(baseline.bias);r=block.base.basis.shape[1];design=leaf_design(block.base.leaf_indices(data['train_x']),len(block.base.coefficients));raw=baseline.raw(data['train_x'])
        folds=make_folds(data['dev_y'],task,case['record']['case']['seed']+20261028);models=[]
        targets=[m for m in case['models'] if m.get('fraction')==.02 and (m.get('window')==40 or m['family']=='rank')]
        for old in targets:
            assert digest(old['artifact'])==old['sha256'];coarse=load_protected(old['artifact']);oldblock=coarse.blocks[index]
            if old['family']=='rank':
                p=old['rank']-r
                if oldblock.base.representation=='low_rank':b=oldblock.base.basis
                else:
                    _,g,_=loss_gradient(raw,data['train_y'],task);rw,cw=risk_weights(data['train_y'],raw,task);b=extend_basis(block.base.basis,risk_basis(-g,task,rw,cw)[0])[:,:r+p];b[:,:r]=block.base.basis
                extra=b[:,r:].astype(float)
                if oldblock.base.representation=='low_rank':initial=oldblock.base.coefficients[:,r:].astype(float)
                else:
                    v=oldblock.base.coefficients.astype(float)
                    if oldblock.base.representation=='reference_class':v=np.column_stack((v,np.zeros(len(v))))
                    delta=contrast(v-block.base.coefficients.astype(float)@block.base.basis.T,task);initial=delta@extra
                values,fit=refine_rank(design,extra,initial,raw,data['train_y'],task,cfg['lambda_l2'],spec['maxiter']);candidate=rank_variant(baseline,index,b,values)
            else:
                keys=oldblock.rows.astype(np.int64)*c+oldblock.columns
                values,fit=refine_support(design,keys,oldblock.values.astype(float),raw,data['train_y'],task,cfg['lambda_l2'],spec['maxiter']);candidate=replace_residual(baseline,index,keys,values)
                np.testing.assert_array_equal(candidate.blocks[index].base.coefficients,block.base.coefficients)
                np.testing.assert_array_equal(candidate.blocks[index].base.basis,block.base.basis)
            assert topology_signature(candidate)==topology_signature(baseline);assert block_cost(candidate)['cost_bytes']<=old['byte_ceiling']
            path=directory/old['uid'];path.mkdir(exist_ok=True);z=candidate.raw(data['dev_x']);ev=evaluate(z,data['dev_y'],task,folds,cfg,path)
            item=serialize(candidate,z,ev,path,old['uid'],dict(family=old['family'],fraction=.02,window=old.get('window'),source=old,fit=fit,byte_ceiling=old['byte_ceiling']))
            models.append(item);records.append(dict(case=cid,uid=old['uid'],fit=fit));event('reserve_precision_model',case=cid,uid=old['uid'],all_converged=fit['all_converged'],maximum_gradient=fit['maximum_gradient'])
        result=dict(case=cid,task=task,record=case['record'],models=models);result['audit']=audit_case(result,data);file=directory/'case.json';write(file,result)
        entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=result['audit']));write(out/'progress.json',dict(cases=entries,records=records));gc.collect();memory_guard()
    write(out/'pretest_seal.json',dict(cases=entries,source=spec['source'],tree_training=False,changed_supports=False))
    points=[]
    for ref in entries:
        case=locked(ref['path'],ref['sha256']);record=case['record'];directory=out/case['case']
        with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(record['dataset']['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        target=directory/'targets.npz';np.savez_compressed(target,y=y,rows=ids)
        for i,m in enumerate(case['models']):
            z=load_protected(m['artifact']).raw(x);control=quality_row(m['evaluations'],case['task'])['refit_control'];pred,loss,brier=sufficient(y,z,case['task'],control['temperature'],control['shifts'])
            file=directory/f'prediction{i:03d}.npz';np.savez_compressed(file,raw=z,prediction=pred,loss=loss,brier=brier)
            points.append(dict(case=case['case'],uid=m['uid'],task=case['task'],family=m['family'],quality=metrics(y,pred,loss,brier,case['task'],z.shape[1]),
                development=quality_row(m['evaluations'],case['task'])['quality'],control=control,cost_bytes=m['cost_bytes'],outputs=z.shape[1],
                raw=dict(path=str(file),sha256=digest(file)),targets=dict(path=str(target),sha256=digest(target))))
    report=dict(status='reserve_precision_complete',cases=entries,points=points,records=records,new_trees=0,environment=environment,
        elapsed_seconds=perf_counter()-started,seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),
        limitations=['only frozen2percent/W40 point; other coarse curves retain60iteration limit','same support/objective, higher coefficient precision only','old tests exploratory'])
    write(out/'report.json',report);event('reserve_precision_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),models=len(points),new_trees=0)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
