"""Evaluate the frozen fifteen-case transfer seal; selected points only."""
from pathlib import Path
from dataclasses import asdict
import gc,json,subprocess,sys
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_core import fitted_control
from abl_quality_evaluation import table
from audit_abl_transfer import independent_task_quality,audit_final
from audit_abl_quality import guard
from experiments.shared_leaf_basis import probabilities
from budget_quality_core import quality as multiclass_quality
from experiments.output_calibration_core import calibrated_quality


def paired_interval(y,a,b,task,repeats=500):
    """Fixed policies, paired rows, complete output domain; no selection."""
    if task=='multiclass':
        from budget_quality_core import paired_bootstrap
        return paired_bootstrap(y,a,b,repeats=repeats,seed=20261026)
    stats=[]
    for z,t,shifts in (a,b):
        assert t==1.
        p=probabilities(z,task);pred=z-np.asarray(shifts)>=0;truth=np.asarray(y,bool)
        loss=-(truth*np.log(p)+(1-truth)*np.log1p(-p)).mean(axis=1)
        stats.append((pred.astype(float),(pred&truth).astype(float),loss))
    rng=np.random.default_rng(20261026);deltas=[];truth=np.asarray(y,float)
    for _ in range(repeats):
        w=rng.multinomial(len(y),np.full(len(y),1/len(y)));support=w@truth;values=[]
        for pred,tp,loss in stats:
            denom=support+w@pred;f1=np.divide(2*(w@tp),denom,out=np.zeros(y.shape[1]),where=denom>0)
            values.append((float(w@loss/w.sum()),float(f1.mean())))
        deltas.append([values[0][0]-values[1][0],100*(values[0][1]-values[1][1])])
    bounds=np.quantile(deltas,[.025,.975],axis=0).T
    return dict(repeats=repeats,seed=20261026,delta_loss_ci95=bounds[0].tolist(),delta_f1_pp_ci95=bounds[1].tolist(),
        scope='conditional fixed-model paired rows; not independent-task uncertainty',support=np.asarray(y).sum(0).tolist())


def evaluate(entry,seal_sha,out):
    case=locked(entry['path'],entry['sha256']);directory=out/'evaluated'/case['case_id'];directory.mkdir(parents=True,exist_ok=True)
    for item,k,sha in ((case['record']['case'],'split_file','split_sha256'),(case['record']['dataset'],'path','sha256')):
        if digest(item[k])!=item[sha]:raise RuntimeError('transfer source changed before test')
    with np.load(case['record']['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(case['record']['dataset']['path'],allow_pickle=False) as f:
        x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
    target=directory/'target.npz';np.savez_compressed(target,target=y,indices=ids)
    wanted={c['chosen']['uid'] for c in (*case['choices'],*case['full_reference_choices']) if c['chosen']}
    wanted|={case['metas'][uid]['parent_uid'] for uid in list(wanted) if 'parent_uid' in case['metas'][uid]}
    raw={};predictions={}
    for uid in sorted(wanted):
        m=case['metas'][uid];source=m['source']
        if digest(source['artifact'])!=source['sha256']:raise RuntimeError('selected transfer model changed')
        z=native_load(source['artifact']).prepare_inference().raw(x);path=directory/(uid.replace('/','-')+'.npz')
        np.savez_compressed(path,raw=z);raw[uid]=z;predictions[uid]=dict(path=str(path),sha256=digest(path))
    fit={r['uid']:r for r in case['rows']};metrics={}
    def measured(uid,policy):
        cp=asdict(fitted_control(fit[uid],policy));key=(uid,policy)
        if key not in metrics:
            metrics[key]=(multiclass_quality(y,raw[uid],cp['temperature'],cp['shifts']) if case['task']=='multiclass'
                else calibrated_quality(y,raw[uid],'multilabel',np.asarray(cp['shifts'])))
        return metrics[key],cp
    def result(c):
        p={k:v for k,v in c.items() if k!='chosen'};chosen=c['chosen']
        if chosen is None:return dict(p,status='no_feasible_candidate')
        uid=chosen['uid'];m=case['metas'][uid];q,cp=measured(uid,chosen['policy'])
        p.update(status='evaluated',uid=uid,policy=chosen['policy'],control=cp,test=q,
            validation=chosen['validation'],cost_bytes=m['cost_bytes'],parameters=m['ledger']['parameters'],
            family=m['family'],rank=m['rank'],nnz=m['nnz'],trees=m['trees'],leaves=m['leaves'],nodes=m['nodes'])
        if 'parent_uid' in m:
            parent,_=measured(m['parent_uid'],chosen['policy'])
            p.update(parent_uid=m['parent_uid'],own_parent_test=parent,own_parent_guard_5percent=guard(q,parent),
                own_parent_memory_fraction=m['cost_bytes']/case['metas'][m['parent_uid']]['cost_bytes'])
        return p
    full=[result(c) for c in case['full_reference_choices']];points=[result(c) for c in case['choices']]
    by={(p['mode'],p['objective'],p['group'],p['budget_ratio']):p for p in points};intervals=[];pairs={}
    for p in points:
        if p['status']!='evaluated':continue
        ref=next(c for c in full if c['mode']==p['mode'] and c['objective']==p['objective'])
        p['full_reference_guard_5percent']=guard(p['test'],ref['test']);p['full_reference_uid']=ref['uid']
        ordinary=by[p['mode'],p['objective'],'native',p['budget_ratio']]
        if ordinary['status']!='evaluated':continue
        p['versus_native']=dict(reference_uid=ordinary['uid'],delta_loss=p['test']['loss']-ordinary['test']['loss'],
            delta_f1_macro_pp=100*(p['test']['f1_macro']-ordinary['test']['f1_macro']))
        if p['mode']=='raw' or p['objective']!='f1_macro' or p['budget_ratio'] not in (.75,1.) or p['group']=='native':continue
        key=(p['uid'],p['policy'],ordinary['uid'],ordinary['policy'])
        if key not in pairs:
            a,b=p['control'],ordinary['control']
            interval=paired_interval(y,(raw[p['uid']],a['temperature'],a['shifts']),
                (raw[ordinary['uid']],b['temperature'],b['shifts']),case['task'])
            pairs[key]=len(intervals);intervals.append(dict(id=len(intervals),left_uid=p['uid'],right_uid=ordinary['uid'],
                left_control=a,right_control=b,**interval))
        p['versus_native']['interval_id']=pairs[key]
    result=dict(case=case['case_id'],task=case['task'],points=points,full_reference_points=full,bootstrap_intervals=intervals,
        target=dict(path=str(target),sha256=digest(target)),predictions=predictions,development=entry,
        pretest_seal_sha256=seal_sha,test_used_for_selection=False)
    path=directory/'evaluation.json';write(path,result)
    table(directory/'points.csv',[{**{k:v for k,v in p.items() if isinstance(v,(str,int,float,bool)) or v is None},
        **{'test_'+k:v for k,v in p.get('test',{}).items()}} for p in points])
    event('abl_transfer_test_case_complete',case=case['case_id'],path=str(path),sha256=digest(path))
    del raw;gc.collect()
    return dict(case=case['case_id'],path=str(path),sha256=digest(path),points=len(points))


def run(cfg,out):
    source=cfg['abl_transfer_development_completed'];dev=locked(source['path'],source['sha256']);root=Path(source['path']).parent
    if dev['status']!='transfer_development_complete' or dev['test_accessed']:raise RuntimeError('completed transfer development required')
    seal=locked(root/'pretest_choices.json',dev['pretest_seal_sha256'])
    audit=locked(root/'development_audit.json',seal['development_audit_sha256'])
    if seal['status']!='transfer_choices_sealed_before_test' or len(seal['cases'])!=15 or audit['status']!='passed':
        raise RuntimeError('whole-study transfer seal and audit required')
    from abl_quality_worker import check_native_environment
    check_native_environment(cfg)
    result=dict(status='complete',source=source,evaluations=[evaluate(e,dev['pretest_seal_sha256'],out) for e in seal['cases']],
        new_tree_fits=0,gpu_execution=False,training_ledger=dev['training_ledger'],
        scope='posthoc transfer; nine old multilabel splits, six fresh multiclass splits; five tasks, not15 independent datasets')
    audit_path=out/'audit_report.json';write(audit_path,audit_final(result))
    result['audit']=dict(path=str(audit_path),sha256=digest(audit_path),status='passed')
    path=out/'report.json';write(path,result);event('abl_transfer_final_complete',path=str(path),sha256=digest(path))


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2])
    native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    if '--native' not in sys.argv:
        subprocess.run([native,__file__,sys.argv[1],sys.argv[2],'--native'],check=True,timeout=21600);return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(cfg,out)


if __name__=='__main__':main()
