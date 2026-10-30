"""Frozen fresh-task external controls, then established shared-cache serving."""
from pathlib import Path
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event
from unified_external_core import data_before_test
from quality_worker import gpu_guard


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['adaptive_external'];source=locked(spec['manifest']['path'],spec['manifest']['sha256'])
    prerequisite=locked(spec['native']['path'],spec['native']['sha256']);assert prerequisite['status']=='adaptive_fresh_training_complete'
    records=[];fits=[];paid=0
    write(out/'external_method_freeze.json',dict(cases=spec['cases'],catboost_rounds=200,xgboost_rounds=64,depth=5,
        reason='count XGBoost scalar class trees; total native10560+CatBoost400+XGBoost1024 <=12000',
        native_paid=prerequisite['paid_new_trees'],new_test_evaluated=False))
    for cid in spec['cases']:
        cc=next(c for c in source['cases'] if f"{c['dataset']}_{c['seed']}"==cid);record=dict(case_id=cid,case=cc,dataset=source['datasets'][cc['dataset']]);data=data_before_test(record)
        for family in ('catboost','xgboost'):
            available=gpu_guard(cfg);directory=out/cid/family;directory.mkdir(parents=True,exist_ok=True);t=perf_counter();c=cc['outputs']
            if family=='catboost':
                from catboost import CatBoostClassifier
                full=CatBoostClassifier(iterations=200,depth=5,learning_rate=.1,l2_leaf_reg=1,loss_function='MultiClass',task_type='GPU',devices='0',
                    border_count=128,bootstrap_type='No',random_strength=0,boosting_type='Plain',random_seed=cc['seed'],thread_count=2,gpu_ram_part=.25,verbose=False,allow_writing_files=False)
                tree_count=200;counts=(10,20,40,80,120,200)
            else:
                import xgboost as xgb
                tree_count=64*c;counts=(10,20,40,64)
            if prerequisite['paid_new_trees']+paid+tree_count>12000:raise RuntimeError('A4 inclusive scalar-tree limit')
            paid+=tree_count;fits.append(dict(case=cid,family=family,paid_scalar_trees=tree_count,status='reserved'))
            write(out/'training_ledger.json',dict(native_paid=prerequisite['paid_new_trees'],external_paid=paid,attempts=fits))
            if family=='catboost':full.fit(data['train_x'],data['train_y'])
            else:
                full=xgb.train(dict(max_depth=5,eta=.1,reg_lambda=1,max_bin=128,subsample=1.,colsample_bytree=1.,objective='multi:softprob',num_class=c,
                    device='cuda',tree_method='hist',nthread=2,seed=cc['seed'],verbosity=1),xgb.DMatrix(data['train_x'],label=data['train_y']),num_boost_round=64)
            fits[-1].update(status='complete',seconds=perf_counter()-t,free_gpu_bytes_before=available)
            artifacts={}
            for rounds in counts:
                uid=f'd5-t{rounds}';path=directory/(uid+('.cbm' if family=='catboost' else '.ubj'))
                if family=='catboost':small=full.copy();small.shrink(rounds);small.save_model(str(path))
                else:small=full[:rounds];small.save_model(path)
                artifacts[uid]=dict(artifact=str(path),sha256=digest(path),native_bytes=path.stat().st_size,rounds=rounds,depth=5)
                del small
            records.append(dict(record,family=family,artifacts=artifacts,fits=[fits[-1]]));del full;gc.collect()
            write(out/'development_manifest.json',dict(status='fresh_external_development_complete',records=records,fits=fits))
            write(out/'training_ledger.json',dict(native_paid=prerequisite['paid_new_trees'],external_paid=paid,attempts=fits));event('fresh_external_fit',**fits[-1])
    cfg=dict(cfg);cfg['unified_external']=dict(cases=spec['cases'],batches=[1,32,256,1024],requests=32,
        source_manifest=dict(path=str(out/'development_manifest.json'),sha256=digest(out/'development_manifest.json')))
    from unified_external_serving import run as established_serving
    established_serving(cfg,out/'serving')
    # Target labels are read only after established serving seals D choices and predictions.
    report_path=out/'serving'/'report.json';served=json.loads(report_path.read_text());points=[]
    from unified_frontier_core import sufficient,metrics
    for row in served['test_predictions']:
        record=row['record'];assert digest(row['path'])==row['sha256']
        with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(record['dataset']['path'],allow_pickle=False) as f:y=f['target'][ids].copy()
        with np.load(row['path'],allow_pickle=False) as f:z=f['raw'].copy()
        control=row['control'];pred,loss,brier=sufficient(y,z,'multiclass',control['temperature'],control['shifts'])
        points.append(dict(case=row['case'],family=row['family'],quality=metrics(y,pred,loss,brier,'multiclass',z.shape[1]),control=control,source=row))
    result=dict(status='adaptive_external_complete',serving=dict(path=str(report_path),sha256=digest(report_path)),test_points=points,
        paid_scalar_trees=paid,native_paid_new_trees=prerequisite['paid_new_trees'],inclusive_a4_tree_work=paid+prerequisite['paid_new_trees'],
        ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')),timings=served['timings'],storage=served['storage'],
        limitations=['two fresh multiclass tasks only; no native multilabel TabPFN comparison','XGBoost64rounds fixed by total scalar-tree allowance, not an exhaustive strong-model search',
        'native-library serialized model cost differs from live numeric TabPFN buffers and process RSS'])
    write(out/'report.json',result);event('adaptive_external_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
