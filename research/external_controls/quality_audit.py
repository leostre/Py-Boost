"""Immutable model reload/candidate replay; whole-cohort audit before any test."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import numpy as np
from external_quality_core import canonical_raw,calibrated_rows,choose,tree_counts,quality_close,audit_then_test
from experiments.output_calibration_core import calibrated_quality
from matched_data import digest
from quality_worker import load_inputs,score_native,gpu_guard,write


def chosen_item(record):
    return record['artifacts'][record['chosen']['candidate_id']]


def restore(record,cfg):
    item=chosen_item(record)
    if digest(item['artifact'])!=item['sha256']: raise RuntimeError('selected model changed')
    family=record['family']
    if family=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(item['artifact'])
    if family=='xgboost':
        import xgboost as xgb
        model=xgb.Booster(model_file=item['artifact'])
        model.set_param(dict(device='cuda',nthread=2))
        return model
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    weights=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']/cfg['weights_filename']
    if digest(weights)!=cfg['weights_sha256']: raise RuntimeError('checkpoint changed')
    return TabPFNClassifier.load_from_fit_state(item['artifact'],device='cuda')


def predict(model,record,x):
    if record['family'] in ('catboost','xgboost'):
        return score_native(model,record['family'],x,record['case']['outputs'])
    return canonical_raw(model.predict_proba(x),model.classes_,record['case']['outputs'],probabilities=True)


def replay_candidates(record,data):
    rebuilt=[]
    for identity,item in record['artifacts'].items():
        if digest(item['artifact'])!=item['sha256']: raise RuntimeError('candidate model changed')
        saved=record['development_predictions'][identity]
        if digest(saved['artifact'])!=saved['sha256']: raise RuntimeError('development logits changed')
        row=next(r for r in record['candidates'] if r['candidate_id']==identity)
        cost=dict(payload_bytes=row['payload_bytes'],parameters=row['parameters'])
        if record['family'] in ('catboost','xgboost'):
            if cost['payload_bytes']!=Path(item['artifact']).stat().st_size+4*record['case']['outputs']:
                raise RuntimeError('native serialized cost mismatch')
            if cost['parameters']!=item['counts']['leaf_parameters']+record['case']['outputs']:
                raise RuntimeError('native coefficient count mismatch')
        else:
            if cost['payload_bytes']!=item['context_file_bytes']+item['shared_checkpoint_bytes']+4*record['case']['outputs']:
                raise RuntimeError('context serialized cost mismatch')
        with np.load(saved['artifact'],allow_pickle=False) as f:
            rebuilt.extend(calibrated_rows(f['calibration'],data['calibration'][1],f['selection'],data['selection'][1],
                'multiclass',identity,cost,GRID,SHRINKAGES))
    if len(rebuilt)!=len(record['candidates']): raise RuntimeError('candidate grid incomplete')
    for actual,expected in zip(rebuilt,record['candidates']):
        if {k:v for k,v in actual.items() if k!='validation'}!={k:v for k,v in expected.items() if k!='validation'}:
            raise RuntimeError('frozen candidate settings changed')
        quality_close(actual['validation'],expected['validation'])
    # Reproduce ranking from frozen, validated rows rather than re-rounded BLAS.
    if choose(record['candidates'])!=record['chosen']: raise RuntimeError('frozen selection changed')


GRID=[0,.25,-.25,.5,-.5,1,-1,2,-2]
SHRINKAGES=[.25,.5,1]


def child(cfg,record,directory,mode,audit_path=None):
    directory.mkdir(parents=True,exist_ok=True)
    case,entry,data=load_inputs(cfg,record['case_id'])
    if case!=record['case'] or entry!=record['dataset']: raise RuntimeError('record data identity mismatch')
    if cfg['decision_grid']!=GRID or cfg['decision_shrinkage']!=SHRINKAGES:
        raise RuntimeError('frozen threshold protocol changed')
    gpu_guard(cfg)
    started=perf_counter()
    if mode=='pretest': replay_candidates(record,data)
    else:
        if audit_path is None: raise RuntimeError('whole-cohort audit missing')
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        expected={(r['case_id'],r['family']) for r in audit['records']}
        if (record['case_id'],record['family']) not in expected or audit['development_sha256']!=cfg['quality_development_sha256']:
            raise RuntimeError('test not authorized by frozen audit')
        for path,value in audit['input_locks'].items():
            if digest(path)!=value: raise RuntimeError('audited input changed before test')
    model=restore(record,cfg)
    if mode=='pretest':
        raw_selection=predict(model,record,data['selection'][0])
        saved=record['development_predictions'][record['chosen']['candidate_id']]
        with np.load(saved['artifact']) as f: expected=f['selection'].copy()
        np.testing.assert_allclose(raw_selection,expected,atol=2e-4,rtol=2e-5)
        q=calibrated_quality(data['selection'][1],raw_selection,'multiclass',np.array(record['chosen']['decision_shifts'],np.float32))
        quality_close(q,record['chosen']['validation'])
        item=chosen_item(record)
        if record['family']=='xgboost':
            actual=tree_counts([json.loads(s) for s in model.get_dump(dump_format='json')])
            if actual!=item['counts'] or model.num_boosted_rounds()!=item['rounds']:
                raise RuntimeError('native tree ledger changed')
        if record['family']=='catboost':
            if int(model.tree_count_)!=item['counts']['trees'] or len(model.get_leaf_values())!=item['counts']['leaf_parameters']:
                raise RuntimeError('CatBoost structure count changed')
        result=dict(case_id=record['case_id'],family=record['family'],status='audit_passed',
            max_raw_replay_error=float(np.max(np.abs(raw_selection-expected))),quality=q,
            candidate_rows_replayed=len(record['candidates']),test_evaluated=False,new_fits=0)
    else:
        # This is the first access to test targets in the external pipeline.
        with np.load(case['split_file'],allow_pickle=False) as p: indices=p['test'].copy()
        with np.load(entry['path'],allow_pickle=False) as f:
            x,y=f['features'][indices].copy(),f['target'][indices].copy()
        z=predict(model,record,x)
        shifts=np.array(record['chosen']['decision_shifts'],np.float32)
        q=calibrated_quality(y,z,'multiclass',shifts)
        predpath=directory/'test_predictions.npz'
        np.savez_compressed(predpath,raw=z,target=y,indices=indices,decision_shifts=shifts)
        result=dict(case_id=record['case_id'],family=record['family'],quality=q,chosen=record['chosen'],
            model=chosen_item(record),prediction_file=str(predpath),prediction_sha256=digest(predpath),
            elapsed_seconds=perf_counter()-started,new_fits=0,test_is_reused_exploratory=True)
    result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(directory/(mode+'.json'),result)
    print(json.dumps(dict(event='external_'+mode,case_id=record['case_id'],family=record['family'],
        quality=result['quality'],new_fits=0)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--case')
    parser.add_argument('--family')
    parser.add_argument('--mode',choices=['pretest','test'])
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    dp=Path(cfg['quality_development_manifest'])
    if digest(dp)!=cfg['quality_development_sha256']: raise RuntimeError('external development seal changed')
    source=json.loads(dp.read_text(encoding='utf-8'))
    records=source['records']
    if args.case:
        record=next(r for r in records if r['case_id']==args.case and r['family']==args.family)
        child(cfg,record,args.output,args.mode,args.audit)
        return
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_external_quality_core.py','tests/test_quality_audit.py'],check=True,timeout=300)
    expected={(c,f) for c in cfg['quality_cases'] for f in cfg['quality_families']}
    if len(records)!=len(expected) or {(r['case_id'],r['family']) for r in records}!=expected:
        raise RuntimeError('external cohort incomplete or duplicated')
    locks={str(dp):digest(dp),cfg['matched_manifest']:cfg['matched_manifest_sha256']}
    for record in records:
        locks[record['dataset']['path']]=record['dataset']['sha256']
        locks[record['case']['split_file']]=record['case']['split_sha256']
        for group in ('artifacts','development_predictions'):
            for item in record[group].values(): locks[item['artifact']]=item['sha256']
    def worker(record,mode,audit=None):
        path=args.output/record['case_id']/record['family']
        command=[sys.executable,__file__,str(args.config),str(path),'--case',record['case_id'],'--family',record['family'],'--mode',mode]
        if audit: command.extend(['--audit',str(audit)])
        subprocess.run(command,check=True,timeout=cfg['worker_timeout_seconds'])
        return json.loads((path/(mode+'.json')).read_text(encoding='utf-8'))
    audit_path=args.output/'pretest_audit.json'
    def seal(evidence):
        if any(digest(p)!=h for p,h in locks.items()): raise RuntimeError('frozen input changed during audit')
        write(audit_path,dict(status='audit_passed',records=evidence,input_locks=locks,
            development_sha256=digest(dp),new_fits=0,test_evaluated=False))
        return audit_path
    evidence,results=audit_then_test(records,lambda r:worker(r,'pretest'),seal,lambda r,a:worker(r,'test',a))
    if any(digest(p)!=h for p,h in locks.items()): raise RuntimeError('frozen input changed during test')
    write(args.output/'report.json',dict(status='complete',results=results,new_fits=0,nn_weight_updates=0,
        pretest_audit_sha256=digest(audit_path),source_development_sha256=digest(dp),
        test_is_reused_exploratory=True,benchmark_scope='fiveprimarytasks; not100independenttasks'))
    print(json.dumps(dict(event='external_frozen_test_complete',cases=len(results),new_fits=0)),flush=True)


if __name__=='__main__': main()
