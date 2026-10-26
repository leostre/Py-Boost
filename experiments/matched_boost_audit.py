"""Replay complete native prefix/cost/threshold table before test evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.uniform_threshold_core import ThresholdModel,threshold_variants,choice_key
from experiments.shared_leaf_basis import LeafEnsemble,reference_class_model
from experiments.closed_budget_study import empty_model
from experiments.parallel_leaf_core import prefix
from experiments.history_basis_study import raw
from experiments.output_calibration_core import calibrated_quality
from experiments.cumulative_budget_audit import quality_matches


def verify_cohort(records,cases,methods):
    expected={(c,m) for c in cases for m in methods}
    actual=[(r['case_id'],r['family']) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise RuntimeError('native matched cohort incomplete or duplicate')


def matched_boost_final_test(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_matched_boost_audit.py'],cwd=ROOT,check=True,timeout=300)
    locks={}
    def lock(path,expected):
        if sha256(path)!=expected: raise RuntimeError('native matched input changed')
        locks[str(path)]=expected
        return Path(path)
    dp=lock(config['matched_development_manifest'],config['matched_development_sha256'])
    rp=lock(config['matched_manifest'],config['matched_manifest_sha256'])
    source=json.loads(dp.read_text(encoding='utf-8'))
    manifest=json.loads(rp.read_text(encoding='utf-8'))
    verify_cohort(source['records'],config['matched_cases'],config['matched_methods'])
    new_trees=reused_trees=0
    evidence=[]
    with threadpool_limits(limits=1):
        for record in source['records']:
            case=next(c for c in manifest['cases'] if f'{c["dataset"]}_{c["seed"]}'==record['case_id'])
            if case!=record['case']: raise RuntimeError('native role metadata changed')
            data=manifest['datasets'][case['dataset']]
            lock(data['path'],data['sha256'])
            lock(case['split_file'],case['split_sha256'])
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            fits={}
            for fit in record['fits']:
                dense=LeafEnsemble.load(lock(fit['artifact'],fit['sha256']))
                if dense.n_trees!=fit['trees'] or fit['trees']!=200: raise RuntimeError('native fit price mismatch')
                fits[fit['depth']]=empty_model(reference_class_model(dense))
                if fit['reused']: reused_trees+=fit['trees']
                else: new_trees+=fit['trees']
            if set(fits)!={3,4,5}: raise RuntimeError('native depth grid incomplete')
            rebuilt=[]
            for identity,item in record['artifacts'].items():
                model=ThresholdModel.load(lock(item['artifact'],item['sha256']))
                expected=prefix(fits[item['depth']],item['rounds'])
                for name in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'):
                    np.testing.assert_array_equal(getattr(model.base,name),getattr(expected.base,name))
                if (model.payload_bytes!=item['coo_bytes'] or model.prepare_inference().resident_bytes!=item['csr_bytes']
                        or model.parameter_count!=item['parameters'] or model.base.n_trees!=item['trees']):
                    raise RuntimeError('native stored array cost does not replay')
                prediction=record['development_predictions'][identity]
                lock(prediction['artifact'],prediction['sha256'])
                with np.load(prediction['artifact'],allow_pickle=False) as f:
                    variants=threshold_variants(f['calibration'],y[roles['calibration']],f['selection'],y[roles['selection']],
                        case['task'],config['decision_grid'],config['decision_shrinkage'])
                rebuilt.extend(dict(candidate_id=identity,parameters=model.parameter_count,payload_bytes=model.payload_bytes,**v)
                               for v in variants)
            if len(rebuilt)!=len(record['candidates']): raise RuntimeError('native candidate count changed')
            for actual,old in zip(rebuilt,record['candidates']):
                if {k:v for k,v in actual.items() if k!='validation'}!={k:v for k,v in old.items() if k!='validation'}:
                    raise RuntimeError('native frozen threshold settings changed')
                quality_matches(actual['validation'],old['validation'])
            if min(record['candidates'],key=choice_key)!=record['chosen']: raise RuntimeError('native selection changed')
            model=ThresholdModel.load(lock(record['selected_artifact'],record['selected_sha256']))
            np.testing.assert_array_equal(model.decision_shifts,np.array(record['chosen']['decision_shifts'],np.float32))
            z=raw(model,x[roles['selection']])
            with np.load(record['development_predictions'][record['chosen']['candidate_id']]['artifact']) as f:
                np.testing.assert_allclose(z,f['selection'],atol=2e-7,rtol=1e-7)
            q=calibrated_quality(y[roles['selection']],z,case['task'],model.decision_shifts)
            quality_matches(q,record['chosen']['validation'])
            evidence.append(dict(case_id=record['case_id'],family=record['family'],candidate_rows=len(rebuilt),
                                 selection_quality=q,test_evaluated=False))
            emit('matched_native_family_audited',case_id=record['case_id'],family=record['family'],test_evaluated=False)
        if new_trees!=source['new_tree_fits'] or reused_trees!=source['reused_tree_fits'] or new_trees>config['matched_tree_cap']:
            raise RuntimeError('native aggregate tree ledger mismatch')
        seal=output/'pretest_audit.json'
        write_report(seal,dict(status='audit_passed',records=evidence,input_locks=locks,new_tree_fits=new_trees,
            reused_tree_fits=reused_trees,new_fits=0,test_evaluated=False))
        sealed=sha256(seal)
        results=[]
        for record in source['records']:
            case=record['case']
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            model=ThresholdModel.load(record['selected_artifact'])
            z=raw(model,x[roles['test']])
            q=calibrated_quality(y[roles['test']],z,case['task'],model.decision_shifts)
            directory=output/record['case_id']/record['family']
            directory.mkdir(parents=True)
            path=directory/'test_predictions.npz'
            np.savez_compressed(path,raw=z,target=y[roles['test']],indices=roles['test'],decision_shifts=model.decision_shifts)
            results.append(dict(case_id=record['case_id'],family=record['family'],quality=q,chosen=record['chosen'],
                artifact=record['selected_artifact'],sha256=record['selected_sha256'],prediction_file=str(path),
                prediction_sha256=sha256(path),coo_bytes=model.payload_bytes,csr_bytes=model.prepare_inference().resident_bytes,
                parameters=model.parameter_count,trees=model.base.n_trees,leaves=len(model.base.coefficients)))
        if sha256(seal)!=sealed or any(sha256(p)!=h for p,h in locks.items()):
            raise RuntimeError('native frozen inputs changed during test')
    report.update(status='complete',results=results,new_fits=0,new_trees=0,pretest_audit_sha256=sealed,
                  source_development_sha256=sha256(dp),test_is_reused_exploratory=True)
    emit('matched_native_frozen_test_complete',records=len(results),new_fits=0,new_trees=0)
