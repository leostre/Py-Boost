"""Audit frozen development routes, then test them without rerouting."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from matched_data import digest
from quality_worker import load_inputs,write
from external_quality_core import quality_close
from external_bank_core import route_outcomes
from experiments.uniform_threshold_core import ThresholdModel
from experiments.output_calibration_core import calibrated_quality


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2])
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_mixed_quality_gate.py'],check=True,timeout=300)
    bp=Path(cfg['real_bank_manifest'])
    if digest(bp)!=cfg['real_bank_manifest_sha256']: raise RuntimeError('frozen task routing changed')
    bank=json.loads(bp.read_text(encoding='utf-8'))
    locks={str(bp):digest(bp),**bank['input_locks']}
    evidence=[]
    with threadpool_limits(limits=1):
        for row in [*bank['records'],*bank['specialists']]:
            case,entry,data=load_inputs(cfg,row['case_id'])
            if digest(row['path'])!=row['sha256']: raise RuntimeError('frozen routed model changed')
            locks[row['path']]=row['sha256']
            model=ThresholdModel.load(row['path'])
            z=model.prepare_inference().raw(data['selection'][0])
            q=calibrated_quality(data['selection'][1],z,'multiclass',model.decision_shifts)
            quality_close(q,row['validation'])
            cost=max(model.payload_bytes,model.prepare_inference().resident_bytes)
            if cost!=row['serving_bytes']: raise RuntimeError('routed model array cost mismatch')
            evidence.append(dict(case_id=row['case_id'],identity=row.get('family',row.get('candidate_id')),
                quality=q,serving_bytes=cost,test_evaluated=False))
        if any(digest(p)!=h for p,h in locks.items()): raise RuntimeError('routed input changed before seal')
        seal=output/'pretest_audit.json'
        write(seal,dict(status='audit_passed',records=evidence,routes=bank['specialist_subsets'],
            input_locks=locks,new_fits=0,test_evaluated=False))
        seal_hash=digest(seal)
        # First use of test evidence occurs only after the full route audit.
        tp=Path(cfg['external_test_report'])
        if digest(tp)!=cfg['external_test_report_sha256']: raise RuntimeError('frozen TFM test source changed')
        test=json.loads(tp.read_text(encoding='utf-8'))
        locks[str(tp)]=digest(tp)
        references={}
        for case_id in cfg['quality_cases']:
            case,entry,_=load_inputs(cfg,case_id)
            row=next(r for r in test['results'] if r['case_id']==case_id and r['family']=='tabpfn_auto')
            if digest(row['prediction_file'])!=row['prediction_sha256']: raise RuntimeError('frozen TFM prediction changed')
            locks[row['prediction_file']]=row['prediction_sha256']
            with np.load(case['split_file']) as f: indices=f['test'].copy()
            with np.load(entry['path']) as f: target=f['target'][indices].copy()
            with np.load(row['prediction_file']) as f:
                np.testing.assert_array_equal(f['indices'],indices)
                np.testing.assert_array_equal(f['target'],target)
                q=calibrated_quality(target,f['raw'],'multiclass',f['decision_shifts'])
            quality_close(q,row['quality'])
            references[case_id]=q
        specialists=[]
        for row in bank['specialists']:
            case,entry,_=load_inputs(cfg,row['case_id'])
            with np.load(case['split_file']) as f: indices=f['test'].copy()
            with np.load(entry['path']) as f: x,y=f['features'][indices].copy(),f['target'][indices].copy()
            model=ThresholdModel.load(row['path'])
            q=calibrated_quality(y,model.prepare_inference().raw(x),'multiclass',model.decision_shifts)
            specialists.append(dict(row,quality=q))
        results=route_outcomes(references,specialists,bank['specialist_subsets'])
        if digest(seal)!=seal_hash or any(digest(p)!=h for p,h in locks.items()):
            raise RuntimeError('routed source changed during test')
    serving_eligible=any(r['routed_tasks']>0 and r['all_quality_pass'] for r in results)
    write(output/'report.json',dict(status='complete',results=results,specialists=specialists,references=references,
        serving_gate_passed=serving_eligible,new_fits=0,new_trees=0,nn_weight_updates=0,
        pretest_audit_sha256=seal_hash,source_bank_sha256=digest(bp),
        test_is_reused_exploratory=True,routes_changed_after_test=False,
        memory_scope='additional standalone coefficient arrays only; not measured mixed runtime/process memory'))
    print(json.dumps(dict(event='mixed_quality_gate_complete',serving_gate_passed=serving_eligible,
        subsets=[dict(extra_budget_bytes=r['extra_budget_bytes'],routed=r['routed_tasks'],
            routed_passes=r['routed_quality_passes'],all_quality_pass=r['all_quality_pass']) for r in results],new_fits=0)),flush=True)


if __name__=='__main__': main()
