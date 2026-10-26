"""CPU-only exploratory decomposition of two immutable specialist failures."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from matched_data import digest
from quality_worker import load_inputs,write
from external_quality_core import quality_close
from experiments.uniform_threshold_core import ThresholdModel
from experiments.output_calibration_core import calibrated_quality
from specialist_loss_core import decompose


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2])
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_specialist_loss_core.py'],check=True,timeout=300)
    gp=Path(cfg['specialist_gate_report'])
    tp=Path(cfg['external_test_report'])
    if digest(gp)!=cfg['specialist_gate_sha256'] or digest(tp)!=cfg['external_test_report_sha256']:
        raise RuntimeError('frozen test result changed')
    gate=json.loads(gp.read_text(encoding='utf-8'))
    test=json.loads(tp.read_text(encoding='utf-8'))
    locks={str(gp):digest(gp),str(tp):digest(tp)}
    results=[]
    with threadpool_limits(limits=1):
        for row in gate['specialists']:
            case,entry,_=load_inputs(cfg,row['case_id'])
            for path,checksum in ((row['path'],row['sha256']),(entry['path'],entry['sha256']),
                                  (case['split_file'],case['split_sha256'])):
                if digest(path)!=checksum: raise RuntimeError('diagnostic input changed')
                locks[path]=checksum
            with np.load(case['split_file']) as f: indices=f['test'].copy()
            with np.load(entry['path']) as f: x,y=f['features'][indices].copy(),f['target'][indices].copy()
            source=next(r for r in test['results'] if r['case_id']==row['case_id'] and r['family']=='tabpfn_auto')
            if digest(source['prediction_file'])!=source['prediction_sha256']: raise RuntimeError('cached predictions changed')
            locks[source['prediction_file']]=source['prediction_sha256']
            with np.load(source['prediction_file']) as f:
                np.testing.assert_array_equal(f['indices'],indices)
                np.testing.assert_array_equal(f['target'],y)
                reference,reference_shifts=f['raw'].copy(),f['decision_shifts'].copy()
            model=ThresholdModel.load(row['path'])
            raw=model.prepare_inference().raw(x)
            q=calibrated_quality(y,raw,'multiclass',model.decision_shifts)
            quality_close(q,row['quality'])
            tq=calibrated_quality(y,reference,'multiclass',reference_shifts)
            quality_close(tq,source['quality'])
            report,a,b=decompose(y,raw,reference,model.decision_shifts,reference_shifts)
            np.testing.assert_allclose([a.mean(),b.mean()],[q['loss'],tq['loss']],rtol=1e-12,atol=1e-12)
            artifact=output/(row['case_id']+'-objects.npz')
            np.savez_compressed(artifact,indices=indices,target=y,raw_specialist=raw,raw_reference=reference,
                specialist_shifts=model.decision_shifts,reference_shifts=reference_shifts,
                specialist_loss=a,reference_loss=b)
            results.append(dict(case_id=row['case_id'],candidate_id=row['candidate_id'],**report,
                specialist_quality=q,reference_quality=tq,artifact=str(artifact),artifact_sha256=digest(artifact)))
    if any(digest(path)!=checksum for path,checksum in locks.items()): raise RuntimeError('diagnostic source changed')
    write(output/'report.json',dict(status='complete',results=results,input_locks=locks,
        new_fits=0,new_trees=0,nn_weight_updates=0,gpu_used=False,
        calibration_or_routes_changed=False,test_is_reused_exploratory=True,
        scope='descriptive fixed-score attribution; cannot establish counterfactual benefit of new calibration'))
    print(json.dumps(dict(event='specialist_loss_diagnostic_complete',results=results,new_fits=0)),flush=True)


if __name__=='__main__': main()
