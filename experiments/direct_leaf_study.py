"""No model training: fixed-support numerical equivalence and CPU solver costs."""
import subprocess
import sys
from time import perf_counter
import tracemalloc
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,emit,write_report
from experiments.direct_leaf_solver import direct_newton
from experiments.online_leaf_core import weighted_projection


def measure(call):
    for _ in range(2): call()
    times=[]
    for _ in range(12):
        start=perf_counter()
        call()
        times.append(1000*(perf_counter()-start))
    tracemalloc.start()
    tracemalloc.reset_peak()
    result=call()
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result,dict(median_ms=float(np.median(times)),raw_ms=times,
        instrumented_heap_peak_bytes=peak,heap_is_not_process_RSS=True)


def direct_leaf_study(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_direct_leaf_solver.py'],cwd=ROOT,check=True,timeout=600)
    rng=np.random.default_rng(20261021)
    records=[]
    with threadpool_limits(limits=1):
        for c in (14,100,159):
            for r in sorted({min(8,c),min(32,c),min(64,c)}):
                g=rng.normal(size=(64,c))
                d=rng.uniform(.1,5,size=g.shape)
                b=np.linalg.qr(rng.normal(size=(c,c)))[0][:,:r].astype(np.float32)
                target=-.1*g/d
                for fraction in (0.,.05):
                    chosen=weighted_projection(target,d,b,fraction)
                    rr,cc=chosen.rows,chosen.columns
                    # Projection-only includes its support search; direct receives
                    # that fixed support. These are not whole-algorithm speedups.
                    projected,timing_p=measure(lambda:weighted_projection(target,d,b,fraction))
                    direct,timing_d=measure(lambda:direct_newton(g,d,b,.1,rr,cc))
                    delta=float(np.max(np.abs(direct.decoded(b)-projected.decoded(b))))
                    if delta>2e-5: raise RuntimeError('direct diagonal equivalence failed')
                    row=dict(leaves=64,outputs=c,rank=r,residual_fraction=fraction,nnz=len(rr),
                        max_abs_difference=delta,weighted_gap=direct.weighted_error-projected.weighted_error,
                        projection=timing_p,direct_fixed_support=timing_d,
                        direct_support_search_included=False,full_target_input_bytes=target.nbytes,
                        direct_lhs_block_bound_bytes=8*min(8,len(g))*r*r,
                        projection_lhs_bytes=8*len(g)*r*r)
                    records.append(row)
                    emit('direct_leaf_solver_case',**row)
    write_report(output/'solver_results.json',dict(records=records,new_fits=0,new_trees=0))
    report.update(status='complete',records=records,new_fits=0,new_trees=0,test_evaluated=False,
        conclusion='same fixed-basis/support diagonal algorithm; workspace engineering only',
        timing_limit='direct support supplied, reference includes search; not end-to-end algorithm speedup')
