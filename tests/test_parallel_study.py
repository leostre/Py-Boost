from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.parallel_leaf_study import execute_requests,train_variant
from experiments.online_leaf_core import stitch


def test_capacity_fallback_preserves_jobs(monkeypatch):
    import experiments.inference_gpu_guard as guard
    class Pool:
        def __init__(self):
            self.jobs = []
        def submit(self,fn,r):
            self.jobs.append(r)
            return SimpleNamespace(result=lambda timeout:dict(seconds=2,fit_started_ns=10,
                fit_finished_ns=30,training_pool_peak_bytes=100,id=r['id']))
    requests = [dict(id=1),dict(id=2)]
    for available in (True,False):
        monkeypatch.setattr(guard,'gpu_probe',lambda *a:(available,None if available else 'capacity',[]))
        pool = Pool()
        records,execution = execute_requests(requests,pool)
        assert pool.jobs == requests and [r['id'] for r in records] == [1,2]
        assert execution['execution'] == ('concurrent' if available else 'sequential')
        assert execution['sum_individual_pool_peaks'] == 200


@pytest.mark.parametrize('variant',['serial','serial_half_step','parallel_dense','parallel_compact'])
def test_four_cycle_orchestration_with_fake_worker_never_resets_checkpoint(variant,tmp_path,monkeypatch):
    from test_prepared_leaf_inference import model_for
    from experiments.parallel_leaf_core import dense_parts
    from experiments.history_basis_study import raw
    from experiments.parallel_leaf_study import train_variant
    import experiments.parallel_leaf_study as study
    original = model_for('multilabel','rank')
    parts = dense_parts(model_for('multilabel','full'))*20
    # Repeated shapes, genuine common decoder and zero residual are sufficient
    # for an I/O/state test. Native optimization is tested separately on GPU.
    dense = stitch(parts,None,original.base.bias,2,'multilabel')
    initial = replace(dense,base=replace(dense.base,coefficients=(dense.base.coefficients@original.base.basis).astype(np.float32),
        basis=original.base.basis,representation='low_rank'))
    x = np.tile(np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32),(3,1))
    y = initial.predict_labels(x)
    splits = dict(train=np.arange(8),validation=np.arange(8,12))
    directory = tmp_path/variant
    directory.mkdir()
    expected_initial = [raw(initial,x[splits['train']]).astype(np.float32)]
    observed = []
    def fake_jobs(requests,pool):
        records = []
        for i,r in enumerate(requests):
            with np.load(r['initial_path']) as f:
                observed.append(f['train'].copy())
            if len(observed) <= len(requests):
                np.testing.assert_array_equal(observed[-1],expected_initial[0])
            value = float(observed[-1].mean())*.001+.01
            toy = dense_parts(model_for('multilabel','full',i))
            toy = [replace(p,coefficients=np.full_like(p.coefficients,value)) for p in toy]*(r['trees']//2)
            increment = stitch(toy,None,np.zeros(4,np.float32),2,'multilabel')
            increment.save(r['output'])
            records.append(dict(seconds=1,training_pool_peak_bytes=100))
        return records,dict(execution='sequential',fit_overlap_seconds=0,sum_worker_fit_seconds=len(records),wall_seconds=1)
    monkeypatch.setattr(study,'execute_requests',fake_jobs)
    final,record = train_variant(variant,initial,x,y,splits,dict(online_residual_fraction=.05),directory,None,17)
    assert record['new_built_trees'] == 160 and record['total_built_trees'] == 200
    assert len(record['cycles']) == 4
    assert final.n_trees == 200 if hasattr(final,'n_trees') else final.base.n_trees == 200
    per_cycle = 2 if variant.startswith('parallel') else 1
    assert len(observed) == 4*per_cycle
    assert not np.array_equal(observed[0],observed[-1])
    for cycle in range(1,4):
        from experiments.block_leaf_model import load_prediction_model
        previous = load_prediction_model(directory/f'cycle{cycle-1}/checkpoint.npz')
        np.testing.assert_array_equal(observed[cycle*per_cycle],raw(previous,x[splits['train']]).astype(np.float32))
