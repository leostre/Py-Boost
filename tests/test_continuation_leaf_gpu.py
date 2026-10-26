import numpy as np
import pytest


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_continuation_uses_checkpoint_and_increment_export_matches_state(task):
    pytest.importorskip('cupy')
    from experiments.a100_runner import build_model
    from experiments.continuation_leaf_boosting import ContinuationLeafBoosting
    from experiments.history_basis_study import raw
    from experiments.shared_leaf_basis import probabilities
    rng = np.random.default_rng(344)
    x = rng.normal(size=(240,5)).astype(np.float32)
    y = np.argmax(x[:,:4],axis=1).astype(np.int32) if task=='multiclass' else (x[:,:4]>0).astype(np.float32)
    initial = np.tile(np.array([.3,-.2,.5,-.7],np.float32),(160,1))
    validation = initial[:80].astype(float)
    config = dict(ntrees=4,lr=.1,max_depth=3,lambda_l2=1.)
    args = build_model('full',task,4,config,499,3,1.).params
    model = ContinuationLeafBoosting(**args,initial_train=initial,initial_validation=[validation],task=task)
    model.fit(x[:160],y[:160],eval_sets=[dict(X=x[160:],y=y[160:])])
    increment = model.export_increment()
    with pytest.raises(NotImplementedError,match='original checkpoint'):
        model.predict(x[160:])
    with pytest.raises(NotImplementedError,match='original checkpoint'):
        model.predict_staged(x[160:])
    assert not np.any(increment.base.bias) and not np.any(increment.decision_shifts)
    np.testing.assert_allclose(probabilities(initial+raw(increment,x[:160]),task),
        probabilities(model.train_final_raw,task),rtol=0,atol=2e-5)
    np.testing.assert_allclose(probabilities(validation+raw(increment,x[160:]),task),
        probabilities(model.validation_final_raw[0],task),rtol=0,atol=2e-5)
    reset = ContinuationLeafBoosting(**args,initial_train=np.zeros_like(initial),
        initial_validation=[np.zeros_like(validation)],task=task)
    reset.fit(x[:160],y[:160],eval_sets=[dict(X=x[160:],y=y[160:])])
    assert not np.allclose(model.train_final_raw,reset.train_final_raw)
