import numpy as np
import pytest
from unified_data_core import reclaim_roles


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_reclamation_only_moves_old_development_to_train(task):
    roles=dict(train=np.arange(45),calibration=np.arange(45,60),control=np.arange(60,72),selection=np.arange(72,100),test=np.arange(100,125))
    rng=np.random.default_rng(9);y=np.arange(55)%4 if task=='multiclass' else (rng.random((55,4))<.2).astype(int)
    new,info=reclaim_roles(roles,y,4,task,17)
    again,_=reclaim_roles(roles,y,4,task,17)
    np.testing.assert_array_equal(new['test'],roles['test'])
    for k in new:np.testing.assert_array_equal(new[k],again[k])
    assert set(roles['train'])<=set(new['train']) and len(new['train'])>len(roles['train'])
    assert sorted(np.concatenate(list(new.values())))==list(range(125))
    assert .76<info['train_fraction_non_test_actual']<.84
    assert not set(new['train'])&set(new['test'])
