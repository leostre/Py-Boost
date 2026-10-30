import numpy as np
import pytest
from unified_meta_core import training_profile


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_descriptive_spectrum_does_not_depend_on_label_names_or_row_order(task):
    rng=np.random.default_rng(18);x=rng.normal(size=(80,3)).astype(np.float32)
    y=np.arange(80)%4 if task=='multiclass' else (rng.random((80,4))<.2).astype(int)
    p=training_profile(dict(train_x=x,train_y=y,dev_y=y[:20]),task,4)
    other=3-y[::-1] if task=='multiclass' else y[::-1,::-1]
    q=training_profile(dict(train_x=x[::-1],train_y=other,dev_y=other[:20]),task,4)
    assert p['target_covariance_effective_rank']==pytest.approx(q['target_covariance_effective_rank'])
    assert 0<=p['target_covariance_effective_rank']<=4.00001
