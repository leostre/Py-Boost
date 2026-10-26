from copy import deepcopy
import pytest
from experiments.guard_evidence import metric_ledger


def row():
    return dict(dataset='toy',seed=1,method='strict',loss=1.,macro_f1=.5,weighted_f1=.5,
        reference_loss=1.,reference_macro_f1=.5,reference_weighted_f1=.5,
        test_quality_pass=True,joint_success=True,state='accepted',memory_fraction=.7,coefficient_fraction=.6,new_trees=0)


def test_ledger_preserves_metrics_but_independently_checks_quality_and_cost():
    value=row();saved=deepcopy(value)
    assert metric_ledger([value])==[dict(method='strict',cases=1,successes=1,new_trees=0)]
    assert value==saved
    for field,bad in [('loss',1.1),('macro_f1',.4),('weighted_f1',.4),('memory_fraction',.8)]:
        changed={**value,field:bad}
        with pytest.raises(ValueError):
            metric_ledger([changed])
    with pytest.raises(ValueError,match='duplicate'):
        metric_ledger([value,value])
