import itertools
import numpy as np
import pytest
from external_bank_core import useful_specialist,select_subsets


@pytest.mark.parametrize('seed',range(40))
def test_exact_subset_matches_independent_bitmask_oracle(seed):
    rng=np.random.default_rng(seed)
    n=int(rng.integers(0,6))
    items=[dict(case_id=str(i),candidate_id=str(i),serving_bytes=int(rng.integers(1,100)),
                macro_relative_gain=float(rng.integers(-4,5))/100,
                loss_relative_gain=float(rng.integers(-4,5))/100) for i in range(n)]
    budget=int(rng.integers(0,300))
    result=select_subsets(items,[budget])[0]
    options=[]
    for mask in range(1<<n):
        chosen=[items[i] for i in range(n) if mask&(1<<i)]
        cost=sum(s['serving_bytes'] for s in chosen)
        if cost<=budget:
            options.append((-sum(s['macro_relative_gain'] for s in chosen),
                -sum(s['loss_relative_gain'] for s in chosen),cost,[s['candidate_id'] for s in chosen]))
    expected=min(options)
    assert (-result['macro_relative_gain'],-result['loss_relative_gain'],result['extra_serving_bytes'],result['specialist_ids'])==expected


def test_joint_quality_guard_and_duplicate_tasks_are_not_bypassed():
    ref=dict(loss=1.,f1_macro=.5,f1_weighted=.6)
    assert useful_specialist(dict(ref,f1_macro=.51),ref)
    assert not useful_specialist(dict(ref,f1_macro=.7,loss=1.06),ref)
    assert not useful_specialist(ref,ref)
    item=dict(case_id='a',candidate_id='x',serving_bytes=1,macro_relative_gain=.1,loss_relative_gain=.1)
    with pytest.raises(ValueError): select_subsets([item,item],[100])
    with pytest.raises(ValueError): select_subsets([item],[-1])
