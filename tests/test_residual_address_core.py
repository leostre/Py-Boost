import numpy as np
import pytest
from experiments.residual_address_core import random_support


@pytest.mark.parametrize('seed',range(50))
def test_random_support_exact_nonzero_count_reproducible(seed):
    rng=np.random.default_rng(seed)
    r=rng.normal(size=(7,5)).astype(np.float32)
    r[rng.random(r.shape)<.3]=0
    count=min(seed%20,np.count_nonzero(r))
    s=random_support(r,count,701)
    assert len(s)==len(np.unique(s))==count
    assert np.all(r.ravel()[s]!=0)
    np.testing.assert_array_equal(s,random_support(r,count,701))
    if count==np.count_nonzero(r):
        np.testing.assert_array_equal(np.sort(s),np.flatnonzero(r.ravel()))


def test_rejects_overflow_and_invalid_input():
    for r,count in ((np.zeros((2,3)),1),(np.ones((2,3)),7),(np.ones((2,3)),-1),
                    (np.ones(4),1),(np.array([[np.nan]]),0),(np.ones((2,3)),True)):
        with pytest.raises(ValueError): random_support(r,count,701)
