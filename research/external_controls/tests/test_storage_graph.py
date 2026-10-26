import itertools
import numpy as np
import pytest
from storage_graph import BufferSpan,union_bytes,inventory


@pytest.mark.parametrize('seed',range(50))
def test_interval_union_matches_discrete_oracle_and_is_order_independent(seed):
    rng=np.random.default_rng(seed)
    spans=[BufferSpan(str(rng.integers(0,2)),int(rng.integers(1,50)),int(rng.integers(0,12))) for _ in range(20)]
    oracle={(s.device,p) for s in spans for p in range(s.address,s.address+s.nbytes)}
    assert union_bytes(spans)==len(oracle)==union_bytes(list(reversed(spans)))
    assert union_bytes(spans+spans)==len(oracle)


def test_numpy_views_and_cycles_do_not_double_count_owners():
    x=np.zeros((10,4),np.float64)
    cycle=[x,x[:,1:],x[::2]]
    cycle.append(cycle)
    result=inventory(cycle)
    assert result['cpu_numerical_bytes']==x.nbytes and result['gpu_numerical_bytes']==0


@pytest.mark.parametrize('device,address,size',[('',1,1),('cpu',0,1),('cpu',True,1),('cpu',1,-1)])
def test_bad_span_rejected(device,address,size):
    with pytest.raises(ValueError): BufferSpan(device,address,size)
