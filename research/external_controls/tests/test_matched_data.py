import numpy as np
import pytest
from matched_data import binary_roles,parse_public


@pytest.mark.parametrize('seed',range(20))
def test_binary_roles_partition_preserve_outer_and_repeat_exactly(seed):
    y=np.r_[np.zeros(212,np.int32),np.ones(357,np.int32)]
    v2,parts=binary_roles(y,seed)
    _,again=binary_roles(y,seed)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(y)))
    for k in parts: np.testing.assert_array_equal(parts[k],again[k])
    for k in ('calibration','test'): np.testing.assert_array_equal(parts[k],v2[k])
    assert set(np.r_[parts['train'],np.setdiff1d(parts['control'],v2['control']),
                      np.setdiff1d(parts['selection'],v2['selection'])])==set(v2['train'])
    assert .43<len(parts['train'])/len(y)<.46 and .13<len(parts['selection'])/len(y)<.17


@pytest.mark.parametrize('target',[np.ones(100),np.zeros((100,2)),np.arange(100)])
def test_invalid_binary_domain_rejected(target):
    with pytest.raises(ValueError): binary_roles(target,4)


def test_unexpected_raw_public_schema_rejected():
    with pytest.raises(RuntimeError): parse_public('wdbc',b'1,B,4\n2,M,8\n')
    with pytest.raises(RuntimeError): parse_public('banknote',b'1,2,0\n')
