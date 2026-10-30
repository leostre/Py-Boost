import numpy as np
import pytest
from adaptive_data_core import numeric_arff,preserve_official_test,official_roles,integer_domain


def test_class_domain_has_exact_integer_semantics_and_portable_json_values():
    import json
    d=integer_domain(np.array([7.,1.,3.,7.]));assert d==[1,3,7]
    assert json.loads(json.dumps(d))==d
    with pytest.raises(ValueError):integer_domain([0.,1.2])


def test_dense_sparse_binary_defaults_and_target_order_are_exact():
    names,a=numeric_arff('@relation t\n@attribute x numeric\n@attribute y {1,0}\n@data\n2,1\n{0 3,1 0}\n{}\n')
    assert names==['x','y'];np.testing.assert_array_equal(a,[[2,1],[3,0],[0,1]])
    with pytest.raises(ValueError):numeric_arff('@attribute x numeric\n@data\n{0 1,0 2}')
    with pytest.raises(ValueError):numeric_arff('@attribute x numeric\n@data\n1,2')


def test_official_test_is_unchanged_and_no_feature_group_crosses_roles():
    x=np.repeat(np.arange(30),2).astype(np.float32)[:,None];y=np.arange(60)%2
    tx=np.array([[0.],[100.],[100.]],np.float32);ty=np.array([1,0,1])
    xx,yy,role,row,info=preserve_official_test(x,y,tx,ty)
    assert info['removed_train_rows_crossing_test']==2
    np.testing.assert_array_equal(xx[role==1],tx);np.testing.assert_array_equal(yy[role==1],ty)
    r,g=official_roles(xx,role,yy[role==0],'multiclass',7)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(r.values()))),np.arange(len(xx)))
    dev=np.concatenate([r[k] for k in ('calibration','control','selection')])
    assert not set(xx[r['train'],0])&set(xx[dev,0])
