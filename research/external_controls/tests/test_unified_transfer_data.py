from pathlib import Path
import numpy as np
import pytest
from unified_transfer_data import read_genbase,grouped_roles


def test_genbase_parser_excludes_identifier_and_identical_features_never_cross_roles():
    root=Path(__file__).resolve().parents[1]/'unified_raw_genbase'
    if not all((root/name).is_file() for name in ('genbase.arff','genbase.xml')):
        pytest.skip('GenBase integration data is not redistributed; restore the fingerprinted ARFF/XML to run this check')
    x,y,labels=read_genbase(root/'genbase.arff',root/'genbase.xml')
    assert x.shape==(662,1185) and y.shape==(662,27) and len(labels)==27
    roles,info=grouped_roles(x,5304);again,_=grouped_roles(x,5304)
    for key in roles:np.testing.assert_array_equal(roles[key],again[key])
    train={r.tobytes() for r in x[roles['train']]};test={r.tobytes() for r in x[roles['test']]}
    dev={r.tobytes() for key in ('calibration','control','selection') for r in x[roles[key]]}
    assert not train&test and not train&dev and not test&dev
