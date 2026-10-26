import numpy as np
import pytest
from experiments.matched_boost_study import reuse_fit


def test_reused_fit_requires_exact_role_and_artifact_hash(tmp_path):
    from experiments.a100_runner import sha256
    path=tmp_path/'fixed'
    path.write_bytes(b'fixed-model')
    case=dict(dataset='a',seed=1,split_sha256='same')
    fit=dict(depth=3,artifact=str(path),sha256=sha256(path),trees=200)
    controls=dict(records=[dict(case,full_fits=[fit])])
    assert reuse_fit(case,controls,3)==fit
    assert reuse_fit(dict(case,dataset='binary'),controls,3) is None
    with pytest.raises(RuntimeError,match='roles'):
        reuse_fit(dict(case,split_sha256='changed'),controls,3)
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='artifact'):
        reuse_fit(case,controls,3)


def test_builder_preserves_author_projection_and_full_leaf_contract(monkeypatch):
    import sys,types
    from experiments.a100_runner import build_model
    class Stub:
        def __init__(self,**kwargs): self.params=kwargs
    monkeypatch.setitem(sys.modules,'py_boost',types.SimpleNamespace(GradientBoosting=Stub,SketchBoost=Stub))
    model=build_model('sketchboost_proj_half','multiclass',101,dict(ntrees=200,lr=.1),9,3,1)
    assert model.params['sketch_outputs']==51 and model.params['sketch_method']=='proj'
    assert model.params['use_hess'] is False and model.params['full_leaf_values'] is True
    assert model.params['deterministic_leaf_values'] is True
