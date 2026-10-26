import numpy as np
import pytest
from external_quality_core import canonical_raw,choose,calibrated_rows,tree_counts
from experiments.shared_leaf_basis import probabilities


def test_quality_function_ast_matches_frozen_source():
    import ast,hashlib,inspect
    from experiments.a100_runner import quality
    fn=ast.parse(inspect.getsource(quality)).body[0]
    # Python 3.12 adds empty FunctionDef.type_params; ignore only that empty
    # syntactic metadata so the frozen 3.10 AST digest remains portable.
    for node in ast.walk(fn):
        if 'type_params' in node._fields and getattr(node,'type_params',None)==[]:
            node._fields=tuple(field for field in node._fields if field!='type_params')
    assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()=='07febdc8948c9ba57a53369060820dafa7510420c7c8f4abc0f87bf499d6d382'


@pytest.mark.parametrize('seed',range(30))
def test_binary_and_probability_canonicalization(seed):
    m=np.random.default_rng(seed).normal(size=15)
    z=canonical_raw(m,[0,1],2)
    np.testing.assert_allclose(probabilities(z,'multiclass')[:,1],1/(1+np.exp(-m)))
    p=probabilities(z,'multiclass')
    np.testing.assert_allclose(probabilities(canonical_raw(p,[0,1],2,probabilities=True),'multiclass'),p)
    with pytest.raises(ValueError): canonical_raw(m,[1,0],2)


def test_rejects_probability_and_axis_errors():
    for a,c,flag in (([[1,1]],[0,1],True),([np.nan],[0,1],False),([[1,2,3]],[0,1],False)):
        with pytest.raises(ValueError): canonical_raw(a,c,2,probabilities=flag)


def test_scalar_tree_counts_are_not_boosting_rounds():
    trees=[dict(children=[dict(leaf=.1),dict(children=[dict(leaf=.2),dict(leaf=.3)])]) for _ in range(3)]
    assert tree_counts(trees)==dict(trees=3,leaves=9,internal_nodes=6,leaf_parameters=9)
    with pytest.raises(ValueError): tree_counts([dict(children=[])])


def test_same_threshold_selection_and_quality_priority():
    z=np.array([[2.,0],[0,2],[1,0],[0,1]])
    y=np.array([0,1,0,1])
    rows=calibrated_rows(z,y,z,y,'multiclass','fixed',dict(payload_bytes=100,parameters=2),[0,.25,-.25],[.25,.5,1])
    assert len(rows)==4 and choose(rows)['validation']['f1_macro']==1
    bad=dict(rows[0],candidate_id='bad',payload_bytes=1,validation=dict(loss=1,f1_macro=.5,f1_weighted=.5))
    assert choose([bad,*rows])['candidate_id']=='fixed'
