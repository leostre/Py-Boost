import numpy as np
import pytest
from packed_preflight import fixture
from packed_serving_core import PackedPlan,prepare_snapshot


def interpret(snapshot,x):
    """Scalar interpreter independent of CUDA's parallel gather/reduction."""
    a=snapshot.arrays;c=snapshot.outputs;out=np.broadcast_to(a['bias'],(len(x),c)).astype(float,copy=True)
    for begin,trees,w,low,ao,bo,po,has in a['meta']:
        for i,row in enumerate(x):
            latent=np.zeros(w);residual=np.zeros(c)
            for t in range(begin,begin+trees):
                p=0
                while p>=0:
                    idx=a['noff'][t]+p
                    if snapshot.layout=='aos':f,cut,l,r=a['nodes'][idx]
                    else:f,cut,l,r=a['feature'][idx],a['threshold'][idx],a['left'][idx],a['right'][idx]
                    v=row[abs(int(f))-1];right=(f>0) if np.isnan(v) else v>cut;p=int(r if right else l)
                leaf=a['loff'][t]-p-1;latent+=a['coefficients'][ao+leaf*w:ao+(leaf+1)*w]
                if has:
                    for s in range(a['ptr'][po+leaf],a['ptr'][po+leaf+1]):residual[a['columns'][s]]+=a['values'][s]
            if low:out[i]+=latent@a['basis'][bo:bo+w*c].reshape(w,c)
            else:out[i,:w]+=latent
            out[i]+=residual
    return out


@pytest.mark.parametrize('c',[6,159,374,983])
@pytest.mark.parametrize('residual',[False,True])
def test_snapshot_all_blocks_routes_and_sparse_offsets_match_source(c,residual):
    model,x,shifts=fixture(c,residual);snapshots=[prepare_snapshot(model,shifts,l) for l in ('aos','soa')]
    for s in snapshots:
        np.testing.assert_allclose(interpret(s,x),model.raw(x),atol=1e-12,rtol=0)
        assert s.trees==model.n_trees
        assert all(not a.flags.writeable for a in s.arrays.values())
    assert snapshots[0].resident_bytes==snapshots[1].resident_bytes


def test_snapshot_breaks_source_array_aliases():
    model,x,shifts=fixture(6);s=prepare_snapshot(model,shifts);before=interpret(s,x)
    model.blocks[0].base.coefficients[:]=0;shifts[:]=3
    np.testing.assert_array_equal(before,interpret(s,x));assert not np.all(s.arrays['shifts']==3)


def test_shared_limit_has_explicit_wide_hybrid_fallback():
    with pytest.raises(ValueError,match='48KiB'):PackedPlan(4,'aos',983,983,True)
    assert PackedPlan(2,'aos',983,983,True).shared_bytes<=49152
    assert PackedPlan(4,'soa',983,983,False).shared_bytes<=49152
    with pytest.raises(ValueError):PackedPlan(3,'aos',6,6,False)
