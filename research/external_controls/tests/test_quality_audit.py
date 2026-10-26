import pytest
from external_quality_core import audit_then_test,quality_close


def test_whole_cohort_audited_and_sealed_before_first_test():
    trace=[]
    rows=['a','b','c']
    def probe(r): trace.append(('probe',r)); return r+'-verified'
    def seal(values):
        assert values==['a-verified','b-verified','c-verified']
        trace.append(('seal',None))
        return 'sealed'
    def evaluate(r,a):
        assert a=='sealed' and trace[:4]==[('probe','a'),('probe','b'),('probe','c'),('seal',None)]
        trace.append(('test',r))
        return r+'-test'
    evidence,results=audit_then_test(rows,probe,seal,evaluate)
    assert results==['a-test','b-test','c-test'] and len(evidence)==3


def test_failed_audit_prevents_all_test_access():
    calls=[]
    def fail(r): raise RuntimeError('bad hash')
    with pytest.raises(RuntimeError,match='bad hash'):
        audit_then_test(['a'],fail,lambda x:calls.append(x),lambda *x:calls.append(x))
    assert not calls
    q=dict(loss=.1,f1_macro=.5,f1_weighted=.6)
    quality_close(q,q)
    with pytest.raises(RuntimeError): quality_close(dict(q,f1_macro=.5001),q)
