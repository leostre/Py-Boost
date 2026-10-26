import ast
from pathlib import Path
import numpy as np
import external_quality_core
from experiments.model_bank_core import traffic_schedule,latency_summary


def test_same_causal_request_plan_covers_all_five_real_tasks():
    for kind in ('uniform','skew'):
        a=traffic_schedule(5,128,kind,20261021)
        assert len(a)==128 and a[:5]==list(range(5)) and set(a)==set(range(5))
        assert a==traffic_schedule(5,128,kind,20261021)
    assert latency_summary([1,2,3])['requests']==3


def test_shared_loader_does_not_silently_replace_auto_by_one_estimator():
    path=Path(__file__).resolve().parents[1]/'tfm_bank_worker.py'
    source=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='shared_spec')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='initialize_tabpfn_model']
    assert len(calls)==1
    assert 'n_estimators_override' not in {k.arg for k in calls[0].keywords}


def test_isolation_reference_is_paired_same_batch_not_different_full_batch():
    path=Path(__file__).resolve().parents[1]/'tfm_bank_worker.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    comparisons=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='allclose']
    assert len(comparisons)==1
    assert [ast.unparse(a) for a in comparisons[0].args[:2]]==['actual','before']
