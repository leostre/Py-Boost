"""Protect the allocator-scope regression without allocating any GPU memory."""
import inspect
import pytest
from experiments.leaf_budget_runtime import fit_dense


def test_native_profiler_uses_nested_scope_not_forbidden_global_setter():
    source = inspect.getsource(fit_dense)
    assert 'cp.cuda.set_allocator(' not in source
    assert 'with cp.cuda.using_allocator(tracked)' in source
    assert 'super()._fit(builder, info)' in source


def test_missing_calendar_limit_requires_explicit_bounded_tree_contract():
    from experiments.leaf_budget_runtime import check_deadline
    with pytest.raises(ValueError): check_deadline({})
    with pytest.raises(ValueError): check_deadline({'bounded_tree_execution':'true'})
    check_deadline({'bounded_tree_execution':True})
    with pytest.raises(TimeoutError):
        check_deadline({'bounded_tree_execution':True,'autonomous_deadline_utc':'2020-01-01T00:00:00Z'})
