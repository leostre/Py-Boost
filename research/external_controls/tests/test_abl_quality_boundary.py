"""Protect the actual sealed development/test boundary, without real datasets."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

import budget_quality_worker  # Select the pinned native import root.
from abl_quality_worker import require_seal


def test_test_reader_rejects_absent_seal(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_seal(tmp_path/'absent.json')


def test_seal_rejects_incomplete_status_and_changed_selection(tmp_path):
    case=tmp_path/'case.json';case.write_text('{"choice":"development winner"}')
    seal=tmp_path/'seal.json'
    payload=dict(status='incomplete',cases=[])
    seal.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='complete development seal'):
        require_seal(seal)
    payload=dict(status='abl_choices_sealed_before_test',cases=[dict(path=str(case),
        sha256=hashlib.sha256(case.read_bytes()).hexdigest())])
    seal.write_text(json.dumps(payload));assert require_seal(seal)==payload
    case.write_text('{"choice":"edited after seeing test"}')
    with pytest.raises(RuntimeError,match='changed after seal'):
        require_seal(seal)


def test_preseal_worker_has_no_training_or_cuda_calls():
    source=Path(__file__).resolve().parents[1]/'abl_quality_worker.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    forbidden={'fit','fit_transform','partial_fit','train','cuda','RawKernel','RawModule'}
    calls=[node.func.attr for node in ast.walk(tree)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert not (set(calls)&forbidden)


def test_test_predictions_enforce_seal_before_reading_data():
    source=Path(__file__).resolve().parents[1]/'abl_quality_worker.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='case_test_predictions')
    first=function.body[0]
    assert isinstance(first,ast.Expr) and isinstance(first.value,ast.Call)
    assert isinstance(first.value.func,ast.Name) and first.value.func.id=='require_seal'
