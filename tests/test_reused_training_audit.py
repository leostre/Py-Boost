import copy
import json
import pytest
from experiments.cumulative_budget_audit import verify_reused_records,cohort_keys
from experiments.a100_runner import sha256


def example(tmp_path):
    original=dict(dataset='toy',seed=1,quality=dict(f1_macro=.5),
        trajectories=[dict(cycles=[dict(fit=dict(trees=40)),dict(state='stopped')])],
        full_fits=[dict(trees=200)])
    parent=tmp_path/'parent.json'
    parent.write_text(json.dumps(dict(records=[original])),encoding='utf-8')
    fresh=dict(copy.deepcopy(original),seed=2)
    source=dict(records=[dict(original,reused=True),fresh],tree_fits=240,reused_tree_fits=240,
        reused_records_manifest=str(parent),reused_records_sha256=sha256(parent))
    return source


def test_exact_reuse_has_separate_price_and_lock(tmp_path):
    source=example(tmp_path)
    locks={}
    assert verify_reused_records(source,locks)==(240,240)
    assert locks[source['reused_records_manifest']]==source['reused_records_sha256']


@pytest.mark.parametrize('bad',['quality','new_price','old_price','duplicate','parent_sha'])
def test_reuse_cannot_hide_changes_or_duplicate_cases(tmp_path,bad):
    source=example(tmp_path)
    if bad=='quality': source['records'][0]['quality']=dict(f1_macro=.9)
    if bad=='new_price': source['tree_fits']=0
    if bad=='old_price': source['reused_tree_fits']=200
    if bad=='duplicate': source['records'][1]['seed']=1
    if bad=='parent_sha': source['reused_records_sha256']='0'*64
    with pytest.raises(RuntimeError): verify_reused_records(source,{})
