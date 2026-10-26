from copy import deepcopy

import pytest

from experiments.guard_reuse_core import all_pass_equivalent


def trace():
    row = dict(candidate_id='good', validation=dict(loss=1.,f1_macro=.5,f1_weighted=.5),
               payload_bytes=100,parameters=20)
    return dict(new_built_trees=200,budget=dict(parameters=1000,payload_bytes=4000),cycles=[dict(
        block=i,state='accepted',accepted_trees=(i+1)*40,parameters=20,payload_bytes=100,
        projection=dict(chosen=deepcopy(row),candidates=[deepcopy(row)],proposal_control=row['validation'].copy()))
        for i in range(5)])


def test_complete_all_pass_trace_can_be_reused_without_mutation():
    data=trace();saved=deepcopy(data)
    assert all_pass_equivalent(data) and data==saved


@pytest.mark.parametrize('failure', ['quality','resource','partial','wrong_choice','tree_count','order'])
def test_one_broken_condition_forbids_counterfactual_reuse(failure):
    data=trace()
    if failure=='quality':
        data['cycles'][2]['projection']['chosen']['validation']['loss']=1.2
    elif failure=='resource':
        data['cycles'][2]['payload_bytes']=4001
    elif failure=='partial':
        data['cycles'].pop()
    elif failure=='wrong_choice':
        better=deepcopy(data['cycles'][1]['projection']['chosen'])
        better['validation']['f1_macro']=.6
        data['cycles'][1]['projection']['candidates'].append(better)
    elif failure=='tree_count':
        data['cycles'][2]['accepted_trees']=119
    else:
        data['cycles'][3]['block']=2
    assert not all_pass_equivalent(data)
