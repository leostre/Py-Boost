import pytest
from external_bank_core import route_outcomes


def test_bad_specialist_remains_bad_and_other_tasks_not_counted_as_improvement():
    q=dict(loss=.1,f1_macro=.5,f1_weighted=.6)
    references={'a':q,'b':q}
    specialists=[dict(case_id='a',candidate_id='s',quality=dict(q,loss=.2,f1_macro=.6))]
    subsets=[dict(specialist_ids=['s'],extra_budget_bytes=100,extra_serving_bytes=10)]
    result=route_outcomes(references,specialists,subsets)[0]
    assert not result['all_quality_pass'] and result['routed_quality_passes']==0 and result['routed_tasks']==1
    assert result['tasks'][1]['quality_pass'] and not result['tasks'][1]['routed']
    assert subsets[0]['specialist_ids']==['s']


def test_empty_route_and_invalid_duplicate_cases():
    q=dict(loss=.1,f1_macro=.5,f1_weighted=.6)
    result=route_outcomes({'a':q},[],[dict(specialist_ids=[])])[0]
    assert result['all_quality_pass'] and result['routed_tasks']==0
    models=[dict(case_id='a',candidate_id=s,quality=q) for s in ('x','y')]
    with pytest.raises(ValueError): route_outcomes({'a':q},models,[dict(specialist_ids=['x','y'])])
