import pytest
from experiments.matched_boost_audit import verify_cohort


def test_native_cohort_cannot_silently_drop_or_duplicate_family():
    records=[dict(case_id=c,family=m) for c in ('a','b') for m in ('full','proj')]
    verify_cohort(records,['a','b'],['full','proj'])
    for bad in (records[:-1],records+[records[0]],records+[dict(case_id='c',family='full')]):
        with pytest.raises(RuntimeError,match='cohort'): verify_cohort(bad,['a','b'],['full','proj'])
