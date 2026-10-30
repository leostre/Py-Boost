"""Generated rank/ownership laws; no dependency installation required."""
import numpy as np
import pytest
hypothesis=pytest.importorskip('hypothesis')
from hypothesis import given,strategies as st,settings

import budget_quality_worker
from abl_quality_core import rank_levels,representation_plans


@given(classes=st.integers(2,150),leaves=st.integers(2,500),rows=st.integers(2,500),
       parameters=st.integers(1,100000))
@settings(max_examples=80)
def test_plan_stays_in_contrast_space_and_never_creates_more_than_five_ranks(classes,leaves,rows,parameters):
    levels=rank_levels(classes,leaves,rows)
    assert 1<=len(levels)<=5 and levels[-1]==min(classes-1,leaves,rows)
    plans=representation_plans(classes,leaves,rows,parameters)
    assert len(plans)==4*len(levels)
    for rank in levels:
        supports=[p.maximum_nnz for p in plans if p.rank==rank]
        assert supports==sorted(supports) and supports[0]==0 and supports[-1]<=parameters*.05
