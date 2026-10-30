from itertools import product
import pytest
import wide_serving


def test_bank_byte_allocation_agrees_with_exhaustive_frozen_options(monkeypatch):
    monkeypatch.setattr(wide_serving,'budget_reference',lambda outputs:100)
    entries=[];options={}
    for case in ('a','b'):
        options[case]={}
        for family,values in [('ordinary',[(30,.4),(70,.6)]),('mixed24',[(25,.45),(60,.65)])]:
            opts=[]
            for k,(cost,f1) in enumerate(values):
                uid=f'{family}{k}';entry=dict(case=case,uid=uid,roles=[family+str(k)],candidates=[{'resident_bytes':cost-8}],
                     model={'assessment':{'quality':{'f1_macro':f1}}},record={'case':{'outputs':6}})
                entries.append(entry);opts.append((cost,f1,uid))
            options[case][family]=opts
        entries.append(dict(case=case,uid='full',roles=['full200'],candidates=[{'resident_bytes':92}],
                            model={'assessment':{'quality':{'f1_macro':.7}}},record={'case':{'outputs':6}}))
    banks=wide_serving.bank_selections(entries)
    for b in banks['banks']:
        if b['family']=='full200':assert b['cost_bytes']==200;continue
        possible=[p for p in product(*(options[c][b['family']] for c in ('a','b'))) if sum(x[0] for x in p)<=b['budget']]
        if not possible:assert b['status']=='infeasible'
        else:assert b['utility']==pytest.approx(max(sum(x[1] for x in p)/2 for p in possible))
