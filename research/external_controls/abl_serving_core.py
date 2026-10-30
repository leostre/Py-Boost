"""Development-sealed serving plans; no target or test metric is accepted."""
from collections import defaultdict
from dataclasses import asdict
import json


def serving_plans(case,control_for_row):
    """Both declared objectives; never replace an inconvenient selected model."""
    rows={(r['uid'],r['plan']):r for r in case['rows']}
    groups=defaultdict(dict)
    for choice in case['budget_choices']:
        if (choice['scheme']!='crossfit' or choice['repeat']!=0 or
                choice['mode']!='temperature_shifts' or choice['budget_ratio'] not in (.5,1,2)):
            continue
        key=choice['objective'],choice['budget_ratio']
        chosen=choice['chosen']
        if chosen is None:
            groups[key][choice['group']]=None;continue
        uid=chosen['uid'];meta=case['metas'][uid]
        control=asdict(control_for_row(rows[uid,choice['plan']],chosen['policy']))
        groups[key][choice['group']]=dict(uid=uid,source=meta['source'],control=control,
            rank=meta['rank'],nnz=meta['nnz'],cost_bytes=meta['cost_bytes'],policy=chosen['policy'])
    return [dict(case=case['case_id'],objective=key[0],budget_ratio=key[1],
        models=models,record=case['record']) for key,models in sorted(groups.items())]


def model_key(model):
    return json.dumps(dict(source=model['source'],control=model['control']),sort_keys=True)


def response_cpu(raw,temperature,shifts):
    from budget_quality_core import probabilities
    return probabilities(raw,temperature),(raw/temperature-shifts).argmax(axis=1)
