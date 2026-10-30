"""Development-only selection with explicit numerical indifference, no fitting."""
from dataclasses import dataclass

from abl_quality_core import allowed_policies,membership
from audit_calibration_pilot import policy_meta


@dataclass(frozen=True)
class NumericalTolerance:
    loss:float=0.
    macro:float=0.

    def __post_init__(self):
        if self.loss not in (0.,1e-6) or self.macro not in (0.,1e-12):
            raise ValueError('outside fixed numerical diagnostic tolerances')


def band(items,score,objective,tolerance):
    values=list(items)
    if objective=='f1_macro':
        best=max(score(v)['f1_macro'] for v in values)
        values=[v for v in values if score(v)['f1_macro']>=best-tolerance.macro]
    loss=min(score(v)['loss'] for v in values)
    return [v for v in values if score(v)['loss']<=loss+tolerance.loss]


def select_stable(rows,metas,group,mode,objective,budget,tolerance):
    candidates=[]
    for row in rows:
        meta=metas[row['uid']]
        eligible=(meta['family'] in ('pyboost','compact')) if group=='deployed' else membership(meta,group)
        if not eligible or meta['cost_bytes'] is None or meta['cost_bytes']>budget:continue
        names=band(allowed_policies(mode,objective),lambda n:row['validation'][n],objective,tolerance)
        def tail(name):
            temperature,fraction=policy_meta(name)
            return int(temperature)+int(fraction>0),fraction,name
        policy=min(names,key=tail)
        candidates.append(dict(uid=row['uid'],policy=policy,
            validation=row['validation'][policy],cost_bytes=meta['cost_bytes']))
    if not candidates:return None
    return min(band(candidates,lambda v:v['validation'],objective,tolerance),
        key=lambda v:(v['cost_bytes'],v['uid']))
