"""Small exact multiple-choice allocation; development-only, no learned meta-rule."""
from itertools import product


def retention(quality,reference):
    def up(value,base):
        return value/base if base>0 else 1.
    loss = reference['loss']/quality['loss'] if quality['loss']>0 else 1.
    return min(up(quality['f1_macro'],reference['f1_macro']),
               up(quality['f1_weighted'],reference['f1_weighted']),loss)


def key(choice):
    present=[r for r in choice if r is not None]
    utilities=[retention(r['validation'],r['reference_validation']) for r in present]
    return (len(present),min(utilities,default=0.),sum(utilities),
            -sum(r['csr_bytes'] for r in present))


def choose_allocation(groups,budget,policy):
    """One option or an explicit omission per task. No test field is consulted.

    Objective: cover most development-feasible tasks, then worst relative
    quality, then total relative quality, then fewer exact CSR bytes.
    """
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<0:
        raise ValueError('invalid absolute bank-model budget')
    if policy not in ('equal','independent','joint') or not groups or any(not g for g in groups):
        raise ValueError('invalid allocation problem')
    eligible=[]
    for group in groups:
        unique={}
        for row in group:
            if row['csr_bytes']<0 or row['task_id'] != group[0]['task_id']:
                raise ValueError('inconsistent candidate group')
            if row['development_pass']:
                unique.setdefault(row['sha256'],row)
        eligible.append(list(unique.values()))
    if policy=='joint':
        result=tuple(None for _ in groups)
        for candidate in product(*[[None,*g] for g in eligible]):
            if sum(r['csr_bytes'] for r in candidate if r is not None)<=budget and key(candidate)>key(result):
                result=candidate
        return result
    if policy=='equal':
        result=[]
        for group in eligible:
            allowed=[r for r in group if r['csr_bytes']<=budget//len(groups)]
            result.append(max(allowed,key=lambda r:key((r,))) if allowed else None)
        return tuple(result)
    result=[]
    used=0
    for original in groups:
        # Same artifact may legitimately be selected by multiple families.
        # Deduplication must not erase policy membership and cause a full fallback.
        group=[r for r in original if r['development_pass']]
        preferred=next((r for r in group if r['family']=='joint_static'),None)
        preferred=preferred or next((r for r in group if r['family']=='full'),None)
        if preferred is not None and used+preferred['csr_bytes']<=budget:
            result.append(preferred)
            used+=preferred['csr_bytes']
        else:
            result.append(None)
    return tuple(result)
