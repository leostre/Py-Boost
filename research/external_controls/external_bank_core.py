"""Fixed-candidate task-ID subsets; no fitting, files, or test inputs."""
import itertools
import external_quality_core  # explicit frozen-contract namespace
from experiments.shared_leaf_basis import quality_guard


def useful_specialist(q,reference):
    return bool(quality_guard(q,reference) and
        (q['f1_macro']>=1.01*reference['f1_macro'] or q['loss']<=.99*reference['loss']))


def select_subsets(specialists,budgets):
    if len(specialists)>5 or len({s['case_id'] for s in specialists})!=len(specialists):
        raise ValueError('invalid specialist task domain')
    if any(s['serving_bytes']<0 for s in specialists) or any(b<0 for b in budgets):
        raise ValueError('negative specialist cost or budget')
    rows=[]
    for budget in budgets:
        candidates=[]
        for mask in itertools.product((False,True),repeat=len(specialists)):
            selected=[s for keep,s in zip(mask,specialists) if keep]
            cost=sum(s['serving_bytes'] for s in selected)
            if cost>budget: continue
            candidates.append(dict(extra_budget_bytes=budget,extra_serving_bytes=cost,
                specialist_ids=[s['candidate_id'] for s in selected],
                macro_relative_gain=sum(s['macro_relative_gain'] for s in selected),
                loss_relative_gain=sum(s['loss_relative_gain'] for s in selected)))
        rows.append(min(candidates,key=lambda r:(-r['macro_relative_gain'],-r['loss_relative_gain'],r['extra_serving_bytes'],r['specialist_ids'])))
    return rows


def route_outcomes(references,specialists,subsets):
    results=[]
    lookup={s['candidate_id']:s for s in specialists}
    if len(lookup)!=len(specialists): raise ValueError('duplicate specialist identity')
    for subset in subsets:
        selected=[lookup[key] for key in subset['specialist_ids']]
        routed={s['case_id']:s for s in selected}
        if len(routed)!=len(selected) or not routed.keys()<=references.keys():
            raise ValueError('invalid routed task domain')
        tasks=[]
        for case,reference in references.items():
            item=routed.get(case)
            q=reference if item is None else item['quality']
            tasks.append(dict(case_id=case,routed=item is not None,quality=q,reference=reference,
                quality_pass=quality_guard(q,reference),specialist_id=None if item is None else item['candidate_id']))
        results.append(dict(**subset,tasks=tasks,routed_tasks=len(routed),
            routed_quality_passes=sum(t['routed'] and t['quality_pass'] for t in tasks),
            all_quality_pass=all(t['quality_pass'] for t in tasks)))
    return results
