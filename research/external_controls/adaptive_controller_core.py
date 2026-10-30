"""Finite local archive actions and train-only diagnostic decisions."""
from dataclasses import dataclass,replace
import numpy as np
import budget_quality_worker
from adaptive_basis_core import risk_weights,risk_basis,all_support_order
from abl_protected_core import append_protected,as_protected,block_cost
from abl_budget_revision_core import decode_model,hessian,pack
from abl_budget_revision import group_model
from unified_training_core import coarsen,prepare_training_group,canonical
from experiments.shared_leaf_basis import functional_basis
from experiments.sparse_leaf_core import contrast


@dataclass(frozen=True)
class Action:
    rank:int
    cap:int
    residual:float
    basis:str='rotate'

    def __post_init__(self):
        if self.rank<1 or self.cap not in (4,6,8) or self.residual not in (0.,.02,.05) or self.basis not in ('rotate','keep'):
            raise ValueError('action outside registered domain')

    @property
    def uid(self):return f'{self.basis}-l{self.cap}-r{self.rank}-p{self.residual:g}'


def promotion(cases):
    """Equal-case medians, D only. A promotion is a next-test choice, not proof."""
    if not cases or len({c['case'] for c in cases})!=len(cases) or any(not c['matched'] for c in cases):
        raise ValueError('nonempty distinct matched cases required')
    stats={name:[] for name in ('risk_basis','risk_residual')}
    for case in cases:
        by={m['uid']:m for m in case['models']};task=case['task']
        def q(m):return next(e['quality'] for e in m['evaluations'] if e['objective']==('loss' if task=='multiclass' else 'f1_macro'))
        values={k:[] for k in stats}
        for pair in case['matched']:
            a=q(by[pair['uids'][0]])
            for name,uid in zip(('risk_basis','risk_residual'),pair['uids'][1:]):
                b=q(by[uid]);values[name].append(1-b['loss']/a['loss'] if task=='multiclass' else b['f1_macro']/max(a['f1_macro'],1e-12)-1)
        for name in stats:stats[name].append(dict(case=case['case'],median_relative_gain=float(np.median(values[name]))))
    scores={name:dict(mean_case_median=float(np.mean([x['median_relative_gain'] for x in data])),
        positive_cases=sum(x['median_relative_gain']>0 for x in data),cases=data) for name,data in stats.items()}
    eligible=[name for name,s in scores.items() if s['positive_cases']>len(cases)/2 and s['mean_case_median']>0]
    chosen=max(eligible,key=lambda name:(scores[name]['mean_case_median'],name)) if eligible else 'functional'
    return dict(chosen=chosen,comparisons=scores,rule='strict majority positive case medians and positive equal-case mean; otherwise functional')


def extend_basis(previous,new):
    if previous is None:return new
    columns=[]
    for v in np.column_stack((previous,new)).T:
        v=v.astype(float).copy()
        for q in columns:v-=q*(q@v)
        for q in columns:v-=q*(q@v)
        norm=np.linalg.norm(v)
        if norm>1e-6:columns.append(v/norm)
        if len(columns)==new.shape[1]:break
    return np.column_stack(columns).astype(np.float32)


def candidates(checkpoint,increment,features,labels,actions,policy,previous_basis=None,max_basis_rank=None):
    teacher=as_protected(increment) if checkpoint is None else append_protected(checkpoint,increment)
    raw_teacher=teacher.raw(features['train']);hh=hessian(raw_teacher,teacher.task)
    state={};groups={};row_w,col_w=risk_weights(labels['train'],raw_teacher,teacher.task)
    for action in dict.fromkeys(actions):
        if action.cap not in state:
            short=coarsen(increment,features['train'],raw_teacher,action.cap)
            source=short if checkpoint is None else pack(checkpoint.blocks+short.blocks,checkpoint.bias+short.bias)
            view,_=decode_model(source)
            view=replace(view,values=contrast(view.values,view.task));contribution=np.asarray(view.design(features['train'])@view.values)
            basis=(functional_basis(contribution,view.task)[1] if policy=='functional' else risk_basis(contribution,view.task,row_w,col_w,max_basis_rank)[0])
            state[action.cap]=(view,basis,extend_basis(previous_basis,basis))
        view,basis,kept=state[action.cap];full=action.rank==len(view.bias)-int(view.task=='multiclass')
        if full and action.residual:continue
        key=(action.cap,action.rank,action.basis)
        if key not in groups:
            b=None if full else (kept if action.basis=='keep' else basis)[:,:action.rank]
            g=prepare_training_group(view,b,features,labels,raw_teacher,hh)
            if not full:g.order=all_support_order(g,labels['train'],raw_teacher,'balanced_margin' if policy=='risk_residual' else 'balanced_mse')
            groups[key]=g
        g=groups[key];count=min(len(g.order),int(action.residual*view.values.size))
        leaf,offset=group_model(g,g.order[:count]);model=canonical(pack([leaf],view.bias+offset))
        raw={k:model.raw(x) for k,x in features.items()}
        error=contrast(raw_teacher-raw['train'],view.task)
        yield dict(model=model,raw=raw,rank=action.rank,leaf_cap=action.cap,residual_fraction=action.residual,
            basis_kind=action.basis,uid=action.uid,cost=block_cost(model),train_mse=float(np.mean(error**2)),
            train_risk_mse=float(np.mean(row_w[:,None]*col_w[None,:]*error**2)),nnz=block_cost(model)['nnz'])


def diagnose(error,coarse_error):
    error=np.asarray(error,float);energy=(error**2).sum(0);total=float(energy.sum())
    eigen=np.linalg.eigvalsh(error.T@error)
    return dict(systematic=float(max(eigen[-1],0)/max(total,1e-30)),
        localized=float(np.sort(energy)[-max(1,int(np.ceil(.1*len(energy)))):].sum()/max(total,1e-30)),
        structural=max(0.,float(np.mean(coarse_error**2)/max(np.mean(error**2),1e-30)-1)))


def local_actions(rank,cap,residual,ranks,diagnostic):
    grid=sorted(set(ranks));i=min(range(len(grid)),key=lambda i:abs(grid[i]-rank));rank=grid[i]
    up=grid[min(i+1,len(grid)-1)];down=grid[max(i-1,0)]
    other_s=.05 if residual<.05 else .02
    structure=8 if diagnostic['structural']>.15 else 4
    actions=[Action(rank,cap,residual,'keep'),Action(rank,cap,residual),Action(up,cap,residual),
        Action(down,cap,other_s if diagnostic['localized']>.6 else residual),
        Action(rank,cap,other_s),Action(rank,structure,residual)]
    if diagnostic['systematic']>.45:actions[2]=Action(up,cap,0.)
    # D queries are bounded by six distinct affordable neighbours. Diagnostics
    # change their order and allocation, never use D/test labels themselves.
    if diagnostic['localized']>.6:actions[2],actions[4]=actions[4],actions[2]
    return tuple(dict.fromkeys(actions))
