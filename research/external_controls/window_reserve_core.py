"""Exact quota accounting and convex coefficient refits on frozen leaf maps."""
from fractions import Fraction
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.special import expit,softmax,logsumexp
from scipy.optimize import minimize


def quota(count,fraction):
    f=Fraction(str(fraction))
    if not 0<=f<=1 or count<0:raise ValueError('invalid quota')
    return int(count*f.numerator//f.denominator)


def support(scores,offsets,fraction,window,kind,seed=0):
    scores=np.asarray(scores,float);offsets=np.asarray(offsets)
    if (scores.ndim!=2 or offsets.ndim!=1 or offsets[0]!=0 or offsets[-1]!=len(scores) or np.any(np.diff(offsets)<=0)
        or not np.isfinite(scores).all() or np.any(scores<0) or window<1):raise ValueError('invalid frozen leaf domain')
    c=scores.shape[1];trees=len(offsets)-1;selected=[];rng=np.random.default_rng(seed);blocks=[]
    def best(a,b,k,random=False):
        keys=np.arange(a*c,b*c,dtype=np.int64)
        return rng.choice(keys,k,replace=False) if random else keys[np.argsort(-scores.ravel()[keys],kind='stable')[:k]]
    if kind in ('floor','carry'):
        spent=0
        for t in range(trees):
            a,b=offsets[t:t+2];k=quota((b-a)*c,fraction) if kind=='floor' else quota(int(b*c),fraction)-spent
            selected.extend(best(a,b,k));spent+=k
            blocks.append(dict(begin=t,end=t+1,allocated=k,global_limit=quota(int(b*c),fraction)))
    elif kind in ('uniform','score','random'):
        for t in range(0,trees,window):
            end=min(t+window,trees);a,b=offsets[[t,end]];k=quota(int(b*c),fraction)-quota(int(a*c),fraction)
            if kind=='uniform':
                capacities=np.diff(offsets[t:end+1])*c;counts=k*capacities//capacities.sum()
                remainders=k*capacities%capacities.sum();left=k-int(counts.sum())
                counts[np.argsort(-remainders,kind='stable')[:left]]+=1
                assert int(counts.sum())==k
                for tt,kk in zip(range(t,end),counts):selected.extend(best(offsets[tt],offsets[tt+1],int(kk)))
            else:selected.extend(best(a,b,k,kind=='random'))
            blocks.append(dict(begin=t,end=end,allocated=k,global_limit=quota(int(b*c),fraction)))
    else:raise ValueError('unknown reserve allocator')
    keys=np.sort(np.asarray(selected,np.int64));assert len(np.unique(keys))==len(keys)
    assert len(keys)<=quota(scores.size,fraction)
    return keys,blocks


def loss_gradient(raw,y,task):
    raw=np.asarray(raw,float)
    if task=='multiclass':
        p=softmax(raw,axis=1);g=p.copy();g[np.arange(len(raw)),y.astype(int)]-=1
        return float(np.sum(logsumexp(raw,axis=1)-raw[np.arange(len(raw)),y.astype(int)])),g,p*(1-p)
    if task!='multilabel':raise ValueError('unknown task')
    p=expit(raw);return float(np.sum(np.logaddexp(0,raw)-y*raw)),p-y,p*(1-p)


def coordinate_scores(design,raw,y,task,regularization=1.):
    _,g,h=loss_gradient(raw,y,task)
    gg=np.asarray(design.T@g);hh=np.asarray(design.T@h)
    return gg*gg/(2*(hh+regularization))


def sparse_operator(design,keys,outputs):
    keys=np.asarray(keys,np.int64);rows,cols=np.divmod(keys,outputs)
    selected=design[:,rows].tocoo()
    return sparse.csr_matrix((selected.data,(selected.row*outputs+cols[selected.col],selected.col)),shape=(design.shape[0]*outputs,len(keys)))


def objective(coefficients,raw,y,task,forward,backward,regularization):
    z=raw+forward(coefficients);value,g,_=loss_gradient(z,y,task)
    return value+.5*regularization*float(coefficients@coefficients),backward(g)+regularization*coefficients


def solve(raw,y,task,forward,backward,size,regularization=1.,maxiter=60):
    started=perf_counter();initial=np.zeros(size,float)
    if size==0:return initial,dict(status='zero_budget',iterations=0,evaluations=0,seconds=0.,objective_before=loss_gradient(raw,y,task)[0],objective_after=loss_gradient(raw,y,task)[0])
    function=lambda w:objective(w,raw,y,task,forward,backward,regularization)
    before=function(initial)[0];result=minimize(function,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-9,gtol=1e-5,maxls=20))
    if not np.isfinite(result.fun) or result.fun>before+1e-7:raise RuntimeError('fixed-support convex objective increased')
    return result.x,dict(status='converged' if result.success else 'iteration_or_line_search_limit',message=str(result.message),
        iterations=int(result.nit),evaluations=int(result.nfev),seconds=perf_counter()-started,objective_before=before,objective_after=float(result.fun),
        gradient_max_abs=float(np.max(np.abs(result.jac))),regularization=regularization,maxiter=maxiter)


def fit_support(design,keys,raw,y,task,regularization=1.,maxiter=60):
    operator=sparse_operator(design,keys,raw.shape[1])
    forward=lambda w:np.asarray(operator@w).reshape(raw.shape)
    backward=lambda g:np.asarray(operator.T@g.ravel()).ravel()
    values,record=solve(raw,y,task,forward,backward,len(keys),regularization,maxiter)
    record['operator_bytes']=sum(a.nbytes for a in (operator.data,operator.indices,operator.indptr))
    return values.astype(np.float32),record


def fit_extra_rank(design,extra,raw,y,task,regularization=1.,maxiter=60):
    shape=(design.shape[1],extra.shape[1])
    forward=lambda w:np.asarray(design@w.reshape(shape))@extra.T
    backward=lambda g:np.asarray(design.T@(g@extra)).ravel()
    values,record=solve(raw,y,task,forward,backward,int(np.prod(shape)),regularization,maxiter)
    return values.reshape(shape).astype(np.float32),record
