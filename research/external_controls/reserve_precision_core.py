"""Same convex objective, scaled coordinates and exact multilabel separation."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from window_reserve_core import sparse_operator,objective,loss_gradient


def scaled_solve(raw,y,task,forward,backward,initial,diagonal,lam=1.,maxiter=300):
    initial=np.asarray(initial,float);diagonal=np.asarray(diagonal,float);started=perf_counter()
    if len(initial)==0:return initial,dict(status='zero_budget',iterations=0,evaluations=0,seconds=0.,gradient_max_abs=0.)
    scale=1/np.sqrt(np.maximum(diagonal,lam));base=lambda w:objective(w,raw,y,task,forward,backward,lam)
    before,gb=base(initial)
    def function(u):
        value,gradient=base(initial+scale*u);return value,gradient*scale
    result=minimize(function,np.zeros_like(initial),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-7,maxls=30))
    coefficients=initial+scale*result.x;after,g=base(coefficients)
    if not np.isfinite(after) or after>before+1e-6:raise RuntimeError('precision refit increased original objective')
    return coefficients,dict(status='converged' if result.success else 'iteration_or_line_search_limit',iterations=int(result.nit),evaluations=int(result.nfev),
        objective_before=before,objective_after=after,gradient_before=float(np.max(np.abs(gb))),gradient_max_abs=float(np.max(np.abs(g))),seconds=perf_counter()-started)


def refine_support(design,keys,initial,raw,y,task,lam=1.,maxiter=300):
    c=raw.shape[1];keys=np.asarray(keys,np.int64);initial=np.asarray(initial,float);records=[];coefficients=np.zeros_like(initial)
    if task=='multilabel':
        for output in np.unique(keys%c):
            idx=np.flatnonzero(keys%c==output);x=design[:,keys[idx]//c].tocsr();z=raw[:,output:output+1];yy=y[:,output:output+1]
            forward=lambda w:np.asarray(x@w).reshape(-1,1)
            backward=lambda g:np.asarray(x.T@g[:,0]).ravel()
            p=expit(z[:,0]+x@initial[idx]);diagonal=np.asarray(x.multiply(x).T@(p*(1-p))).ravel()+lam
            values,record=scaled_solve(z,yy,task,forward,backward,initial[idx],diagonal,lam,maxiter);coefficients[idx]=values;record['output']=int(output);records.append(record)
    else:
        operator=sparse_operator(design,keys,c);forward=lambda w:np.asarray(operator@w).reshape(raw.shape);backward=lambda g:np.asarray(operator.T@g.ravel()).ravel()
        _,_,h=loss_gradient(raw+forward(initial),y,task);diagonal=np.asarray(operator.multiply(operator).T@h.ravel()).ravel()+lam
        coefficients,record=scaled_solve(raw,y,task,forward,backward,initial,diagonal,lam,maxiter);records.append(record)
    return coefficients.astype(np.float32),dict(subproblems=records,all_converged=all(r['status'] in ('converged','zero_budget') for r in records),
        maximum_gradient=max((r['gradient_max_abs'] for r in records),default=0.),seconds=sum(r['seconds'] for r in records))


def refine_rank(design,extra,initial,raw,y,task,lam=1.,maxiter=300):
    shape=initial.shape;forward=lambda w:np.asarray(design@w.reshape(shape))@extra.T
    backward=lambda g:np.asarray(design.T@(g@extra)).ravel()
    current=raw+forward(initial.ravel())
    if task=='multiclass':
        from scipy.special import softmax
        p=softmax(current,axis=1);diagonal=p@(extra*extra)-(p@extra)**2
    else:p=expit(current);diagonal=(p*(1-p))@(extra*extra)
    diagonal=np.asarray(design.multiply(design).T@diagonal)+lam
    values,record=scaled_solve(raw,y,task,forward,backward,initial.ravel(),diagonal.ravel(),lam,maxiter)
    return values.reshape(shape).astype(np.float32),dict(subproblems=[record],all_converged=record['status']=='converged',maximum_gradient=record['gradient_max_abs'],seconds=record['seconds'])
