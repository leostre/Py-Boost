"""Equal-size random support without access to any labels."""
import numpy as np


def random_support(residual,count,seed):
    residual=np.asarray(residual)
    if residual.ndim!=2 or not np.isfinite(residual).all():
        raise ValueError('invalid residual')
    eligible=np.flatnonzero(residual.ravel()!=0)
    if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or not 0<=count<=len(eligible):
        raise ValueError('invalid support size')
    return np.random.default_rng(seed).choice(eligible,size=count,replace=False)
