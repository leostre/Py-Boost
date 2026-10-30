"""Descriptive training/development properties; no test inputs or fitted rule."""
import numpy as np


def training_profile(data,task,outputs):
    def indicator(y):return np.eye(outputs)[np.asarray(y,int)] if task=='multiclass' else np.asarray(y,float)
    y=indicator(data['train_y']);dev=indicator(data['dev_y']);x=data['train_x'];support=y.sum(0);dev_support=dev.sum(0)
    centered=y-y.mean(0);eigen=np.maximum(np.linalg.eigvalsh(centered.T@centered),0.)
    positive=eigen[eigen>max(float(eigen.max(initial=0))*1e-12,1e-15)]
    p=positive/positive.sum() if len(positive) else positive
    effective=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.
    return dict(train_rows=len(y),development_rows=len(dev),features=x.shape[1],outputs=outputs,task=task,
        feature_zero_fraction=float(np.mean(x==0)),feature_missing_fraction=float(np.mean(np.isnan(x))),
        label_cardinality=float(y.sum(1).mean()),label_density=float(y.mean()),train_label_support=support.astype(int).tolist(),
        train_support_zero=int(np.count_nonzero(support==0)),train_support_below10=int(np.count_nonzero(support<10)),
        development_support_zero=int(np.count_nonzero(dev_support==0)),target_covariance_effective_rank=effective,
        scope='descriptive target covariance, not gradient effective rank, predictive meta-model or causal explanation')
