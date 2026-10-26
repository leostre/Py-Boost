"""Post-seal diagnostics, never a model-selection or budget-control input."""
import numpy as np


def training_profile(features, target, train_raw, task):
    x, z = np.asarray(features), np.asarray(train_raw, float)
    c = z.shape[1]
    centered = z-z.mean(axis=0)
    if task == 'multiclass':
        centered -= centered.mean(axis=1, keepdims=True)
        support = np.bincount(np.asarray(target, int), minlength=c)
    else:
        support = np.asarray(target).sum(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    p = energy[energy > max(total*1e-24, 0)]/total if total else np.empty(0)
    return dict(train_samples=len(x), features=x.shape[1], outputs=c, task=task,
                train_label_density=float(support.sum()/len(x)/c),
                train_label_cardinality=float(support.sum()/len(x)),
                minimum_train_support=int(support.min()), rare_train_outputs=int((support <= .01*len(x)).sum()),
                train_constant_features=int(np.sum(np.ptp(x, axis=0) == 0)),
                prediction_effective_rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
                prediction_stable_rank=total/float(energy[0]) if total else 0.,
                prediction_rank_95=int(np.searchsorted(np.cumsum(energy), .95*total)+1) if total else 0)


def per_label(target, predicted, task, outputs):
    if task == 'multiclass':
        truth = np.eye(outputs, dtype=bool)[np.asarray(target, int)]
        chosen = np.eye(outputs, dtype=bool)[np.asarray(predicted, int)]
    elif task == 'multilabel':
        truth, chosen = np.asarray(target, bool), np.asarray(predicted, bool)
    else:
        raise ValueError('unknown diagnostic task')
    if truth.shape != chosen.shape or truth.ndim != 2 or truth.shape[1] != outputs:
        raise ValueError('diagnostic label shape')
    support = truth.sum(axis=0)
    tp = (truth & chosen).sum(axis=0)
    fp = (~truth & chosen).sum(axis=0)
    fn = (truth & ~chosen).sum(axis=0)
    den = 2*tp+fp+fn
    scores = np.divide(2*tp, den, out=np.zeros(outputs, float), where=den > 0)
    used = (support+chosen.sum(axis=0) > 0) if task == 'multiclass' else np.ones(outputs, bool)
    macro = float(scores[used].mean()) if used.any() else 0.
    weighted = float(np.sum(scores*support)/support.sum()) if support.sum() else 0.
    return dict(support=support, tp=tp, fp=fp, fn=fn, f1=scores, used=used,
                macro=macro, weighted=weighted)


def compare_labels(target, full_labels, candidate_labels, train_target, task, outputs):
    full, current = per_label(target, full_labels, task, outputs), per_label(target, candidate_labels, task, outputs)
    train_support = (np.bincount(np.asarray(train_target, int), minlength=outputs) if task == 'multiclass' else
                     np.asarray(train_target).sum(axis=0)).astype(int)
    rare = train_support <= .01*len(train_target)
    rows = [dict(label=j, train_positive=int(train_support[j]), test_positive=int(full['support'][j]),
                 rare_train=bool(rare[j]), full_f1=float(full['f1'][j]), f1=float(current['f1'][j]),
                 delta_f1=float(current['f1'][j]-full['f1'][j]), full_tp=int(full['tp'][j]), tp=int(current['tp'][j]),
                 full_fp=int(full['fp'][j]), fp=int(current['fp'][j]), full_fn=int(full['fn'][j]), fn=int(current['fn'][j]),
                 full_macro_member=bool(full['used'][j]), macro_member=bool(current['used'][j])) for j in range(outputs)]
    rare_full = float(full['f1'][rare].mean()) if rare.any() else None
    rare_current = float(current['f1'][rare].mean()) if rare.any() else None
    return rows, dict(rare_labels=int(rare.sum()), rare_full_macro_fixed_domain=rare_full,
        rare_macro_fixed_domain=rare_current,
        rare_macro_retained=(rare_current/rare_full) if rare_full else None,
        label_f1_losses=int((current['f1'] < full['f1']-1e-12).sum()),
        label_f1_gains=int((current['f1'] > full['f1']+1e-12).sum()),
        full_macro_domain=int(full['used'].sum()), macro_domain=int(current['used'].sum()),
        independent_macro=current['macro'], independent_weighted=current['weighted'])
