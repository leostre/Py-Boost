import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def optimize_global_threshold(y_true, y_proba, metric=f1_score):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_score = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = metric(y_true, y_pred, average="micro")
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def optimize_per_label_thresholds(y_true, y_proba, metric=f1_score):
    n_classes = y_true.shape[1]
    best_thresholds = []
    best_scores = []

    for i in range(n_classes):
        thresh, score = optimize_global_threshold(
            y_true[:, i : i + 1],
            y_proba[:, i : i + 1],
            metric=metric,
        )
        best_thresholds.append(thresh)
        best_scores.append(score)

    return np.array(best_thresholds), np.array(best_scores)


def predict_with_per_label_thresholds(proba, thresholds):
    return (proba >= thresholds).astype(int)


def get_metrics_separate_thresholds(model, xte, yte, xval, yval, metrics: dict):
    res = {}
    proba = model.predict(xte)
    res["rocauc"] = roc_auc_score(yte, proba, average="weighted")
    thrs = optimize_per_label_thresholds(yval, model.predict(xval))[0]
    proba = model.predict(xte)
    ypred = predict_with_per_label_thresholds(proba, thrs)
    res.update({name: func(yte, ypred) for name, func in metrics.items()})
    return res
