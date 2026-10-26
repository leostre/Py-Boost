"""Only the exact quality function from 5fdca6e; no runtime/launcher imports."""
import numpy as np
from sklearn.metrics import f1_score
from experiments.compactness_core import normalize_multiclass_probabilities


def quality(target, probabilities, task):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError('nonfinite predictions')
    if task == 'multiclass':
        probabilities = normalize_multiclass_probabilities(probabilities)
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        loss = float(-np.log(p[np.arange(len(target)), target.astype(int)]).mean())
        predicted = probabilities.argmax(axis=1)
    else:
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise RuntimeError('multilabel prediction outside [0, 1]')
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        loss = float(-(target * np.log(p) + (1 - target) * np.log1p(-p)).mean())
        predicted = probabilities >= .5
    return {'loss': loss,
            'f1_weighted': float(f1_score(target, predicted, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(target, predicted, average='macro', zero_division=0))}
