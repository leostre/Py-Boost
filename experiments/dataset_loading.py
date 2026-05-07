from ucimlrepo import fetch_ucirepo
import numpy as np

RANDOM_SEED = 42
TRAIN_SIZE = 0.8
STRATIFY = True 
CLASS_THRESHOLD = 0.05

def _filter_minimal(X, y):
    y = np.array(y)
    counts = np.unique(y, return_counts=True)

def _encode_order(y):
    mapping = {label: i for i, label in enumerate(np.unique(y))}
    return y.map(lambda x: mapping[x])

__all__ = (
    'yeast',
    'wine'
)

def yeast():
    dataset = fetch_ucirepo('yeast')
    X = dataset['data']['features']
    y = dataset['data']['targets']
    y = _encode_order(y)
    return X.values, y.values, len(np.unique(y))

def wine():
    dataset = fetch_ucirepo('wine')
    X = dataset['data']['features']
    y = dataset['data']['targets']
    y = _encode_order(y)
    return X.values, y.values, len(np.unique(y))


__locals = locals() 

DATASETS = {
    name: __locals[name] for name in __all__
}