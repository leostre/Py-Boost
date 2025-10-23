from pathlib import Path

SAVE_PATH = Path('./results')
DATA_PATH = Path('./data')

DEFAULTS = dict(
    loss='crossentropy', 
    metric=None,
    ntrees=100,
    lr=0.05,
    min_gain_to_split=0,
    lambda_l2=1,
    gd_steps=1,
    max_depth=6,
    min_data_in_leaf=10,
    colsample=1.,
    subsample=1.,
    quantization='Quantile',
    quant_sample=2000000,
    max_bin=256,
    min_data_in_bin=3,
    es=float('inf'),
    seed=42,
    verbose=10,

    sketch_outputs=1,
    sketch_method='proj',
    use_hess=False,

    callbacks=None,
    sketch_params=None
)


# ============================ Constants

METRICS = [
    # 'bce', 
    # 'logloss', 
    'precision', 
    'recall',  
    'f1', 
    'accuracy', 
    # 'acc', 'auc', 'roc', 'rmse', 'l2', 'rmsle', 'r2', 'r2_score'
]

# ============================ HP lists


SKETCH_METHODS = [
    'filter', 
    'svd', 
    'topk', 
    'rand', 
    'proj', 
    # None
]

LR = [
    0.1,
    0.05,
    0.01,
    0.005
]

# QUANTIZATION = [
#     'Quantile',
#     'Uniform',
#     'Uniquant'
# ]

USE_HESS = [
    True,
    False 
]

SAMPLE_RATIO = [
    0.05, 
    0.25,
    0.5,
    0.75
]




