import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Generator
from scipy.io import arff

from experiments.data.dataloader import DatasetMetadata, DatasetLoader, setup_logging
from experiments.dataset_loading import RANDOM_SEED
from py_boost.paths import EXPERIMENTS_DATA_PATH


def load_age_prediction():
    fold = 0
    assert os.path.exists(EXPERIMENTS_DATA_PATH + '/age_pred')
    while os.path.exists(EXPERIMENTS_DATA_PATH + f'/age_pred/fold_{fold}'):
        test = pd.read_csv(EXPERIMENTS_DATA_PATH + f'/age_pred/fold_{fold}/emb_test.csv')
        train = pd.read_csv(EXPERIMENTS_DATA_PATH + f'/age_pred/fold_{fold}/emb_train.csv')
        ytr = train['bins']
        yte = test['bins']
        xtr = train.drop(['bins', 'client_id'], axis=1)
        xte = test.drop(['bins', 'client_id'], axis=1)
        # pca = PCA(20)
        # xtr = pca.fit_transform(train)
        # xte = pca.transform(test)
        lb = LabelBinarizer()
        ytr = lb.fit_transform(ytr)
        yte = lb.transform(yte)
        yield xtr, ytr, xte, yte, fold
        fold += 1
    assert fold > 0, 'No folds were returned'


def load_mnist():
    fold = 0
    path = EXPERIMENTS_DATA_PATH + '/mnist'
    assert os.path.exists(path)
    while os.path.exists(path + f'/fold_{fold}'):
        fold_path = path + f'/fold_{fold}'
        xtr = np.load(fold_path + '/xtr.npy')
        ytr = np.load(fold_path + '/ytr.npy')
        xte = np.load(fold_path + '/xte.npy')
        yte = np.load(fold_path + '/yte.npy')
        yield xtr, ytr, xte, yte, fold
        fold += 1
    assert fold > 0, 'No folds were returned'


def load_cifar10():
    fold = 0
    path = EXPERIMENTS_DATA_PATH + '/cifar10'
    assert os.path.exists(path)
    while os.path.exists(path + f'/fold_{fold}'):
        fold_path = path + f'/fold_{fold}'
        xtr = np.load(fold_path + '/xtr.npy')
        ytr = np.load(fold_path + '/ytr.npy')
        xte = np.load(fold_path + '/xte.npy')
        yte = np.load(fold_path + '/yte.npy')
        lb = LabelBinarizer()
        ytr = lb.fit_transform(ytr)
        yte = lb.transform(yte)ы
        yield xtr, ytr, xte, yte, fold
        fold += 1
    assert fold > 0, 'No folds were returned'


def load_mediamill():
    raw_data, meta = arff.loadarff(os.path.join(EXPERIMENTS_DATA_PATH, 'mediamill', 'mediamill.arff'))
    data = pd.DataFrame(raw_data)
    xcols = [c for c in data.columns if c.startswith('Att')]
    ycols = [c for c in data.columns if c.startswith('Cl')]
    y =  data[ycols].astype(str).astype(int)
    X = data[xcols]
    return {
        'features': X,
        'target': y,
        'metadata': DatasetMetadata(
                    name='mediamill',
                    source='custom',
                    shape=X.shape,
                )
    }


def load_mbd(prop_keep=0.05):
    def drop_empty(X, y, prop_keep):
        np.random.seed(RANDOM_SEED)
        valuable_mask = y.any(axis=1)
        full_idx = np.arange(len(y))
        empty_idx = full_idx[~valuable_mask]
        idx2keep = np.random.permutation(empty_idx)[:int(len(y) * prop_keep)]
        final_idx = np.concat([full_idx[valuable_mask], idx2keep])
        np.random.shuffle(final_idx)
        return X[final_idx], y[final_idx]
    fold = 0
    path = EXPERIMENTS_DATA_PATH + '/mbd'
    assert os.path.exists(path)
    while os.path.exists(path + f'/fold_{fold}'):
        fold_path = path + f'/fold_{fold}'
        xtr = pd.read_parquet(fold_path + '/xtr.parquet').values
        ytr = pd.read_parquet(fold_path + '/ytr.parquet').values
        xte = pd.read_parquet(fold_path + '/xte.parquet').values
        yte = pd.read_parquet(fold_path + '/yte.parquet').values
        xtr, ytr = drop_empty(xtr, ytr, prop_keep)
        yield xtr, ytr, xte, yte, fold
        fold += 1
    assert fold > 0, 'No folds were returned'


# TODO: remove/refactor legacy method
def split_benchmark_data(dataset_dict: dict, use_subsample=None):
    X = dataset_dict['features'].values.astype('float32')
    y = dataset_dict['target'].values
    if y.dtype == object:
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return dict(train_features=X,
                test_features=X_test,
                train_target=y,
                test_target=y_test)


def preprocess_dataset(X, dataset_name: str):
    logger = setup_logging()
    logger.info(f"Starting preprocessing for dataset: {dataset_name}")

    if isinstance(X, pd.Series):
        X = X.to_frame()
        was_series = True
    else:
        was_series = False

    logger.info(f"Initial shape: {X.shape}, data types: {X.dtypes.value_counts().to_dict()}")

    X_processed = X.copy()
    nan_count = X_processed.isna().sum().sum()
    
    if nan_count > 0:
        logger.warning(f"Dataset {dataset_name} contains {nan_count} NaN values")
        initial_shape = X_processed.shape
        X_processed = X_processed.dropna(thresh=X_processed.shape[1] // 2)
        rows_dropped = initial_shape[0] - X_processed.shape[0]
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with excessive NaN values in {dataset_name}")

        for column in X_processed.columns:
            if X_processed[column].isna().any():
                column_nan_count = X_processed[column].isna().sum()
                logger.info(f"Processing column {column} with {column_nan_count} NaN values in {dataset_name}")
                if X_processed[column].dtype == 'object':
                    fill_value = (X_processed[column].mode()[0] 
                                if not X_processed[column].mode().empty 
                                else 'missing')
                    X_processed[column].fillna(fill_value, inplace=True)
                    logger.debug(f"Filled {column_nan_count} NaN values in categorical column {column} with mode: {fill_value}")
                else:
                    fill_value = X_processed[column].median()
                    X_processed[column].fillna(fill_value, inplace=True)
                    logger.debug(f"Filled {column_nan_count} NaN values in numerical column {column} with median: {fill_value}")
    else:
        logger.info(f"No NaN values found in dataset {dataset_name}")

    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    categorical_count = len(categorical_cols)
    
    if categorical_count > 0:
        logger.info(f"Encoding {categorical_count} categorical columns in {dataset_name}")
        encoders = {}
        for col in categorical_cols:
            unique_count = X_processed[col].nunique()
            logger.info(f"Encoding categorical column {col} with {unique_count} unique values")
            encoder = LabelEncoder()
            X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
            encoders[col] = encoder
            logger.debug(f"Completed encoding for column {col}")
    else:
        logger.info(f"No categorical columns found in dataset {dataset_name}")

    if was_series:
        X_processed = X_processed.iloc[:, 0]

    return X_processed


def load_and_preprocess_datasets(dataset_config: Dict[str, Any], to_numpy=True) -> Generator[str, Dict, Any]:
    loader = DatasetLoader(dataset_config)
    datasets = loader.dataset_gen()
    n_classes = processed_dataset = None

    for name, dataset_info in datasets:
        n_classes = processed_dataset = None
        try:
            features = dataset_info['features']
            target = dataset_info['target']
            metadata = dataset_info['metadata']

            n_classes = target.nunique() if len(target.shape) == 1 else target.shape[1]

            processed_features = preprocess_dataset(features, name)
            processed_target = preprocess_dataset(target, f"{name}_target")

            if to_numpy:
                processed_features = np.array(processed_features)
                processed_target = np.array(processed_target)

            if len(processed_target.shape) == 1:
                processed_target = LabelBinarizer().fit_transform(processed_target)
            
            metadata.processed_shape = processed_features.shape

            processed_dataset = {
                'features': processed_features,
                'target': processed_target,
                'metadata': metadata,
            }
        except:
            pass
        finally:
            yield name, processed_dataset, n_classes


DATASETS = {
    'yeast': {'id': 'yeast', 'source': 'openml', 'version': 1},
    'genbase': {'id': 'genbase', 'source': 'openml', 'version': 2},
    'birds': {'id': 'birds', 'source': 'openml', 'version': 3},
    'rt_iot2022': {'id': 942, 'source': 'uci'},
    'age_prediction': {},
    'mediamill': {'id': 'mediamill', 'source': 'custom',
                      'method': load_mediamill,
                      'method_params': {}},
    'mnist': {'id': 'mnist_784', 'source': 'custom', 'method': load_mnist},
    'cifar10': {'id': 'cifar10', 'source': 'custom', 'method': load_cifar10},
    'mbd': {'id': 'ai-lab/MBD-mini', 'source': 'custom', 'method': load_mbd}
}

SPECIAL_LOADERS = {
    'mnist': {'loader': load_mnist, 'n_classes': 10, 'task': 'onelabel'},
    'cifar10': {'loader': load_cifar10, 'n_classes': 10, 'task': 'onelabel'},
    'age_prediction': {'loader': load_age_prediction, 'n_classes': 4, 'task': 'onelabel'},
    'mbd': {'loader': load_mbd, 'n_classes': 4, 'task': 'multilabel'}
}
