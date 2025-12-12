from logging import Logger
import os
import io
import tarfile
from pathlib import Path 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Generator, Optional
from scipy.io import arff
from huggingface_hub import hf_hub_download

from experiments.data.dataloader import DatasetMetadata, DatasetLoader, setup_logging
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

def load_mbd(fold: int = 1, modality: str = 'all', base_data_path: Optional[str] = None, logger: Optional[Logger] = None):
    if logger is None:
        logger = setup_logging('load_mbd', verbose=True)
    if base_data_path is None:
        base_data_path = os.path.join(EXPERIMENTS_DATA_PATH, 'mbd')

    os.makedirs(base_data_path, exist_ok=True)

    client_split_path = Path(base_data_path) / 'client_split.tar.gz'
    if not client_split_path.exists():
        logger.info(f"Downloading client_split.tar.gz to {base_data_path}...")
        hf_hub_download(repo_id="ai-lab/MBD-mini",
                        filename="client_split.tar.gz",
                        repo_type="dataset",
                        local_dir=base_data_path)

    logger.info(f"Extracting client IDs for fold {fold} from client_split.tar.gz...")
    fold_client_ids = []
    with tarfile.open(client_split_path, 'r:gz') as tar:
        fold_pattern = f"client_split/fold={fold}/"

        for member in tar.getmembers():
            if member.name.endswith('.parquet') and fold_pattern in member.name:
                f = tar.extractfile(member)
                df = pd.read_parquet(io.BytesIO(f.read()))

                if 'client_id' in df.columns:
                    fold_client_ids.extend(df['client_id'].tolist())

    logger.info(f"Found {len(fold_client_ids)} unique clients in fold {fold}")

    fold_client_ids_set = set(fold_client_ids)

    targets_path = Path(base_data_path) / 'targets.tar.gz'
    if not targets_path.exists():
        logger.info(f"Downloading targets.tar.gz to {base_data_path}...")
        hf_hub_download(repo_id="ai-lab/MBD-mini",
                        filename="targets.tar.gz",
                        repo_type="dataset",
                        local_dir=base_data_path)

    logger.info(f"Loading targets for fold {fold}...")
    targets_dfs = []
    with tarfile.open(targets_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.parquet'):
                f = tar.extractfile(member)
                df = pd.read_parquet(io.BytesIO(f.read()))

                if 'fold' in df.columns:  # filter by fold column if it exists, otherwise by client_id
                    fold_targets = df[df['fold'] == fold]
                elif 'client_id' in df.columns:
                    fold_targets = df[df['client_id'].isin(fold_client_ids_set)]
                else:
                    continue
                
                if not fold_targets.empty:
                    targets_dfs.append(fold_targets)

    targets_df = pd.concat(targets_dfs, ignore_index=True) if targets_dfs else pd.DataFrame()
    logger.info(f"Loaded {len(targets_df)} target rows")

    logger.info(f"Loading features for modality '{modality}'...")

    if modality == 'detail':
        feature_file = 'detail.tar.gz'
    else:  # 'all', 'dialog', 'geo', 'trx', 'ptls'
        feature_file = 'ptls.tar.gz'

    feature_path = Path(base_data_path) / feature_file
    if not feature_path.exists():
        logger.info(f"Downloading {feature_file} to {base_data_path}...")
        hf_hub_download(repo_id="ai-lab/MBD-mini",
                        filename=feature_file,
                        repo_type="dataset",
                        local_dir=base_data_path)

    feature_dfs = []
    with tarfile.open(feature_path, 'r:gz') as tar:
        files_processed = 0
        for member in tar.getmembers():
            if member.name.endswith('.parquet'):
                files_processed += 1
                f = tar.extractfile(member)
                df = pd.read_parquet(io.BytesIO(f.read()))

                if 'fold' in df.columns:  # filter for our fold
                    fold_features = df[df['fold'] == fold]
                elif 'client_id' in df.columns:
                    fold_features = df[df['client_id'].isin(fold_client_ids_set)]
                else:
                    continue

                if not fold_features.empty:
                    feature_dfs.append(fold_features)

    features_df = pd.concat(feature_dfs, ignore_index=True) if feature_dfs else pd.DataFrame()
    logger.info(f"Loaded {len(features_df)} feature rows from {files_processed} files")

    if feature_file == 'ptls.tar.gz' and modality in ['dialog', 'geo', 'trx']:
        logger.info(f"Filtering for {modality} features...")

        if not features_df.empty:
            columns_to_keep = ['client_id', 'fold'] if 'fold' in features_df.columns else ['client_id']
            if modality == 'dialog' and 'dialog' in features_df.columns:
                columns_to_keep.append('dialog')
            elif modality == 'geo' and 'geo' in features_df.columns:
                columns_to_keep.append('geo')
            elif modality == 'trx' and 'trx' in features_df.columns:
                columns_to_keep.append('trx')

            features_df = features_df[columns_to_keep]

    logger.info(f"Features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
    logger.info(f"Targets:  {targets_df.shape[0]} rows, {targets_df.shape[1]} columns")

    if not features_df.empty:
        logger.info(f"Feature columns: {features_df.columns.tolist()}")
    if not targets_df.empty:
        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        logger.info(f"Target columns: {target_cols}")

    return {
        'features': features_df,
        'target': targets_df,
        'metadata': DatasetMetadata(name='mbd', source='custom', shape=features_df.shape)
    }


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
    processed_datasets = {}

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

            is_multilabel = True
            if len(processed_target.shape) == 1:
                processed_target = LabelBinarizer().fit_transform(processed_target)
            
            metadata.processed_shape = processed_features.shape

            processed_dataset = {
                'features': processed_features,
                'target': processed_target,
                'metadata': metadata,
            }
        except: pass
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
    'mnist': {'id': 'mnist_784', 'version': 1, 'source': 'openml'},
    'mbd': {'id': 'ai-lab/MBD-mini', 'source': 'custom', 'method': load_mbd}
}

SPECIAL_LOADERS = {
    'age_prediction': {'loader': load_age_prediction, 'n_classes': 4, 'task': 'onelabel'},
    'mbd': {'loader': load_mbd}
}


if __name__ == "__main__":
    dataset_config = DATASETS
    # dataset_config = {'age_prediction': {}}
    datasets = load_and_preprocess_datasets(dataset_config)
