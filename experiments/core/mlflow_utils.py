from contextlib import contextmanager
from typing import Dict, Optional

import mlflow


@contextmanager
def start_run_for_dataset(dataset_name: str, run_name: str):
    """
    Convenience wrapper to start an MLflow run with a standard name.
    """
    with mlflow.start_run(run_name=f"{dataset_name}_{run_name}") as run:
        yield run


def log_param_dict(params: Dict, allowed_keys: Optional[list] = None) -> None:
    """
    Log a filtered dict of parameters into the current MLflow run.
    """
    if allowed_keys is not None:
        params = {k: v for k, v in params.items() if k in allowed_keys}
    if params:
        mlflow.log_params(params)

