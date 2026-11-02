"""Explicit bridge from old threshold archives to current shared-bias blocks."""
from dataclasses import replace
import budget_quality_worker
from experiments.block_leaf_model import load_prediction_model
from abl_protected_core import as_protected

def load_saved_serving(path):
    model=load_prediction_model(path)
    if hasattr(model,'blocks'):return model
    shifts=model.decision_shifts.copy()
    return replace(as_protected(model),decision_shifts=shifts)
