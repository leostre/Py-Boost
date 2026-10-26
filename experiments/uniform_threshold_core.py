"""Identical decision calibration and standalone artifacts for all families."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from experiments.shared_leaf_basis import LeafEnsemble, probabilities
from experiments.sparse_leaf_core import SparseLeafModel, leaf_design
from experiments.output_calibration_core import fit_decision_shifts, calibrated_quality, labels_from_raw


def threshold_variants(raw_calibration, calibration_target, raw_selection, selection_target,
                       task, grid, shrinkages):
    """Fit on calibration only; return selection metrics and explicit C-vectors."""
    fractions = sorted(set([0., *map(float, shrinkages)]))
    if not all(0 <= f <= 1 for f in fractions):
        raise ValueError('threshold shrinkages outside [0, 1]')
    fitted = fit_decision_shifts(raw_calibration, calibration_target, task, grid)
    return [{'threshold_fraction': f, 'decision_shifts': (f*fitted).astype(np.float32).tolist(),
             'validation': calibrated_quality(selection_target, raw_selection, task,
                                               (f*fitted).astype(np.float32))}
            for f in fractions]


def choice_key(row):
    return (-row['validation']['f1_macro'], row['validation']['loss'], row['payload_bytes'],
            row['parameters'], row['candidate_id'])


def family_allows(row, family, reference_parameters):
    if row['kind'] == 'rank_only':
        return family in ('rank_only', 'sparse_01', 'sparse_02', 'sparse_05')
    percent = {'sparse_01': 1, 'sparse_02': 2, 'sparse_05': 5}.get(family)
    return percent is not None and row['kind'] == 'sparse' and row['nnz']*100 <= percent*reference_parameters


@dataclass(frozen=True)
class ThresholdModel:
    base: LeafEnsemble
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    decision_shifts: np.ndarray

    def __post_init__(self):
        if (self.decision_shifts.dtype != np.float32 or self.decision_shifts.ndim != 1
                or len(self.decision_shifts) != len(self.base.bias)
                or not np.isfinite(self.decision_shifts).all()):
            raise ValueError('all families must retain exactly C finite float32 thresholds')
        if self.base.representation == 'low_rank':
            SparseLeafModel(self.base, self.rows, self.columns, self.values)
        elif (self.rows.dtype != np.int32 or self.columns.dtype != np.int32
              or self.values.dtype != np.float32
              or any(a.shape != (0,) for a in (self.rows, self.columns, self.values))):
            raise ValueError('full reference cannot retain a residual')

    @property
    def parameter_count(self):
        return self.base.parameter_count + len(self.values) + len(self.decision_shifts)

    @property
    def payload_bytes(self):
        return self.base.payload_bytes + sum(a.nbytes for a in
            (self.rows, self.columns, self.values, self.decision_shifts))

    def raw_from_design(self, design):
        if self.base.representation == 'low_rank':
            return SparseLeafModel(self.base, self.rows, self.columns, self.values).raw_from_design(design)
        raw = design @ self.base.coefficients
        if self.base.representation == 'reference_class':
            raw = np.column_stack((raw, np.zeros(len(raw))))
        return raw + self.base.bias

    def predict(self, features):
        design = leaf_design(self.base.leaf_indices(features), len(self.base.coefficients))
        return probabilities(self.raw_from_design(design), self.base.task)

    def predict_labels(self, features):
        design = leaf_design(self.base.leaf_indices(features), len(self.base.coefficients))
        return labels_from_raw(self.raw_from_design(design), self.base.task, self.decision_shifts)

    def save(self, path):
        names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
        np.savez(path, version=np.array(4), **{n: getattr(self.base, n) for n in names},
                 representation=np.array(self.base.representation), n_features=np.array(self.base.n_features),
                 task=np.array(self.base.task), rows=self.rows, columns=self.columns, values=self.values,
                 decision_shifts=self.decision_shifts)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')
            expected = {*names, 'version', 'representation', 'n_features', 'task',
                        'rows', 'columns', 'values', 'decision_shifts'}
            if set(data.files) != expected or int(data['version']) != 4:
                raise ValueError('unknown uniform-threshold model schema')
            base = LeafEnsemble(**{n: data[n].copy() for n in names}, n_features=int(data['n_features']),
                                task=str(data['task']), representation=str(data['representation']))
            return cls(base, *(data[n].copy() for n in ('rows', 'columns', 'values', 'decision_shifts')))
