"""Packed varying-basis blocks with one shared bias/decision vector.

No duplicate per-block bias or thresholds are serialized. A shared immutable
zero C-vector is the only extra runtime constant for the reference adapters.
This CPU predictor is a correctness/reference implementation,not a new GPU
latency claim. Frozen H-K3 inference implementations remain unchanged.
"""
from dataclasses import dataclass,replace
from pathlib import Path
import numpy as np

from experiments.shared_leaf_basis import LeafEnsemble,probabilities
from experiments.uniform_threshold_core import ThresholdModel
from experiments.prepared_leaf_inference import frozen_array
from experiments.output_calibration_core import labels_from_raw
from experiments.sparse_leaf_core import leaf_design

ARRAY_NAMES = ('nodes','node_offsets','leaf_offsets','coefficients','basis')
SPARSE_NAMES = ('rows','columns','values')


@dataclass(frozen=True)
class BlockLeafModel:
    blocks: tuple
    bias: np.ndarray
    decision_shifts: np.ndarray

    def __post_init__(self):
        if not isinstance(self.blocks,tuple) or not self.blocks:
            raise ValueError('nonempty block tuple required')
        for a in (self.bias,self.decision_shifts):
            if a.dtype != np.float32 or a.ndim != 1 or not np.isfinite(a).all():
                raise ValueError('invalid shared vectors')
        if self.bias.shape != self.decision_shifts.shape:
            raise ValueError('shared vector widths differ')
        task,features = self.blocks[0].base.task,self.blocks[0].base.n_features
        zero = frozen_array(np.zeros(len(self.bias),np.float32))
        normalized = []
        for block in self.blocks:
            if (block.base.task != task or block.base.n_features != features or len(block.base.bias) != len(zero)
                    or np.any(block.base.bias != 0) or np.any(block.decision_shifts != 0)):
                raise ValueError('blocks must share task/features and have zero local vectors')
            normalized.append(replace(block,base=replace(block.base,bias=zero),decision_shifts=zero))
        object.__setattr__(self,'blocks',tuple(normalized))
        object.__setattr__(self,'_zero',zero)

    @property
    def task(self):
        return self.blocks[0].base.task

    @property
    def n_features(self):
        return self.blocks[0].base.n_features

    @property
    def n_trees(self):
        return sum(b.base.n_trees for b in self.blocks)

    @property
    def parameter_count(self):
        return sum(b.base.coefficients.size+b.base.basis.size+len(b.values) for b in self.blocks)+self.bias.size+self.decision_shifts.size

    @property
    def payload_bytes(self):
        return sum(sum(getattr(b.base,k).nbytes for k in ARRAY_NAMES)+sum(getattr(b,k).nbytes for k in SPARSE_NAMES)
                   for b in self.blocks)+self.bias.nbytes+self.decision_shifts.nbytes

    @property
    def resident_array_bytes(self):
        return self.payload_bytes+self._zero.nbytes

    def raw(self,features):
        x = np.asarray(features,dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features or np.isinf(x).any():
            raise ValueError('invalid feature matrix')
        out = np.broadcast_to(self.bias,(len(x),len(self.bias))).astype(float,copy=True)
        if len(x):
            for b in self.blocks:
                out += b.raw_from_design(leaf_design(b.base.leaf_indices(x),len(b.base.coefficients)))
        return out

    def predict(self,features):
        scores = self.raw(features)
        return probabilities(scores,self.task) if len(scores) else scores

    def predict_labels(self,features):
        return labels_from_raw(self.raw(features),self.task,self.decision_shifts)

    def with_shifts(self,shifts):
        return replace(self,decision_shifts=np.asarray(shifts,dtype=np.float32))

    def save(self,path):
        arrays = dict(version=np.array(6),block_count=np.array(len(self.blocks)),task=np.array(self.task),
            n_features=np.array(self.n_features),bias=self.bias,decision_shifts=self.decision_shifts)
        for i,b in enumerate(self.blocks):
            arrays[f'b{i}_representation'] = np.array(b.base.representation)
            arrays.update({f'b{i}_{k}':getattr(b.base,k) for k in ARRAY_NAMES})
            arrays.update({f'b{i}_{k}':getattr(b,k) for k in SPARSE_NAMES})
        np.savez(path,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as f:
            def scalar(key,kinds):
                a = f[key]
                if a.ndim != 0 or a.dtype.kind not in kinds:
                    raise ValueError('invalid scalar model metadata: '+key)
                return a.item()
            if scalar('version','iu') != 6:
                raise ValueError('unknown block model schema')
            count = int(scalar('block_count','iu'))
            if not 1 <= count <= 10000:
                raise ValueError('invalid block count')
            expected = {'version','block_count','task','n_features','bias','decision_shifts'}
            expected.update(f'b{i}_{k}' for i in range(count) for k in (*ARRAY_NAMES,*SPARSE_NAMES,'representation'))
            if set(f.files) != expected:
                raise ValueError('unexpected/missing block model field')
            bias,shifts = f['bias'].copy(),f['decision_shifts'].copy()
            zero = np.zeros_like(bias)
            task,features = str(scalar('task','US')),int(scalar('n_features','iu'))
            blocks = []
            for i in range(count):
                base = LeafEnsemble(**{k:f[f'b{i}_{k}'].copy() for k in ARRAY_NAMES},bias=zero,
                    n_features=features,task=task,representation=str(scalar(f'b{i}_representation','US')))
                blocks.append(ThresholdModel(base,*(f[f'b{i}_{k}'].copy() for k in SPARSE_NAMES),zero))
        return cls(tuple(blocks),bias,shifts)


def load_prediction_model(path):
    with np.load(path,allow_pickle=False) as f:
        version = int(f['version'])
    return BlockLeafModel.load(path) if version == 6 else ThresholdModel.load(path)
