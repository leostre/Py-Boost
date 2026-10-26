"""Export the existing benchmark representations; never train a model here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_datasets import load_benchmark_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('a100_config.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in config['datasets']:
        data = load_benchmark_dataset(
            name, max_samples=config['max_samples'][name],
            dataset_seed=config['dataset_seed'], cache_dir=args.cache_dir,
        )
        path = args.output_dir / f'{name}.npz'
        if path.exists():
            raise FileExistsError(f'refusing to overwrite prepared data: {path}')
        np.savez_compressed(path, features=data.features, target=data.target,
                            source_indices=data.sample_indices)
        manifest[name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'task': data.spec.task.value, 'source': data.spec.source,
            'representation': data.spec.representation,
            'features_shape': list(data.features.shape),
            'target_shape': list(data.target.shape),
            'source_shape': list(data.source_shape),
            'dataset_seed': config['dataset_seed'],
            'policy': 'fixed public benchmark subset; new 60/20/20 splits',
        }
        print(json.dumps({'prepared': name, **manifest[name]}, ensure_ascii=False), flush=True)
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
