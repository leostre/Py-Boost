"""Explicit immutable fingerprint of the external Python/CUDA environment."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def environment_snapshot():
    return {
        'python': platform.python_version(),
        'executable': sys.executable,
        'system': platform.system(),
        'gpu': subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        'packages': sorted((d.metadata['Name'].lower(), d.version)
                           for d in importlib.metadata.distributions()),
    }


def main():
    path = Path('/root/hasboost_compactness_20260928/environment.json')
    snapshot = environment_snapshot()
    serialized = json.dumps(snapshot, indent=2, sort_keys=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(serialized):
            raise RuntimeError('refusing to replace an existing environment contract')
    else:
        path.write_text(serialized, encoding='utf-8')
    print(serialized, flush=True)


if __name__ == '__main__':
    main()
