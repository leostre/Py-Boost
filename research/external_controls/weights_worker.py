"""Only explicitly licensed public revision; no auth discovery or dataset access."""
import hashlib
import json
from math import prod
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2])
    if not cfg['tabpfn_weights_authorized']: raise RuntimeError('user license gate missing')
    if (cfg['weights_repo']!='Prior-Labs/tabpfn_3_5'
            or cfg['weights_revision']!='06bf2ba35c80a92a3b9abb436b99cf49e7a0365e'
            or cfg['weights_filename']!='tabpfn-v3.5-20260909.safetensors'):
        raise RuntimeError('unreviewed checkpoint source')
    destination=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']
    paths={}
    for filename in ('LICENSE','config.json',cfg['weights_filename']):
        path=Path(hf_hub_download(repo_id=cfg['weights_repo'],filename=filename,
            revision=cfg['weights_revision'],token=False,local_dir=destination))
        paths[filename]=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)
    weight=paths[cfg['weights_filename']]
    if weight['sha256']!=cfg['weights_sha256'] or weight['bytes']!=cfg['weights_bytes']:
        raise RuntimeError('checkpoint LFS integrity mismatch')
    with safe_open(weight['path'],framework='pt',device='cpu') as f:
        shapes={name:f.get_slice(name).get_shape() for name in f.keys()}
        metadata=f.metadata()
    report=dict(status='ready',files=paths,revision=cfg['weights_revision'],tensor_shapes=shapes,
        tensor_parameters=sum(prod(shape) for shape in shapes.values()),safetensors_metadata=metadata,
        new_fits=0,inference_calls=0,datasets_loaded=False,implicit_token_disabled=True,
        license_authority='explicit human confirmation21Oct2026 for research use',
        environment_sha256=cfg['external_environment_sha256'])
    (output/'weights_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='ready',weight=weight,tensor_parameters=report['tensor_parameters'],
                         new_fits=0,inference_calls=0)),flush=True)


if __name__=='__main__': main()
