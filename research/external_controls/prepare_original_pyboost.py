"""Extract byte-pinned baseline sources from an existing Git clone; no install/run."""
from pathlib import Path
import argparse,hashlib,json,subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);args=p.parse_args()
    target=Path(__file__).resolve().parent/'original_pyboost'
    pin=json.loads((target/'source.json').read_text(encoding='utf8'))
    payloads={}
    for name,expected in pin['files'].items():
        if not name.startswith('py_boost/') or '..' in Path(name).parts:raise ValueError('Invalid source path')
        payload=subprocess.check_output(['git','show',pin['commit']+':'+name],cwd=args.repository)
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Original blob mismatch: '+name)
        path=target/name
        if path.exists() and path.read_bytes()!=payload:raise FileExistsError('Will not overwrite changed source: '+str(path))
        payloads[path]=payload
    for path,payload in payloads.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    print(json.dumps(dict(status='original_sources_verified',files=len(payloads),commit=pin['commit'])))

if __name__=='__main__':main()
