"""Read-only SSH export of SHA-pinned scientific JSON metadata, never secrets."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('report');parser.add_argument('sha256');parser.add_argument('destination');parser.add_argument('--include-final-arrays',action='store_true');args=parser.parse_args()
    dest=Path(args.destination).resolve();dest.mkdir(parents=True,exist_ok=True);archive=dest/('evidence-with-arrays.tar.gz' if args.include_final_arrays else 'metadata.tar.gz')
    script='''import sys,json,hashlib,tarfile
from pathlib import Path
p=Path(REPORT);expected=SHA
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(p)==expected
d=json.loads(p.read_text());items=[(p,'report.json')]
config=p.parents[2]/'external_config.json'
if config.is_file():items.append((config,'run_config.json'))
if 'ledger' in d:
 q=d['ledger'];assert digest(q['path'])==q['sha256'];items.append((Path(q['path']),'training_ledger.json'))
for i,r in enumerate(d.get('cases',[])):
 if not isinstance(r,dict) or 'path' not in r:continue
 assert digest(r['path'])==r['sha256'];items.append((Path(r['path']),r.get('case',str(i))+'/case.json'))
for key in ('seal','manifest','training_profiles'):
 if key in d and isinstance(d[key],dict) and 'path' in d[key]:
  ref=d[key];assert digest(ref['path'])==ref['sha256'];items.append((Path(ref['path']),key+'.json'))
for key in ('case_representations','matched_native'):
 for i,r in enumerate(d.get(key,[])):
  assert digest(r['path'])==r['sha256'];name=key+'/'+r.get('case',str(i))+'/'+r.get('family','model')+'/case.json'
  items.append((Path(r['path']),name))
if INCLUDE_ARRAYS:
 assert d['status']=='unified_final_complete'
 for cid,r in d['targets'].items():
  assert digest(r['path'])==r['sha256'];items.append((Path(r['path']),'targets/'+cid+'.npz'))
 for i,row in enumerate(d['rows']):
  r=row['raw'];assert digest(r['path'])==r['sha256'];items.append((Path(r['path']),f'raw/{i:04d}.npz'))
 assert sum(path.stat().st_size for path,_ in items)<=1024**3,'export exceeds registered 1GiB bundle bound'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path,name in items:
  assert path.is_file() and not path.is_symlink();tar.add(path,arcname=name,recursive=False)
'''.replace('REPORT',repr(args.report)).replace('SHA',repr(args.sha256)).replace('INCLUDE_ARRAYS',repr(args.include_final_arrays))
    with archive.open('wb') as target:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','hasboost-a100','python3 -'],
            input=script.encode(),stdout=target,stderr=subprocess.PIPE,check=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            target=(dest/'data'/member.name).resolve()
            if not target.is_relative_to(dest/'data') or not member.isfile():raise ValueError('invalid metadata archive member')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
    actual=hashlib.sha256((dest/'data/report.json').read_bytes()).hexdigest();assert actual==args.sha256
    receipt=dict(report_path=args.report,report_sha256=actual,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size)
    (dest/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
