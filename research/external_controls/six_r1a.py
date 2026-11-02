"""Read already answered R1; recover summary without recomputing predictions."""
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.stats import spearmanr
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment

def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);env=check_native_environment(cfg);spec=cfg['six'];seal=locked(spec['R1_seal']['path'],spec['R1_seal']['sha256']);pairs=[]
    refs=spec['R1_tests'];assert len(refs)==36
    for ref in seal['cases']:
        case=locked(ref['path'],ref['sha256']);test=next(t for t in refs if t['case']==case['case']);result=locked(test['path'],test['sha256']);by={m['uid']:m for m in result['models']}
        for p in case['pairs']:
            if p['B']:p.update(test_delta_pp=100*(by[p['A']]['quality']['f1_macro']-by[p['B']]['quality']['f1_macro']),test_A=by[p['A']]['quality'],test_B=by[p['B']]['quality'])
            pairs.append(p)
    profiles=list(csv.DictReader((Path(__file__).parent/'six_train_profiles.csv').open()));summary=[];meta=[]
    for family in ('direct_AB','direct_ABL'):
        for ratio in (.25,.5,.7,1.):
            rows=[p for p in pairs if p['family']==family and p['ratio']==ratio and p['B']];names=sorted({p['dataset'] for p in rows})
            means=[float(np.mean([p['test_delta_pp'] for p in rows if p['dataset']==n])) for n in names]
            row=dict(family=family,ratio=ratio,pairs=len(rows),datasets=len(names),mean_dataset_delta_pp=float(np.mean(means)),positive_datasets=int(sum(v>0 for v in means)))
            summary.append(row);event('R1_result',**row)
            x=[]
            for name,gain in zip(names,means):
                pp=[p for p in profiles if p['dataset']==name]
                values={k:float(np.mean([float(p[k]) for p in pp])) for k in ('outputs','train_rows','features','label_density','top64_energy_fraction')}
                x.append([np.log2(values['outputs']),values['label_density'],values['top64_energy_fraction']])
                meta.append(dict(dataset=name,family=family,ratio=ratio,test_delta_pp=gain,**values))
            x=np.asarray(x);y=np.asarray(means);pred=[];baseline=[]
            for i in range(len(x)):
                mask=np.arange(len(x))!=i;scale=np.maximum(x[mask].std(0),1e-8);nearest=np.argmin(np.sum(((x[mask]-x[i])/scale)**2,axis=1))
                pred.append(float(y[mask][nearest]));baseline.append(float(np.median(y[mask])))
            row['exploratory_LODO_1NN']=dict(sign_accuracy=float(np.mean((np.array(pred)>0)==(y>0))),baseline_sign_accuracy=float(np.mean((np.array(baseline)>0)==(y>0))),
                MAE_pp=float(np.mean(np.abs(np.array(pred)-y))),baseline_MAE_pp=float(np.mean(np.abs(np.array(baseline)-y))),n=len(y))
            row['exploratory_spearman']={k:float(spearmanr(x[:,i],y).statistic) if np.std(x[:,i])>1e-9 else None for i,k in enumerate(('log2_outputs','label_density','top64_energy_fraction'))}
    result=dict(status='R1_complete',cases=seal['cases'],pairs=pairs,summary=summary,train_meta=meta,source_seal=spec['R1_seal'],source_tests=refs,new_trees=0,new_gpu_queries=0,environment=env,
        limitations=['R1 terminal status failed only at final NumPy-int summary serialization after all36 test receipts were saved; no scientific computation repeated',
                    'actual packed arrays plus8byte temperature;5percent byte tolerance, not quality threshold','12historically observed tasks; shared test across3partitions',
                    'meta correlations/LODO are exploratory from12 tasks; no universal applicability predictor established'])
    write(out/'report.json',result);event('R1A_complete',report=str(out/'report.json'),sha256=digest(out/'report.json'),matched=int(sum(p['B'] is not None for p in pairs)),total=len(pairs),new_trees=0,new_gpu_queries=0)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
