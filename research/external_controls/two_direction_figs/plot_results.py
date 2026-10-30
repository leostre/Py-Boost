"""Vector figures from exported run CSVs only. No model fitting or demo data."""
from pathlib import Path
import argparse,csv,json
from collections import defaultdict
import numpy as np
import orx_figstyle as style
from orx_figstyle import PALETTE,WIDE,figure,figure_grid,panel_labels,save,use_style

FAMILIES={'ordinary':('обычный','black','o'),'sketch':('SketchBoost','cyan','s'),
          'posthoc':('после обучения','orange','D'),'direct_AB':('AB в обучении','purple','^'),
          'direct_ABL':('AB+L в обучении','red','v'),'controller':('регулятор','green','P'),'mixed24':('смешанный 12+12','blue','X')}
NAMES={'emotions':'Emotions','scene':'Scene','yeast':'Yeast','birds':'Birds','medical':'Medical','genbase':'GenBase',
       'bibtex':'Bibtex','mediamill':'MediaMill','enron':'Enron','cal500':'CAL500','corel5k':'Corel5k','delicious':'Delicious'}


def rows(path):
    with Path(path).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def finish(fig,path):
    problems=style._audit(fig)
    fatal=[p for p in problems if 'font' not in p.lower()]
    if fatal:raise RuntimeError('figure layout audit: '+str(fatal))
    # Internal visual QA only; PDF/SVG are the delivered scientific figures.
    fig.savefig(str(path)+'-preview.png',dpi=140)
    return save(fig,str(path))


def coverage(data,out):
    unique={r['dataset']:r for r in rows(data/'dataset_profiles.csv')};fig,ax=figure(width=WIDE,ratio=.64)
    for new,marker,color,label in [(False,'o','black','сохранённый тест'),(True,'s','orange','первый тест метода на новом наборе')]:
        rr=[r for name,r in unique.items() if (name in ('enron','cal500','corel5k','delicious'))==new]
        ax.scatter([int(r['rows']) for r in rr],[int(r['outputs']) for r in rr],marker=marker,s=30,
                   facecolors=PALETTE[color] if new else 'none',edgecolors=PALETTE[color],label=label)
    for name,r in unique.items():
        offset=(5,4) if name!='emotions' else (5,-10)
        ax.annotate(NAMES[name],(int(r['rows']),int(r['outputs'])),xytext=offset,textcoords='offset points',fontsize=7)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(350,28000);ax.set_ylim(4,1800)
    ax.set_xlabel('Объекты в фиксированной когорте, шт.');ax.set_ylabel('Метки, шт.')
    fig.legend(loc='outside lower center',ncol=2);finish(fig,out/'dataset-coverage')


def budget_curves(data,out):
    all_rows=rows(data/'all_quality_points.csv');selected=rows(data/'selected_quality.csv')
    for dataset in NAMES:
        fig,ax=figure(width=WIDE,ratio=.7);handles=[]
        for family,(label,color,marker) in FAMILIES.items():
            rr=[r for r in all_rows if r['dataset']==dataset and r['family']==family]
            if rr:ax.scatter([float(r['cost_ratio_B0']) for r in rr],[float(r['f1_macro']) for r in rr],s=5,alpha=.12,marker=marker,color=PALETTE[color],linewidths=0)
            chosen=[r for r in selected if r['dataset']==dataset and r['family']==family and r.get('uid')]
            xx=[];yy=[];xe=[];ye=[]
            for ratio in (.25,.5,.7,1.):
                pp=[r for r in chosen if float(r['ratio'])==ratio]
                if not pp:continue
                x=np.array([float(r['cost_ratio_B0']) for r in pp]);y=np.array([float(r['f1_macro']) for r in pp]);xm=x.mean();ym=y.mean()
                xx.append(xm);yy.append(ym);xe.append((xm-x.min(),x.max()-xm));ye.append((ym-y.min(),y.max()-ym))
            if xx:
                ax.errorbar(xx,yy,xerr=np.maximum(0,np.array(xe).T),yerr=np.maximum(0,np.array(ye).T),fmt=marker+'--',markersize=4,
                            linewidth=.9,elinewidth=.6,capsize=2,color=PALETTE[color],label=label)
        ax.set_xscale('log');ax.set_xlabel('Память модели / B0 (меньше лучше)');ax.set_ylabel('macro-F1 (больше лучше)')
        ax.set_ylim(bottom=0);fig.legend(loc='outside lower center',ncol=3)
        finish(fig,out/('budget-'+dataset))


def paired_quality(data,out):
    source=rows(data/'selected_quality.csv');fig,ax=figure(width=WIDE,ratio=.78)
    chosen=['posthoc','direct_AB','direct_ABL','controller','mixed24'];datasets=list(NAMES)
    for j,family in enumerate(chosen):
        label,color,marker=FAMILIES[family];xx=[];yy=[];errors=[]
        for i,name in enumerate(datasets):
            r=[r for r in source if r['dataset']==name and r['family']==family and r['ratio']=='0.7' and r.get('delta_ordinary_pp')!='']
            if not r:continue
            v=np.array([float(p['delta_ordinary_pp']) for p in r]);xx.append(v.mean());yy.append(i+(j-(len(chosen)-1)/2)*.13);errors.append((v.mean()-v.min(),v.max()-v.mean()))
        ax.errorbar(xx,yy,xerr=np.maximum(0,np.array(errors).T),fmt=marker,markersize=4,capsize=2,elinewidth=.7,color=PALETTE[color],label=label)
    ax.axvline(0,color='#7F7F7F',linewidth=.7);ax.set_yticks(np.arange(len(datasets)),[NAMES[d] for d in datasets]);ax.invert_yaxis()
    ax.set_xlabel('Разница с обычным бустингом при 0.7 B0, п.п.');ax.set_ylabel('Набор данных')
    fig.legend(loc='outside lower center',ncol=3);finish(fig,out/'quality-at-budget70')


def serving(data,out):
    source=rows(data/'inference_summary.csv');fig,axes=figure_grid(1,2,width=WIDE,ratio=.5,sharey=True)
    roles=[('full200','полная','black','o'),('direct_ABL200','AB+L','red','v'),('ordinary70','небольшая обычная','cyan','s')]
    for ax,metric in zip(np.ravel(axes),('gpu_p50_ms','request_p95_ms')):
        for role,label,color,marker in roles:
            points=[]
            for r in source:
                if r['backend']!='packed' or int(r['batch'])!=1024 or role not in json.loads(r['roles']):continue
                old=next(p for p in source if p['case']==r['case'] and p['uid']==r['uid'] and p['batch']==r['batch'] and p['backend']=='legacy')
                field='gpu_session_p50_ms' if metric=='gpu_p50_ms' else 'request_session_p95_ms'
                ratios=np.asarray(json.loads(old[field]))/np.asarray(json.loads(r[field]));middle=float(np.median(ratios))
                points.append((int(r['outputs']),middle,middle-float(ratios.min()),float(ratios.max())-middle))
            ax.errorbar([p[0] for p in points],[p[1] for p in points],yerr=np.maximum(0,np.array([[p[2],p[3]] for p in points]).T),
                        fmt=marker,markersize=4,capsize=2,elinewidth=.6,color=PALETTE[color],label=label)
        ax.set_xscale('log');ax.set_yscale('log');ax.axhline(1,color='#7F7F7F',linewidth=.7);ax.set_xlabel('Число выходов, шт.')
    axes[0].set_ylabel('Ускорение нового исполнителя, раз');axes[1].set_ylabel('Ускорение нового исполнителя, раз')
    panel_labels(axes,['(a) GPU, p50','(b) весь запрос, p95']);handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    finish(fig,out/'executor-speedup')


def main(data,out):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True);use_style()
    if (data/'dataset_profiles.csv').exists():coverage(data,out)
    if (data/'all_quality_points.csv').exists():budget_curves(data,out);paired_quality(data,out)
    if (data/'inference_summary.csv').exists():serving(data,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.data,a.output)
