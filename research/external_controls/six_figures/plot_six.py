"""Two evidence-only vector comparisons; no synthetic points or smoothing."""
from pathlib import Path
from collections import defaultdict
import csv,sys
import numpy as np
import orx_figstyle as style
from orx_figstyle import use_style,figure,figure_grid,PALETTE,save,panel_labels,WIDE

NAMES={'emotions':'Emotions','scene':'Scene','yeast':'Yeast','birds':'Birds','medical':'Medical','genbase':'GenBase',
       'bibtex':'Bibtex','mediamill':'MediaMill','enron':'Enron','cal500':'CAL500','corel5k':'Corel5k','delicious':'Delicious'}
def read(path):
    with path.open(encoding='utf8') as f:return list(csv.DictReader(f))
def finish(fig,path):
    defects=[p for p in style._audit(fig) if 'font' not in p.lower()]
    if defects:raise RuntimeError(str(defects))
    fig.savefig(str(path)+'-preview.png',dpi=150)
    save(fig,str(path))


def run(data,out):
    use_style();q=read(data/'quality_points.csv');groups=defaultdict(list)
    for r in q:
        if r['stage'] in ('R3','R3C') and r['uid'] in ('remove/t50','refit/t50','refit_ABL/t50'):groups[r['dataset'],r['uid']].append(float(r['delta_pp']))
    order=sorted({k[0] for k in groups},key=lambda n:np.mean(groups[n,'refit/t50']))
    fig,ax=figure(width=WIDE,ratio=.68)
    for j,(uid,label,color,marker) in enumerate([('remove/t50','Удаление без обновления','black','o'),('refit/t50','Обновление полных листьев','blue','s'),('refit_ABL/t50','Обновление + AB + L','orange','D')]):
        values=[groups[n,uid] for n in order];mean=np.array([np.mean(v) for v in values]);lo=mean-np.array([min(v) for v in values]);hi=np.array([max(v) for v in values])-mean
        ax.errorbar(mean,np.arange(len(order))+.19*(j-1),xerr=np.array([lo,hi]),fmt=marker,ms=3.7,lw=.8,c=PALETTE[color],capsize=1.8,label=label)
    ax.axvline(0,color=PALETTE['black'],lw=.65,ls='--');ax.set_yticks(np.arange(len(order)),[NAMES[n] for n in order]);ax.set_xlabel('Δ macro-F1 к обычным 50 деревьям, п.п.');ax.set_ylabel('Набор данных')
    fig.legend(loc='outside upper center',ncol=3,frameon=False);finish(fig,out/'structure_refit')
    rows=read(data/'quantization.csv');quant={}
    for r in rows:
        if r['stage']=='R5' and r['codec']=='int8' and r['policy']=='fixed' and r['source_uid'] in ('ordinary/d3/t200','direct_ABL/t200'):
            quant[r['dataset'],r['source_uid']]=r
    order=[n for n in NAMES if (n,'direct_ABL/t200') in quant]
    fig,axes=figure_grid(1,2,width=WIDE,ratio=.67,sharey=True);panel_labels(axes)
    for j,(uid,label,color,marker) in enumerate([('ordinary/d3/t200','Полная модель','black','o'),('direct_ABL/t200','AB + L','blue','s')]):
        memory=[100*float(quant[n,uid]['memory_ratio']) for n in order];delta=[float(quant[n,uid]['delta_pp']) for n in order];y=np.arange(len(order))+.14*(j-.5)
        axes[0].plot(memory,y,marker,ms=3.7,c=PALETTE[color],label=label);axes[1].plot(delta,y,marker,ms=3.7,c=PALETTE[color])
    axes[0].set_xlim(0,100);axes[0].set_yticks(np.arange(len(order)),[NAMES[n] for n in order]);axes[0].set_xlabel('Память INT8 / FP32, %');axes[0].set_ylabel('Набор данных')
    axes[1].axvline(0,color=PALETTE['black'],lw=.65,ls='--');axes[1].set_xlabel('Δ macro-F1, п.п.')
    fig.legend(loc='outside upper center',ncol=2,frameon=False);finish(fig,out/'quantization_memory_quality')

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
