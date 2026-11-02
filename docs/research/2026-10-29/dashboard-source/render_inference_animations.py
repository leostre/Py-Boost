"""Render educational inference stages to SVG, GIF and dashboard records."""
from pathlib import Path
import argparse,base64,io,json,textwrap,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch
import numpy as np
from PIL import Image
from inference_animation_model import ENGINES,FEATURES,THRESHOLDS,records,plan

plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'hasboost-journey-v1'})
W,H=1320,760
COLORS={'native':'#627f9c','legacy':'#aa741f','packed':'#008b96'}
INK='#19354c';MUTED='#647c8e';BORDER='#d7e1e8';A_COLOR='#008b96';B_COLOR='#b47a22'

def render(model,record,values,fraction=0.):
    fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor='white')
    ax=fig.add_axes((0,0,1,1));ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
    color=COLORS[record['engine']]
    def text(x,y,s,size=11,c=INK,ha='left',weight='normal'):
        ax.text(x,y,s,fontsize=size,color=c,ha=ha,va='center',fontweight=weight)
    def box(x,y,w,h,edge=BORDER,fill='#f7fafc',lw=1.2):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={min(10,w/4,h/4)}',edgecolor=edge,facecolor=fill,linewidth=lw))
    def arrow(x1,y1,x2,y2,active=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=16,color=color if active else '#aebfc9',linewidth=2 if active else 1.2))
    def dot(x,y,c,r=5):ax.add_patch(Circle((x,y),r,color=c,zorder=9))
    text(36,34,ENGINES[record['engine']]+' · полный путь батча',20,weight='bold')
    text(36,65,f'Учебный пример · 2 объекта · C = {model["outputs"]} · показаны первые 3 выхода · '+('AB + L' if record['compact'] else 'полные листья W'),11,c=MUTED)
    phases=[('cpu_in','Батч на CPU'),('upload','CPU → GPU'),('gpu','Вычисления на GPU'),('download','GPU → CPU'),('cpu_out','Готовый ответ')]
    current=[p[0] for p in phases].index(record['phase'])
    for i,(key,label) in enumerate(phases):
        x=36+254*i;selected=i==current
        box(x,92,232,42,edge=color if selected else BORDER,fill='#edf7f8' if selected else '#f6f8fa',lw=2 if selected else 1)
        text(x+116,113,label,11,c=color if selected else MUTED,ha='center',weight='bold' if selected else 'normal')
        if i<4:arrow(x+235,113,x+251,113)
    box(36,178,240,255,edge=color if current==0 else BORDER)
    text(55,205,'Вход X · CPU',13,weight='bold')
    text(55,232,'Один батч, те же два объекта',9,c=MUTED)
    for j,name in enumerate(('x₁','x₂')):text(150+j*70,268,name,11,ha='center')
    for i,name in enumerate(('A','B')):
        text(65,309+i*48,name,12,c=(A_COLOR,B_COLOR)[i],weight='bold')
        for j in range(2):text(150+j*70,309+i*48,f'{model["x"][i,j]:+.1f}',12,ha='center')
    text(55,400,'Параметры модели уже на GPU',9,c=MUTED)
    arrow(281,302,326,302,current==1)
    box(334,158,616,435,edge=color if current==2 else BORDER,fill='#fbfdfe',lw=2 if current==2 else 1.2)
    kernel=record['kernel'] or ('Результат готов к передаче' if current>2 else 'GPU ожидает вход')
    text(354,184,kernel,12,c=color,weight='bold')
    text(354,210,'Листья выбираются отдельно для объектов A и B',9,c=MUTED)
    motion=[]
    for t in range(4):
        x=408+t*146;y=275;active=t in record['active'];done=t in record['completed']
        tree_color=color if active else '#849bab' if done else '#cbd6df'
        text(x,244,f'Дерево {t+1}',10,ha='center',c=tree_color,weight='bold' if active else 'normal')
        for sign in (-1,1):ax.plot([x,x+30*sign],[y,y+55],color=tree_color,lw=1.8,zorder=1)
        dot(x,y,tree_color,4)
        for sign in (-1,1):box(x+30*sign-7,y+51,14,10,edge=tree_color,fill=tree_color,lw=.5)
        if active:
            for i,c in enumerate((A_COLOR,B_COLOR)):
                sign=-1 if values['leaves'][i,t]==0 else 1
                offset=-3 if i==0 else 3
                ax.plot([x+offset,x+30*sign+offset],[y,y+55],color=c,lw=2.2,alpha=.85)
                if record['id'].startswith('walk'):
                    dot(x+30*sign*fraction+offset,y+55*fraction,c,4)
                    motion.append(((x+offset,y),(x+30*sign+offset,y+55),c))
        text(x,355,f'x{FEATURES[t]+1} > {THRESHOLDS[t]:g}',9,ha='center',c=MUTED)
        if active or done:
            route='A:'+str(values['leaves'][0,t])+'  B:'+str(values['leaves'][1,t])
            text(x,377,route,9,ha='center',c=tree_color)
    box(353,400,578,173,fill='#f0f6fa')
    buffer=record['buffer'];compact=record['compact']
    if buffer=='none':
        text(374,432,'Вычисление ещё не началось',12,c=MUTED)
        text(374,465,'Сначала батч должен попасть на GPU.',10,c=MUTED)
    elif buffer=='J':
        text(374,428,'Индексы достигнутых листьев J',12,weight='bold')
        text(374,461,'Подсвеченные пути показывают A и B.',10,c=MUTED)
        text(374,493,'Сумма не изменяется до сбора значений.',10,c=MUTED)
    else:
        if buffer=='Z':
            numbers=np.concatenate((values['coordinates'],values['residual'][:,:3]),axis=1)
            labels=('Z₁','Z₂','E₁','E₂','E₃')
            location='локально, внутри одного ядра' if record['engine']=='packed' or not record['wide'] else 'временные массивы GPU'
            heading='Координаты Z и поправки E · '+location
        else:
            numbers=values['probability'][:,:3] if buffer=='p' else values['block'][:,:3] if buffer=='blockS' else values['raw'][:,:3]
            labels=('p₁','p₂','p₃') if buffer=='p' else ('S₁','S₂','S₃')
            heading='Вероятности на GPU' if buffer=='p' else 'Вектор текущего блока' if buffer=='blockS' else 'Накопленный прогноз S · bias учтён один раз'
        text(374,425,heading,10,weight='bold')
        for j,label in enumerate(labels):text(465+j*(440/len(labels)),458,label,10,ha='center',c=MUTED)
        for i,name in enumerate(('A','B')):
            text(379,491+i*36,name,11,c=(A_COLOR,B_COLOR)[i],weight='bold')
            for j,value in enumerate(numbers[i]):text(465+j*(440/len(labels)),491+i*36,f'{value:.3f}' if buffer=='p' else f'{value:+.2f}',11,ha='center')
        if buffer=='p':text(740,552,'порог p ≥ 0,5',9,c=MUTED)
    arrow(956,302,994,302,current==3)
    box(1000,178,284,340,edge=color if record['cpu_ready'] else BORDER)
    text(1020,205,'Ответ · CPU',13,weight='bold')
    text(1020,233,'Первые 3 вероятности',10,c=MUTED)
    for i,name in enumerate(('A','B')):
        text(1020,275+i*55,name,11,c=(A_COLOR,B_COLOR)[i],weight='bold')
        for j in range(3):text(1078+j*78,275+i*55,f'{values["probability"][i,j]:.3f}' if record['cpu_ready'] else '—',11,ha='center')
    text(1020,385,'Метки при пороге 0,5',10,c=MUTED)
    for i,name in enumerate(('A','B')):
        label='  '.join(str(int(v)) for v in values['labels'][i,:3]) if record['cpu_ready'] else '—  —  —'
        text(1020,420+i*36,name+':  '+label,12,c=INK if record['cpu_ready'] else MUTED)
    if current in (1,3):
        a,b=((282,302),(326,302)) if current==1 else ((956,302),(994,302))
        position=np.array(a)+(np.array(b)-a)*fraction
        box(position[0]-5,position[1]-7,10,14,edge=color,fill=color,lw=.5)
        motion.append((a,b,color))
    text(38,630,f'Шаг {record["index"]+1} · '+record['title'],15,weight='bold')
    for i,line in enumerate(textwrap.wrap(record['detail'],width=122)):
        text(38,660+i*23,line,11,c=MUTED)
    text(38,731,'Учебные числа и условный темп. Длительность анимации не является замером ускорения.',10,c=MUTED)
    return fig,motion

def svg_bytes(fig):
    stream=io.BytesIO();fig.savefig(stream,format='svg',metadata={'Date':None})
    return stream.getvalue().replace(b"font-family: 'DejaVu Sans'",b"font-family: 'DejaVu Sans', Arial, sans-serif")

def animated_frame(svg,motion):
    root=ET.fromstring(svg);ns='http://www.w3.org/2000/svg'
    for a,b,color in motion:
        circle=ET.SubElement(root,'{'+ns+'}circle',r='3.6',fill=color)
        path=f'M {a[0]*.72:g} {a[1]*.72:g} L {b[0]*.72:g} {b[1]*.72:g}'
        ET.SubElement(circle,'{'+ns+'}animateMotion',path=path,dur='1.2s',repeatCount='indefinite')
    return ET.tostring(root,encoding='unicode')

def standalone(rows,path):
    n=len(rows);duration=1.2*n;end=100/n
    chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%" viewBox="0 0 1320 760" preserveAspectRatio="xMidYMid meet"><title>{ENGINES[rows[0]["engine"]]}: учебный путь батча</title><style>@keyframes step{{0%,{end-.01:g}%{{opacity:1}}{end:g}%,100%{{opacity:0}}}}.frame{{opacity:0;animation:step {duration:g}s linear infinite}}@media(prefers-reduced-motion:reduce){{.frame{{animation:none}}.frame0{{opacity:1}}}}</style>']
    for i,row in enumerate(rows):
        uri='data:image/svg+xml;base64,'+base64.b64encode(row['animated_svg'].encode('utf8')).decode('ascii')
        chunks.append(f'<g class="frame frame{i}" style="animation-delay:{1.2*i:g}s"><image width="1320" height="760" xlink:href="{uri}"/></g>')
    chunks.append('</svg>');path.write_text(''.join(chunks),encoding='utf8')

def gif(model,rows,path):
    images=[];durations=[];stages=plan(rows[0]['engine'],rows[0]['compact'],rows[0]['wide'])
    for row,stage in zip(rows,stages):
        from inference_animation_model import state
        values=state(model,stage)
        for fraction in (.1,.5,.9):
            fig,_=render(model,row,values,fraction);fig.canvas.draw()
            image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            images.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=128));durations.append(400)
            plt.close(fig)
    durations[-1]=2000
    images[0].save(path,save_all=True,append_images=images[1:],duration=durations,loop=0,disposal=2,optimize=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gif',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);all_rows=[];files=[]
    exports={'native/full/narrow':'native','legacy/full/narrow':'legacy','packed/full/narrow':'packed','packed/compact/narrow':'packed-abl'}
    for engine in ENGINES:
        for compact in ((False,) if engine=='native' else (False,True)):
            for wide in (False,True):
                model,data=records(engine,compact,wide);rows=[]
                for record,values in data:
                    fig,motion=render(model,record,values)
                    raw=svg_bytes(fig);plt.close(fig)
                    row=dict(record,svg=raw.decode('utf8'),animated_svg=animated_frame(raw,motion))
                    rows.append(row);all_rows.append(row)
                variant=rows[0]['variant']
                if variant in exports:
                    name=exports[variant];standalone(rows,args.output/(name+'.svg'))
                    poster_index=next((i for i,r in enumerate(rows) if r['id'].startswith('decode')),4)
                    fig,_=render(model,*data[poster_index]);fig.savefig(args.output/(name+'-poster.png'));plt.close(fig)
                    if args.gif:gif(model,rows,args.output/(name+'.gif'))
                    files.append(dict(name=name,variant=variant,stages=len(rows)))
                print(variant,len(rows),flush=True)
    query=dict(rows=all_rows,source=dict(label='Учебная анимация инференса',provider='Авторский синтетический пример; не измерения HASBoost',
        filters=['Два объекта; C = 3 или 144, показаны первые 3 координаты. Числа не относятся к датасетам исследования.'],
        metricDefinitions=[dict(label='Время',definition='Время шага условно. По длительности GIF или анимации нельзя судить об ускорении.'),dict(label='Точная факторизация примера',definition='W=A×B+L задано точно только для учебного ансамбля. Сжатие исследовательских моделей обычно приближённое.')],
        links=[dict(label='Реализация и границы',url='https://github.com/leostre/Py-Boost/blob/hasboost/20261030/six-direction-study/docs/research/methodology/animations/README.md')],
        evidenceFlow=[dict(title='Числа примера',detail='inference_animation_model.py; конечные оценки A=(0.6,0.05,1), B=(-0.65,1,-1.3) при C=3.'),dict(title='Проверки',detail='Независимая сумма полных листьев, разложение, маршруты, границы CPU/GPU и группировка запусков.')]))
    (args.output/'frames.json').write_text(json.dumps(query,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    (args.output/'exports.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Rendered',len(all_rows),'stages; exported',len(files),'animations',flush=True)

if __name__=='__main__':main()
