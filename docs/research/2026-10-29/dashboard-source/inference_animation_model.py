"""An authored, exactly checkable teaching ensemble, never benchmark data."""
from dataclasses import dataclass, asdict
import numpy as np

ENGINES={'native':'Исходный Py-Boost','legacy':'Legacy','packed':'Packed'}
X=np.array([[-1.,.4],[1.2,-.7]])
COORDINATES=np.array([[[.2,-.1],[-.1,.3]],[[.1,.2],[.3,.1]],[[.25,.05],[-.15,.25]],[[-.2,.15],[.05,-.1]]])
FEATURES=(0,1,0,1);THRESHOLDS=(0.,0.,1.,.2)

def example(outputs=3):
    if outputs not in (3,144):raise ValueError('teaching example supports C=3/144')
    basis=np.tile(np.array([[1.,0.,1.],[0.,1.,-1.]]),(1,outputs//3))
    residual=np.zeros((4,2,outputs));residual[1,1,2]=.2;residual[3,0,0]=-.1
    bias=np.tile([-.2,.1,-.05],outputs//3)
    full=COORDINATES@basis+residual
    leaves=np.column_stack([X[:,f]>threshold for f,threshold in zip(FEATURES,THRESHOLDS)]).astype(int)
    return dict(x=X.copy(),coordinates=COORDINATES.copy(),basis=basis,residual=residual,bias=bias,full=full,leaves=leaves,outputs=outputs)

@dataclass(frozen=True)
class Stage:
    id:str
    phase:str
    title:str
    detail:str
    active:tuple=()
    completed:tuple=()
    kernel:str=''
    buffer:str='none'
    probabilities:bool=False
    cpu_ready:bool=False

def plan(engine,compact=False,wide=False):
    if engine not in ENGINES:raise ValueError('unknown engine')
    if engine=='native':compact=False
    result=[]
    def add(id,phase,title,detail,**kw):result.append(Stage(id,phase,title,detail,**kw))
    add('receive','cpu_in','Принимаем батч','Два объекта A и B поступили на CPU. Параметры готовой модели уже находятся на GPU.')
    add('prepare','cpu_in','Готовим запрос','Проверяем форму и приводим признаки к FP32. Подготовка модели и компиляция сделаны заранее.')
    add('upload','upload','Передаём признаки CPU → GPU','На GPU копируется один и тот же батч; коэффициенты модели заново не загружаются.')
    if engine=='native':
        for t in range(4):
            done=tuple(range(t))
            add(f'walk{t}','gpu',f'Дерево {t+1}: находим листья','Отдельный запуск обходит дерево для объектов батча и записывает индексы листьев J.',active=(t,),completed=done,kernel=f'Ядро {2*t+1}: обход',buffer='J')
            add(f'gather{t}','gpu',f'Дерево {t+1}: прибавляем значения','Следующий запуск читает полные векторы W[J] и прибавляет их к накопленному прогнозу S.',active=(t,),completed=tuple(range(t+1)),kernel=f'Ядро {2*t+2}: сбор',buffer='S')
    elif engine=='legacy':
        for block,trees in enumerate(((0,1),(2,3))):
            done=tuple(range(2*block));final=tuple(range(2*block+2))
            group=f'Ядро блока {block+1}'
            add(f'walk{block}','gpu',f'Блок {block+1}: обходим два дерева','Вместе обрабатываются деревья одного блока. Для A и B находятся свои достигнутые листья.',active=trees,completed=done,kernel=group,buffer='J')
            add(f'gather{block}','gpu',f'Блок {block+1}: собираем значения',
                'Координаты Z и поправки E записываются во временные GPU-массивы.' if wide and compact else ('Координаты и поправки суммируются локально внутри того же ядра.' if compact else 'Собираем полные значения достигнутых листьев.'),
                active=trees,completed=done,kernel=group,buffer='Z' if compact else 'blockS')
            if wide:
                add(f'decode{block}','gpu',f'Блок {block+1}: получаем полный вектор',
                    'Отдельное умножение Z × B и добавление E восстанавливают выходы этого блока.' if compact else 'Полный вектор уже собран: малорангового декодирования здесь нет.',
                    active=trees,completed=done,kernel='Отдельные операции блока',buffer='blockS')
            add(f'add{block}','gpu',f'Блок {block+1}: объединяем прогноз',
                'Результат блока прибавляется к общему накопителю S; начальный прогноз учтён один раз.',active=trees,completed=final,kernel='Суммирование блоков',buffer='S')
    else:
        group='Одно CUDA-ядро: все блоки'
        for block,trees in enumerate(((0,1),(2,3))):
            done=tuple(range(2*block));final=tuple(range(2*block+2))
            add(f'walk{block}','gpu',f'Блок {block+1}: обходим деревья внутри ядра','Группы потоков находят листья. Это этап внутри одного запуска для всех блоков, а не новый CUDA-запуск.',active=trees,completed=done,kernel=group,buffer='J')
            add(f'gather{block}','gpu',f'Блок {block+1}: накапливаем локальные суммы','Суммы находятся в регистрах и общей памяти CUDA-блока. Глобальный массив «батч × ранг» не создаётся.',active=trees,completed=done,kernel=group,buffer='Z' if compact else 'blockS')
            add(f'decode{block}','gpu',f'Блок {block+1}: применяем базис и поправки' if compact else f'Блок {block+1}: объединяем значения',
                'Применяем базис блока к сумме координат и добавляем поправки. Затем в том же ядре переходим к следующему блоку.' if compact else 'Локальные суммы прибавляются к общему прогнозу внутри того же ядра; начальный прогноз учтён один раз.',
                active=trees,completed=final,kernel=group,buffer='S')
    add('post','gpu','Вычисляем вероятности и метки',
        'Сигмоида и сравнение с порогом выполняются внутри того же packed-ядра.' if engine=='packed' else 'После суммирования отдельные операции вычисляют сигмоиду и метки; они могут включать несколько CUDA-запусков.',
        completed=(0,1,2,3),kernel='Одно CUDA-ядро: все блоки' if engine=='packed' else 'Обработка выхода отдельно',buffer='p',probabilities=True)
    add('download','download','Возвращаем результат GPU → CPU','На CPU копируются вероятности и метки. Передача входит во время полного запроса.',completed=(0,1,2,3),buffer='p',probabilities=True,cpu_ready=True)
    add('return','cpu_out','Предсказание готово','Все три пути вычисляют один учебный ансамбль. Темп анимации не показывает скорость из бенчмарка.',completed=(0,1,2,3),buffer='p',probabilities=True,cpu_ready=True)
    return result

def state(model,stage):
    def sum_rows(values,trees):
        return sum((values[t,model['leaves'][:,t]] for t in trees),np.zeros((2,values.shape[-1])))
    raw=np.tile(model['bias'],(2,1))+sum_rows(model['full'],stage.completed)
    active=stage.active or stage.completed
    coordinates=sum_rows(model['coordinates'],active)
    residual=sum_rows(model['residual'],active)
    block=sum_rows(model['full'],active)
    all_raw=np.tile(model['bias'],(2,1))+sum_rows(model['full'],range(4))
    probability=1/(1+np.exp(-all_raw))
    return dict(raw=raw,coordinates=coordinates,residual=residual,block=block,probability=probability,labels=all_raw>=0,leaves=model['leaves'])

def records(engine,compact,wide):
    model=example(144 if wide else 3)
    if engine=='native':compact=False
    key=f'{engine}/{"compact" if compact else "full"}/{"wide" if wide else "narrow"}'
    return model,[(dict(variant=key,engine=engine,compact=compact,wide=wide,index=i,outputs=model['outputs'],**asdict(stage)),state(model,stage)) for i,stage in enumerate(plan(engine,compact,wide))]
