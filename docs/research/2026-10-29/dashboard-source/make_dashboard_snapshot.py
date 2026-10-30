"""Prepare reviewed result rows for the Data dashboard; no model execution."""
from pathlib import Path
import csv,json,hashlib,argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
    result=[]
    for row in csv.DictReader(p.open(encoding='utf-8',newline='')):
        out={}
        for k,v in row.items():
            if v=='':out[k]=None
            elif v in ('True','False'):out[k]=v=='True'
            elif v.startswith('['):out[k]=json.loads(v)
            else:
                try:out[k]=int(v)
                except ValueError:
                    try:out[k]=float(v)
                    except ValueError:out[k]=v
        if 'dataset' in out:out['cohort']='new' if out['dataset'] in ('enron','cal500','corel5k','delicious') else 'historical'
        result.append(out)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path);a=p.parse_args()
    queries={};files=[]
    definitions=[dict(label='Разница качества',definition='Для F1: 100×(A−B), п.п. Для NLL/Brier: B−A. Положительное значение означает преимущество A. Сопоставляются одинаковые case и потолок памяти; это не тест значимости.'),
        dict(label='Разброс',definition='Минимум и максимум трёх train/D-разбиений с общим тестом; не доверительный интервал.'),
        dict(label='B0',definition='Аналитический ориентир памяти до обучения. Фактическая стоимость и массивы GPU учитываются отдельно.'),
        dict(label='p95 и ускорение',definition='96 запросов на сочетание: три сеанса по 32. Ускорение одного исполнителя относительно другого — медиана трёх отношений сеансов; междатасетная медиана не является p95 общего потока.'),
        dict(label='Равные средние задачи',definition='Сначала среднее по доступным разбиениям каждого набора, затем равное среднее по наборам. Пропущенные модели не заменяются нулём.')]
    for file in sorted((a.evidence/'data').glob('*.csv')):
        name=file.stem;data=read_csv(file);files.append(dict(file='data/'+file.name,sha256=sha(file),rows=len(data)))
        queries[name]=dict(rows=data,source=dict(label=file.name,provider='Аудированная серия HASBoost, 29 октября 2026',
            files=['data/'+file.name],links=[dict(label='PR #24',url='https://github.com/leostre/Py-Boost/pull/24')],
            filters=['Только опубликованные агрегированные результаты; без признаков и разметки отдельных объектов.'],metricDefinitions=definitions,
            evidenceFlow=[dict(title='Исходный снимок',detail='HASBoost-A100-External d339851c2ab943a967dd3123ce64460fb85e8baf; модели и решения зафиксированы до теста.'),
                          dict(title='Контрольная сумма CSV',detail=sha(file)),dict(title='Преобразование',detail='CSV → числа, логические значения и null без изменения оценок; cohort=new только для Enron/CAL500/Corel5k/Delicious.')]))
    pairs=json.loads((a.evidence/'inference_pairs.json').read_text())['pairs']
    queries['executor_pairs']=dict(rows=pairs,source=dict(label='Парные сеансы исполнителей',files=['inference_pairs.json'],metricDefinitions=definitions,
        evidenceFlow=[dict(title='Парное отношение',detail='Одинаковые case, uid и batch; прежний / новый исполнитель. Медиана трёх отношений сеансов. Данные взяты из опубликованного inference_pairs.json.')]))
    assert len(queries['all_quality_points']['rows'])==4941 and len(queries['selected_quality']['rows'])==1008
    meta=dict(files=files,as_of='2026-10-29',new_datasets=['enron','cal500','corel5k','delicious'],
        limitations=['Три train/D-разбиения используют общий тест.', 'Восемь тестов уже использовались при разработке.', 'Окончательный выбор всех моделей выполнен по D до теста.',
                      'Ранг прямого базиса ограничен 64; глубины 3/5, до 200 деревьев.', 'Банк: первое разбиение каждой задачи; максимальная вместимость A100 не измерена.',
                      'Первые 120/130 индивидуальных замеров имели внешний контекст около 20 ГиБ при 0% загрузки до запроса; последние десять — без него.',
                      'CAL500: обычный контроль уступает постоянному решению «все метки положительные».'])
    result=dict(surface='dashboard',title='HASBoost · результаты экспериментов',status='reviewed',buildStatus='creating',generatedAt='2026-10-29T22:36:00+03:00',filters=[],queries=queries,research=meta)
    if a.metadata:
        metadata=json.loads(a.metadata.read_text(encoding='utf-8'))
        if 'queries' in metadata or 'research' in metadata:raise ValueError('Metadata must not replace reviewed evidence')
        result.update(metadata)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('Reviewed snapshot:',{k:len(v['rows']) for k,v in queries.items()})

if __name__=='__main__':main()
