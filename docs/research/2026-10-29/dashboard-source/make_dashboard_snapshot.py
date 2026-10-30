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

def explanatory_queries(catalog_path):
    catalog=json.loads(catalog_path.read_text(encoding='utf8'))
    labels={'ordinary':'Обычный бустинг','direct_AB':'AB · прямое обучение','direct_ABL':'AB + L · прямое обучение','posthoc':'Сжатие после обучения','controller':'Регулятор бюджета','sketch':'Sketch','mixed24':'Смешанный отбор'}
    assert {m['id'] for m in catalog['methods']}==set(labels)
    paths=sorted({f for m in catalog['methods']+catalog['executors'] for f in m['sources']}|{'research/external_controls/wide_probe.py'})
    native_paths={f:e['source_commit'] for e in catalog['executors'] if e.get('source_commit') for f in e['sources']}
    links=[dict(label=f,url=f"https://github.com/leostre/Py-Boost/blob/{native_paths.get(f,catalog['source_commit'])}/{f}") for f in paths]
    source=dict(label='Пояснения, сверенные с реализацией',provider='Код опубликованного исследовательского снимка',links=links,
        filters=['Концептуальные схемы фактически проверенных вариантов; не новые результаты и не обещание выигрыша.'],
        metricDefinitions=[dict(label='Схемы',definition='Количество ячеек и узлов на рисунках условно; численные замеры находятся в отдельных таблицах.'),
            dict(label='Два сравнения',definition='Исполнители: время legacy той же модели / время packed. Модели: время выбранного ориентира на packed / время измеряемой модели на packed.')],
        evidenceFlow=[dict(title='Снимок кода',detail=catalog['source_commit']),
            dict(title='Сверка',detail='broad_training/core: обучение и отбор; direct_leaf_gpu: AB/L; unified_serving_gpu, packed_serving_gpu, wide_probe: исполнение и границы замера.')])
    methods=[dict(id=m['id'],title=labels[m['id']],category=m['category'],summary=m['summary'],storage=m['storage'],scope=m['scope'],difference=m['difference'],source_files='; '.join(m['sources'])) for m in catalog['methods']]
    executors=[dict(id=e['id'],title=e['title'],mechanism=e['mechanism'],source_files='; '.join(e['sources'])) for e in catalog['executors']]
    return dict(method_mechanisms=dict(rows=methods,source=source),executor_mechanisms=dict(rows=executors,source=source))

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
        if name=='native_inference_summary':
            queries[name]['source'].update(provider='Дополнительная парная серия A100, 30 октября 2026',
                files=['data/native_inference_summary.csv','native_inference_report.json'],
                filters=['12 прежних наборов; первое сохранённое разбиение каждого; батчи 1/32/256/1024; без обучения и новой оценки на тесте.',
                         'Исходный master b2842af: арифметика float32; legacy/packed: float64 суммы. Проверка вероятностей и меток — на D и входе с NaN.',
                         'Нового замера памяти нет. Δ macro-F1 взята из прежнего финального отчёта.'],
                metricDefinitions=[dict(label='Общий ориентир',definition='Та же полная модель ordinary/d3/t200 на неизменённых Ensemble.predict и Tree.predict. Для AB/ABL ориентир остаётся полной моделью; это совмещённый эффект представления и исполнения.'),
                    dict(label='Ускорение',definition='Медиана трёх отношений p95(native)/p95(backend) или p50 GPU(native)/p50 GPU(backend); затем равная медиана по выбранным наборам. Абсолютные времена — по 96 запросам вместе.'),
                    dict(label='GPU-время',definition='CUDA elapsed при входе на GPU; включая паузы отправки ядер, исходные выделения и синхронизацию. Полный запрос дополнительно включает H2D/D2H вероятностей и меток.')],
                evidenceFlow=[dict(title='Исходный код',detail='leostre/Py-Boost master b2842af472e21c7332d9487d8c83e53cb48d1486; исходные файлы проверены по SHA-256.'),
                    dict(title='Парные измерения',detail='orx run 8d178a2d-3f3a-4ac1-bdbe-dda44b5020dc; свободная A100; 3×32 запроса; 336 численных проверок; без переобучения.'),
                    dict(title='Контрольная сумма CSV',detail=sha(file))])
    pairs=json.loads((a.evidence/'inference_pairs.json').read_text())['pairs']
    queries['executor_pairs']=dict(rows=pairs,source=dict(label='Парные сеансы исполнителей',files=['inference_pairs.json'],metricDefinitions=definitions,
        evidenceFlow=[dict(title='Парное отношение',detail='Одинаковые case, uid и batch; прежний / новый исполнитель. Медиана трёх отношений сеансов. Данные взяты из опубликованного inference_pairs.json.')]))
    catalog=Path(__file__).parent/'content/dashboard/mechanisms.json'
    if catalog.exists():queries.update(explanatory_queries(catalog))
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
