"""Evidence-only rollup. All choices were frozen before these test comparisons."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys,shutil
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def describe(values):
    a=np.asarray(values,float)
    return dict(p50_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),mean_ms=float(a.mean()),requests=len(a))


def quality_tables(cfg,out):
    spec=cfg['broad_analysis'];final=locked(spec['final']['path'],spec['final']['sha256']);training=locked(spec['training']['path'],spec['training']['sha256'])
    assert final['status']=='broad_final_complete' and training['status']=='broad_training_complete' and final['source_training']==spec['training']
    manifest=locked(cfg['broad']['manifest']['path'],cfg['broad']['manifest']['sha256']);all_rows=[];profiles=[];lookup={};source_cases={};constants=[]
    for ref in final['cases']:
        case=locked(ref['path'],ref['sha256']);cid=case['case'];source_cases[cid]=case
        for m in case['models']:lookup[cid,m['uid']]=m
        anchor=next(m for m in case['models'] if m['uid']=='ordinary/d3/t200');cc=case['record']['case'];dd=case['record']['dataset'];geometry=case['train_profile']
        for name,q in case['diagnostic_baselines'].items():constants.append(dict(case=cid,dataset=cc['dataset'],seed=cc['seed'],baseline=name,
            f1_macro=q['f1_macro'],f1_weighted=q['f1_weighted'],loss=q['loss'],full200_f1=anchor['quality']['f1_macro'],
            full200_loss=anchor['quality']['loss'],delta_full_pp=100*(anchor['quality']['f1_macro']-q['f1_macro'])))
        profiles.append(dict(dataset=cc['dataset'],seed=cc['seed'],rows=dd['rows'],features=dd['features'],outputs=cc['outputs'],
             train_rows=cc['sizes']['train'],D_rows=sum(cc['sizes'][k] for k in ('calibration','control','selection')),test_rows=cc['sizes']['test'],
             label_density=cc['train_meta_profile']['label_density'],train_zero_support_labels=cc['train_zero_support_labels'],
             D_positive_labels=geometry['decision_grid_coverage']['D_positive_labels'],
             D_grid_unreachable_positive_labels=geometry['decision_grid_coverage']['positive_labels_without_any_reachable_TP'],
             rank90=geometry['rank90'],rank90_lower_bound=geometry['rank90_lower_bound'],top64_energy_fraction=geometry['top64_energy_fraction'],source=dd['provenance']['kind']))
        from broad_core import budget_reference
        reference=budget_reference(cc['outputs'])
        for m in case['models']:
            q=m['quality'];raw=m['raw_quality'];fraction=q['f1_macro']/anchor['quality']['f1_macro'] if anchor['quality']['f1_macro']>0 else None
            all_rows.append(dict(case=cid,dataset=cc['dataset'],seed=cc['seed'],uid=m['uid'],family=m['family'],
                f1_macro=q['f1_macro'],f1_weighted=q['f1_weighted'],loss=q['loss'],brier=q['brier'],raw_f1_macro=raw['f1_macro'],
                threshold_gain_pp=100*(q['f1_macro']-raw['f1_macro']),cost_bytes=m['cost_bytes'],cost_ratio_B0=m['cost_bytes']/reference,
                cost_ratio_full200=m['cost_bytes']/anchor['cost_bytes'],f1_fraction_full200=fraction,below_half_full_quality=fraction is not None and fraction<.5,
                packed_device_array_bytes=m['packed_device_array_bytes'],trees=m['trees'],leaves=m['leaves'],nodes=m['nodes'],blocks=m['blocks'],rank=m['rank'],
                nnz=m['nnz'],basis_values=m['basis_values'],is_low_rank=m['basis_values']>0,has_sparse=m['nnz']>0,parameters=m['parameters']))
    table={(r['case'],r['uid']):r for r in all_rows};points=locked(final['selected_points']['path'],final['selected_points']['sha256'])['points']
    ordinary={(p['case'],p['ratio']):p for p in points if p['family']=='ordinary'};selected=[]
    for p in points:
        common={k:p[k] for k in ('case','family','ratio','limit_bytes','queries','uid')};cc=source_cases[p['case']]['record']['case']
        row=dict(common,dataset=cc['dataset'],seed=cc['seed'])
        if p['uid'] is not None:
            r=table[p['case'],p['uid']];row.update({k:v for k,v in r.items() if k not in ('family',)})
            row['model_family']=r['family'];baseline=ordinary[p['case'],p['ratio']]
            row['delta_ordinary_pp']=100*(r['f1_macro']-table[baseline['case'],baseline['uid']]['f1_macro']) if baseline['uid'] is not None else None
        selected.append(row)
    for row in constants:
        reference=ordinary[row['case'],1.]
        if reference['uid'] is not None:
            q=table[row['case'],reference['uid']];row.update(ordinary100_f1=q['f1_macro'],ordinary100_loss=q['loss'],
                delta_ordinary100_pp=100*(q['f1_macro']-row['f1_macro']))
    csv_write(out/'data/dataset_profiles.csv',profiles);csv_write(out/'data/all_quality_points.csv',all_rows);csv_write(out/'data/selected_quality.csv',selected)
    csv_write(out/'data/constant_diagnostics.csv',constants)
    gains={}
    for family in sorted({r['family'] for r in selected}-{'ordinary'}):
        datasets=[]
        for name in manifest['datasets']:
            rows=[r for r in selected if r['dataset']==name and r['family']==family and r['ratio']==.7 and r.get('delta_ordinary_pp') is not None]
            if rows:datasets.append(dict(dataset=name,mean_delta_pp=float(np.mean([r['delta_ordinary_pp'] for r in rows])),
                          minimum_delta_pp=float(min(r['delta_ordinary_pp'] for r in rows)),maximum_delta_pp=float(max(r['delta_ordinary_pp'] for r in rows)),
                          lowrank_choices=sum(r['is_low_rank'] for r in rows),sparse_choices=sum(r['has_sparse'] for r in rows),
                          selected_families=[r['model_family'] for r in rows]))
        gains[family]=dict(datasets=datasets,mean_dataset_delta_pp=float(np.mean([r['mean_delta_pp'] for r in datasets])) if datasets else None,
                           positive_datasets=sum(r['mean_delta_pp']>0 for r in datasets),datasets_evaluated=len(datasets))
    result=dict(status='broad_quality_rollup_complete',points=len(all_rows),selected_points=len(selected),cases=len(source_cases),families_at70=gains,
                           paid_new_trees=training['paid_new_trees'],test_audits=final['audits'],constant_diagnostics=constants)
    write(out/'quality_summary.json',result);return result,lookup


def inference_tables(cfg,out):
    spec=cfg['broad_analysis'];report=locked(spec['measurements']['path'],spec['measurements']['sha256']);assert report['status']=='wide_measurements_complete'
    assert report['source_tuning']==spec['tuning']
    tuning=locked(spec['tuning']['path'],spec['tuning']['sha256']);entries={(e['case'],e['uid']):e for e in tuning['entries']};memory={};latency=[]
    for ref in report['results']:
        probe=locked(ref['path'],ref['sha256']);key=ref['case'],ref['uid']
        assert probe['status']=='wide_probe_complete' and (probe['case'],probe['uid'])==key
        if ref['mode'].startswith('memory_'):
            backend='packed' if ref['mode']=='memory_packed' else 'legacy'
            for r in probe['rows']:memory[key+(r['batch'],backend)]=r
        else:latency.append(probe)
    rows=[]
    for probe in latency:
        key=probe['case'],probe['uid'];entry=entries[key]
        for batch in cfg['bank_batches']:
            sessions=[r for r in probe['rows'] if r['batch']==batch]
            assert len(sessions)==3 and {r['session'] for r in sessions}=={0,1,2}
            for backend in ('packed','legacy'):
                assert all(len(r['samples'][backend][metric])==32 for r in sessions for metric in ('gpu_ms','request_ms'))
                gpu=[v for r in sessions for v in r['samples'][backend]['gpu_ms']];request=[v for r in sessions for v in r['samples'][backend]['request_ms']]
                g=describe(gpu);q=describe(request);mem=memory[key+(batch,backend)]
                rows.append(dict(case=probe['case'],dataset=entry['dataset'],uid=probe['uid'],roles=json.dumps(entry['roles']),batch=batch,backend=backend,
                    outputs=entry['record']['case']['outputs'],gpu_p50_ms=g['p50_ms'],gpu_p95_ms=g['p95_ms'],request_p50_ms=q['p50_ms'],request_p95_ms=q['p95_ms'],
                    objects_per_second=1000*batch/q['mean_ms'],requests=g['requests'],
                    gpu_session_p50_ms=json.dumps([float(np.median(r['samples'][backend]['gpu_ms'])) for r in sessions]),
                    request_session_p95_ms=json.dumps([float(np.percentile(r['samples'][backend]['request_ms'],95)) for r in sessions]),
                    resident_device_array_bytes=mem['resident_device_array_bytes'],peak_pool_used_bytes=mem['peak_pool_used_bytes'],
                    incremental_peak_pool_bytes=mem['incremental_peak_pool_bytes']))
    csv_write(out/'data/inference_summary.csv',rows)
    pairs=[]
    for r in rows:
        if r['backend']!='packed':continue
        old=next(o for o in rows if o['backend']=='legacy' and (o['case'],o['uid'],o['batch'])==(r['case'],r['uid'],r['batch']))
        gpu_ratios=np.asarray(json.loads(old['gpu_session_p50_ms']))/np.asarray(json.loads(r['gpu_session_p50_ms']))
        request_ratios=np.asarray(json.loads(old['request_session_p95_ms']))/np.asarray(json.loads(r['request_session_p95_ms']))
        pairs.append(dict(case=r['case'],uid=r['uid'],roles=json.loads(r['roles']),batch=r['batch'],
                          gpu_speedup=float(np.median(gpu_ratios)),request_p95_speedup=float(np.median(request_ratios)),
                          gpu_session_ratios=gpu_ratios.tolist(),request_session_ratios=request_ratios.tolist(),
                          peak_reduction_percent=100*(1-r['peak_pool_used_bytes']/old['peak_pool_used_bytes'])))
    write(out/'inference_pairs.json',dict(pairs=pairs));return rows,pairs,tuning


def bank_tables(cfg,out,lookup):
    spec=cfg['broad_analysis'];report=locked(spec['banks']['path'],spec['banks']['sha256']);assert report['status']=='wide_banks_complete'
    assert report['source_tuning']==spec['tuning'];rows=[]
    for ref in report['results']:
        probe=locked(ref['path'],ref['sha256']);bank=probe['bank'];choices=[lookup[c,u] for c,u in zip(bank['cases'],bank['choices'])]
        assert probe['status']=='wide_bank_probe_complete'
        mean_quality=float(np.mean([m['quality']['f1_macro'] for m in choices]));case_index={c:i for i,c in enumerate(bank['cases'])}
        for distribution in ('uniform','hot_wide'):
            tasks=probe['request_plans'][distribution];experienced_quality=float(np.mean([choices[i]['quality']['f1_macro'] for i in tasks]))
            for batch in cfg['bank_batches']:
                sessions=[r for r in probe['rows'] if r['distribution']==distribution and r['batch']==batch]
                assert len(sessions)==3 and all(len(r['request_ms'])==32 for r in sessions)
                times=[v for r in sessions for v in r['request_ms']];q=describe(times);res=probe['residency']
                rows.append(dict(bank=bank['name'],backend=probe['backend'],distribution=distribution,batch=batch,
                    model_count=res['models'],device_array_bytes=res['actual_device_array_bytes'],peak_pool_used_bytes=max(r['peak_pool_used_bytes'] for r in sessions),
                    request_p50_ms=q['p50_ms'],request_p95_ms=q['p95_ms'],objects_per_second=1000*batch/q['mean_ms'],requests=q['requests'],
                    mean_task_f1=mean_quality,request_weighted_f1=experienced_quality,lowrank_models=sum(m['basis_values']>0 for m in choices),
                    sparse_models=sum(m['nnz']>0 for m in choices),hot_wide_case=probe['hot_wide_case']))
    csv_write(out/'data/bank_summary.csv',rows);return rows


def representation_table(rows,lookup,out):
    result=[];cases=list(dict.fromkeys(r['case'] for r in rows))
    for case in cases:
        for batch in sorted({r['batch'] for r in rows}):
            pool=[r for r in rows if r['case']==case and r['batch']==batch and r['backend']=='packed']
            full=next(r for r in pool if 'full200' in json.loads(r['roles']))
            ordinary=next(r for r in pool if 'ordinary70' in json.loads(r['roles']))
            for role in ('direct_AB200','direct_ABL200','direct_AB70','direct_ABL70','posthoc70','controller70','mixed2470'):
                candidate=next((r for r in pool if role in json.loads(r['roles'])),None)
                if candidate is None:continue
                m=lookup[case,candidate['uid']];f=lookup[case,full['uid']];o=lookup[case,ordinary['uid']]
                result.append(dict(case=case,dataset=candidate['dataset'],role=role,batch=batch,uid=candidate['uid'],
                    gpu_vs_full_speedup=full['gpu_p50_ms']/candidate['gpu_p50_ms'],request_vs_full_speedup=full['request_p95_ms']/candidate['request_p95_ms'],
                    gpu_vs_small_ordinary_speedup=ordinary['gpu_p50_ms']/candidate['gpu_p50_ms'],request_vs_small_ordinary_speedup=ordinary['request_p95_ms']/candidate['request_p95_ms'],
                    model_bytes_ratio_full=candidate['resident_device_array_bytes']/full['resident_device_array_bytes'],
                    model_bytes_ratio_ordinary70=candidate['resident_device_array_bytes']/ordinary['resident_device_array_bytes'],
                    peak_bytes_ratio_full=candidate['peak_pool_used_bytes']/full['peak_pool_used_bytes'],
                    test_delta_full_pp=100*(m['quality']['f1_macro']-f['quality']['f1_macro']),test_delta_ordinary70_pp=100*(m['quality']['f1_macro']-o['quality']['f1_macro']),
                    actual_family=m['family'],is_lowrank=m['basis_values']>0,nnz=m['nnz']))
    csv_write(out/'data/representation_comparison.csv',result);return result


def markdown_report(quality,pairs,banks,representations,out):
    mixed=quality['families_at70']['mixed24'];names={'emotions':'Emotions','scene':'Scene','yeast':'Yeast','birds':'Birds','medical':'Medical','genbase':'GenBase',
        'bibtex':'Bibtex','mediamill':'MediaMill','enron':'Enron','cal500':'CAL500','corel5k':'Corel5k','delicious':'Delicious'}
    full_pairs=[p for p in pairs if p['batch']==1024 and 'full200' in p['roles']]
    ab_pairs=[p for p in representations if p['batch']==1024 and p['role']=='direct_ABL200']
    def med(rows,key):return float(np.median([r[key] for r in rows]))
    def speed(ratio):return f'быстрее в {ratio:.2f} раза' if ratio>=1 else f'медленнее в {1/ratio:.2f} раза'
    constant_by_task=defaultdict(list)
    for r in quality['constant_diagnostics']:
        if r['baseline']=='all_positive':constant_by_task[r['dataset']].append(r['delta_ordinary100_pp'])
    constant_positive=sum(np.mean(v)>0 for v in constant_by_task.values())
    family_names={'direct_AB':'AB в обучении','direct_ABL':'AB+L в обучении','controller':'Регулятор',
                  'posthoc':'Сжатие после обучения','mixed24':'Смешанный 12+12','sketch':'SketchBoost'}
    family_lines=['| Семейство | Средняя разница на 12 наборах, п.п. | Положительное среднее | Среднее на четырёх новых наборах, п.п. |',
                  '|---|---:|---:|---:|']
    for family,label in family_names.items():
        result=quality['families_at70'][family]
        fresh=[r['mean_delta_pp'] for r in result['datasets'] if r['dataset'] in ('enron','cal500','corel5k','delicious')]
        family_lines.append(f"| {label} | {result['mean_dataset_delta_pp']:+.2f} | {result['positive_datasets']}/{result['datasets_evaluated']} | {float(np.mean(fresh)):+.2f} |")
    lines=['# Два направления HASBoost: результаты автономной серии','',
       f"Проверены 12 многометочных наборов и 36 разбиений обучающего пула. Число проверенных представлений — {quality['points']}. Расход новых деревьев с учётом предварительных проверок — {quality['paid_new_trees']} при пределе 42 000. Для каждого метода сохранены модели, прогнозы, калибровка и контрольные суммы.",'',
       'D — общая отложенная часть нетестовых данных. Деревья учатся на train; для оценки калибровки каждый объект D получает решение с порогами, подобранными на другом блоке D. Итоговые пороги переоцениваются на всей D, а окончательное качество измеряется на отдельном фиксированном тесте.','',
       '## Что означает результат по бюджету','',
       'Ниже сопоставлены все заранее заданные семейства при одном потолке 0.7 B0. Указана разница тестовой macro-F1 с обычным бустингом, выбранным на D при том же потолке. Положительное среднее не является проверкой статистической значимости. Последний столбец отдельно показывает Enron, CAL500, Corel5k и Delicious; выбор по тесту для них не проводился.','',
       *family_lines,'',
       'AB и AB+L обучались по разным траекториям обратной связи. Их различие нельзя целиком приписать добавлению одного готового остатка. Эта серия меняла его бюджет 0/2/5%; отдельная шкала ослабления R4 на всех 12 наборах не повторялась.','',
       f"При потолке 0.7 B0 смешанный каталог дал среднюю разницу с небольшими обычными бустингами {mixed['mean_dataset_delta_pp']:+.2f} процентного пункта macro-F1. Положительное среднее изменение получено на {mixed['positive_datasets']} из {mixed['datasets_evaluated']} наборов. Это сравнение выбранных на D моделей, а не выбор лучшей точки по тесту.",'',
       'Смешанный каталог содержит обычные и послеобучающие представления. Его положительный результат сам по себе не доказывает пользу малорангового базиса: в таблице ниже отдельно указано, сколько раз он выбрал такой базис. Полноранговое представление с меньшим числом листьев относится к управлению структурой; его нельзя выдавать за результат низкого ранга.','',
       '| Набор | Разница macro-F1, п.п. | Разброс трёх разбиений | Выборов с базисом | Выборов с остатком |',
       '|---|---:|---:|---:|---:|']
    for r in mixed['datasets']:
        lines.append(f"| {names[r['dataset']]} | {r['mean_delta_pp']:+.2f} | {r['minimum_delta_pp']:+.2f} … {r['maximum_delta_pp']:+.2f} | {r['lowrank_choices']}/3 | {r['sparse_choices']}/3 |")
    lines+=['','![Разница качества при общем бюджете](figures/quality-at-budget70.svg)','',
       'Средние точки и отрезки показывают среднее и полный разброс трёх разбиений train/D. Это не доверительные интервалы; тест у разбиений общий. У каждого семейства не более 24 обращений к D. В смешанном каталоге общий предел также 24: 12 обычных и 12 послеобучающих вариантов.','',
       f'Дополнительный контроль: обычная модель, выбранная при B0, превосходит по macro-F1 постоянное решение «все метки положительные» в среднем на {constant_positive} из {len(constant_by_task)} наборов. Это отдельная диагностика абсолютного качества, без новых деревьев и без участия в отборе. Для разреженных меток macro-F1 следует читать вместе со взвешенной F1 и NLL; все три сохранены в таблицах.','',
       '## Что означает результат по инференсу','',
       f"Для полной модели на батче 1024 новый исполнитель по медиане наборов выполняет GPU-стадию {speed(med(full_pairs,'gpu_speedup'))}; p95 полного запроса — {speed(med(full_pairs,'request_p95_speedup'))}. Здесь сравнивается одна и та же функция модели на двух исполнителях; результат относится к коду инференса.",'',
       f"Если сравнить AB+L с полной моделью уже на новом исполнителе, при 200 деревьях медианное отношение объёмов постоянных массивов равно {med(ab_pairs,'model_bytes_ratio_full'):.3f}, а p95 полного запроса у гибрида — {speed(med(ab_pairs,'request_vs_full_speedup'))}. Качество этих моделей различается; полный разбор каждого набора и батча приведён в representation_comparison.csv. По одному отношению памяти нельзя утверждать, что гибрид быстрее.",'',
       '![Ускорение исполнителя](figures/executor-speedup.svg)','',
       'На рисунке: одна сохранённая модель на набор, батч 1024; три сеанса по 32 парных запроса в перемешанном порядке. Отрезки показывают разброс отношения времени по трём сеансам. Слева — GPU, справа — запрос с передачами и возвращением вероятностей и меток на CPU. Параметры float32, накопления и вероятности float64 у обоих исполнителей.','',
       '## Банки моделей','',
       '| Банк, новый исполнитель | Моделей | Массивы GPU, МиБ | Пик собственного пула, МиБ | Средняя macro-F1 | p95 запроса, мс |',
       '|---|---:|---:|---:|---:|---:|']
    for r in banks:
        if r['backend']=='packed' and r['batch']==1024 and r['distribution']=='uniform':
            lines.append(f"| {r['bank']} | {r['model_count']} | {r['device_array_bytes']/1024**2:.2f} | {r['peak_pool_used_bytes']/1024**2:.2f} | {r['mean_task_f1']:.4f} | {r['request_p95_ms']:.3f} |")
    lines+=['',
       'Все модели банка действительно размещены одновременно. Небольшой обычный и смешанный банки отбирались при одном общем потолке памяти по качеству на D. Отбор использовал четыре уже зафиксированные точки бюджета каждой задачи; новые обращения к D не добавлялись. Полный банк служит отдельным ориентиром. Если оба банка помещаются на A100, это подтверждает экономию байтов; максимальное число различных задач этим опытом не установлено.','',
       '## Что реализовано','',
       '- Общая схема train/D без отдельных выборок для каждого порога: два блока по группам признаков, одинаковые правила для всех методов, итоговые пороги по всей D. Температура многометочных выходов T=1; настройка порогов меняет решения, а не вероятности.',
       '- Обычные модели глубины 3 и 5, SketchBoost, прямое обучение AB и AB+L, послеобучающее управление листьями/рангом/остатком и регулятор с реальной обратной связью. Остаток может отсутствовать. Для прямого обучения квота поправок переносится между деревьями.',
       '- Упакованный исполнитель для нескольких базисов: обход, сбор коэффициентов, остаток, декодирование, пороги и вероятности в одном CUDA-ядре. Нет глобальных промежуточных матриц «батч × ранг», «батч × остаток» и отдельного выхода на каждый блок. Полная модель получает ту же оптимизацию.',
       '- Для широких выходов рассчитываются первые 64 направления взвешенного базиса. Полный выход остаётся контролем. Проверены численные инварианты, экспорт моделей, калибровка и фактические начальные прогнозы каждого следующего участка обучения.','',
       '## Границы выводов','',
       'Восемь наборов используют уже виденный исторический тест и дают исследовательскую оценку. Enron, CAL500, Corel5k и Delicious впервые оценены после фиксации этой серии. Для Enron и CAL500 тест выделен по признакам без использования меток; для двух остальных сохранено исходное разбиение хранилища KDIS. Дубликаты признаков не пересекают train/D/test. У MediaMill сохранён прежний тест, один совпавший с ним обучающий объект исключён; используется прежняя когорта, а не весь исходный набор.','',
       'B0 — аналитическая координата памяти, рассчитанная до обучения. 0.7 B0 не означает автоматически 30% экономии относительно фактически обученной полной модели. В CSV приведены абсолютные байты и оба отношения. Память модели отделена от входов, выходов и пика собственного пула CuPy. Сторонний контекст A100 в наши байты не включён; окружения сохранены.','',
       'Столбец parameters сохраняет прежний счётчик коэффициентов, смещений и калибровки. Число внутренних узлов дано отдельно в nodes; поля структуры полностью включены в байты. Поэтому выводы об экономии памяти не зависят от трактовки числа обучаемых параметров.','',
       'При батче больше D строки повторяются. Эти измерения характеризуют стенд, а не произвольный поток запросов. p95 оценён по 96 запросам, включая три сеанса; это не проверка эксплуатационного SLA. Для каждого нового исполнителя подбор схемы шёл на отдельных замерах D до итогового сравнения.','',
       'Сетка порогов фиксирована и одинакова для всех методов. В профилях дополнительно отмечены метки, для которых ни один положительный объект D не достигает её нижней границы. Это диагностирует ограничение калибровки без новых настроек; его нельзя автоматически приписывать плохому базису. Адаптивная сетка здесь не проверялась.','',
       'Глубины ограничены 3/5 и 200 деревьями; прямой малоранговый ранг ограничен 64. Это границы проверенного каталога, а не доказательство мирового оптимума или превосходства над всеми настройками CatBoost/XGBoost. История G/H, параллельные ветви обучения, TabPFN и перенос в другие проекты в эту серию не входили.','',
       '## Материалы для проверки','',
       '- [Все точки качества](data/all_quality_points.csv), [зафиксированный отбор](data/selected_quality.csv), [профили данных](data/dataset_profiles.csv).',
       '- [Постоянные диагностические контроли](data/constant_diagnostics.csv): решения фиксированы, вероятности взяты из частот обучающей выборки.',
       '- [Замеры моделей](data/inference_summary.csv), [сравнение представлений](data/representation_comparison.csv), [реальные банки](data/bank_summary.csv).',
       '- [Сводка качества](quality_summary.json), [парные ускорения](inference_pairs.json), [общий отчёт и ссылки на первичные запуски](report.json).','',
       'Кривые каждого набора сохраняют все измеренные варианты бледными точками. Выделенные точки выбраны только по D; линии лишь соединяют эти точки и не означают измерение промежуточных моделей. Отрезки показывают разброс трёх разбиений. Непрошедшие малый бюджет варианты не скрываются из полной таблицы.','']
    for name in names:lines.append(f"- [{names[name]}: качество и память](figures/budget-{name}.svg)")
    (out/'Итоги двух направлений.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);environment=check_native_environment(cfg)
    quality,lookup=quality_tables(cfg,out);rows,pairs,tuning=inference_tables(cfg,out);banks=bank_tables(cfg,out,lookup)
    representations=representation_table(rows,lookup,out);markdown_report(quality,pairs,banks,representations,out)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    for name in ('plot_results.py','orx_figstyle.py'):shutil.copyfile(Path(__file__).parent/'two_direction_figs'/name,figs/name)
    import subprocess
    # Numerical rollup stays in the native runtime. Matplotlib is already part
    # of the separately frozen external runtime; neither environment is changed.
    plot_root=Path(cfg['environment_root']);plot_python=plot_root/'venv/bin/python'
    assert digest(plot_root/'environment.json')==cfg['external_environment_sha256']
    plot_environment=json.loads(subprocess.check_output([str(plot_python),'-c',
        'import sys,json,matplotlib;print(json.dumps(dict(python=sys.version,matplotlib=matplotlib.__version__)))'],text=True,timeout=30))
    plot_environment.update(executable=str(plot_python),descriptor_sha256=cfg['external_environment_sha256'])
    subprocess.run([str(plot_python),'-B',str(figs/'plot_results.py'),str(out/'data'),str(figs)],check=True,timeout=600)
    report=dict(status='two_directions_analysis_complete',sources=cfg['broad_analysis'],environment=environment,
                plot_environment=plot_environment,
                quality_points=quality['points'],serving_rows=len(rows),bank_rows=len(banks),new_trees=0,
                outputs=[dict(path=str(p),sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report.json'])
    write(out/'report.json',report);event('two_directions_analysis_complete',quality_points=quality['points'],serving_rows=len(rows),bank_rows=len(banks))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
