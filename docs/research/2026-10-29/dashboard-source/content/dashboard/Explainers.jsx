import React from 'react';
import {DataComponent, DataTable, Dropdown, SectionHeader, SegmentedControl, useDataApp} from '../../data-app-public.jsx';
import catalog from './mechanisms.json';
import {FAMILIES, LABELS, COLORS, ROLES} from './analytics.js';
import './explainers.css';

export const BACKENDS={legacy:'Legacy · по блокам',packed:'Packed · единое ядро'};
const baseUrl=`https://github.com/leostre/Py-Boost/blob/${catalog.source_commit}/`;
const byId=Object.fromEntries(catalog.methods.map(m=>[m.id,m]));

function Tiles({x,y,rows,cols,kind='dense',cell=8}) {
  return <g>{Array.from({length:rows*cols},(_,i)=>{
    const row=Math.floor(i/cols),col=i%cols;
    const active=kind==='sparse'?(row*3+col*5)%11===0:kind==='sketch'?col%2===0:true;
    return <rect key={i} x={x+col*(cell+2)} y={y+row*(cell+2)} width={cell} height={cell} rx="1.5" fill="currentColor" opacity={active?(kind==='sparse'?.95:.28+((row+col)%3)*.17):.06}/>;
  })}</g>;
}

/** Schematic topology/coefficient marks, with no fabricated measured quantities. */
function Glyph({type}) {
  let drawing;
  if(type==='tree'||type==='feedback') drawing=<><path d="M70 14v9M70 23L40 38M70 23l30 15M40 38L25 57M40 38l15 19M100 38L85 57M100 38l15 19"/><circle cx="70" cy="14" r="5"/><circle cx="40" cy="38" r="4"/><circle cx="100" cy="38" r="4"/>{[25,55,85,115].map(x=><rect key={x} x={x-7} y="56" width="14" height="10" rx="2" fill="currentColor" opacity=".55"/>)}{type==='feedback'&&<path d="M117 73H20V29m-5 7 5-7 5 7" strokeDasharray="4 3"/>}</>;
  else if(type==='dense'||type==='gradient'||type==='sketch') drawing=<><Tiles x={30} y={14} rows={5} cols={8} kind={type==='sketch'?'sketch':'dense'}/>{type==='gradient'&&<path d="M18 17v44m-4-39 4-5 4 5M18 61h95" opacity=".5"/>}</>;
  else if(type==='basis') drawing=<><path d="M30 64V18m0 46 42-37M30 64h69M27 24l3-6 3 6m60 37 6 3-6 3M65 30l7-3-2 7"/><Tiles x={92} y={15} rows={2} cols={4} cell={7}/><path d="M81 43h18m-5-4 5 4-5 4"/></>;
  else if(type==='latent') drawing=<><Tiles x={14} y={17} rows={5} cols={2}/><text x="45" y="45">×</text><Tiles x={60} y={29} rows={2} cols={7}/></>;
  else if(type==='hybrid') drawing=<><Tiles x={5} y={21} rows={4} cols={2} cell={6}/><text x="25" y="43">×</text><Tiles x={39} y={28} rows={2} cols={5} cell={6}/><text x="83" y="43">+</text><Tiles x={99} y={21} rows={4} cols={5} kind="sparse" cell={6}/></>;
  else if(type==='sparse') drawing=<><Tiles x={30} y={14} rows={5} cols={8} kind="sparse"/><path d="M20 55 30 45m-3 7 3-7-7 3"/></>;
  else if(type==='controller') drawing=<><rect x="41" y="24" width="58" height="26" rx="8"/><text x="70" y="42" textAnchor="middle">бюджет</text><path d="M41 37H18V16h30m51 21h23v29H94M48 12l5 4-5 4m46 42-5 4 5 4"/><circle cx="27" cy="66" r="8" opacity=".55"/><circle cx="70" cy="66" r="8" opacity=".55"/></>;
  else drawing=<><rect x="14" y="15" width="43" height="44" rx="5"/><rect x="77" y="15" width="43" height="44" rx="5"/><path d="M57 36h20m-6-4 6 4-6 4"/><path d="m87 38 7 7 16-19" strokeWidth="3"/></>;
  return <svg className="hb-explain-glyph" viewBox="0 0 140 84" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{drawing}</svg>;
}

function MatrixRepresentation() {
  return <div className="hb-representation" aria-label="Матрица значений листьев W приближается произведением A B и разреженной поправкой L">
    <div><Glyph type="dense"/><strong>W</strong><span>M листьев × C выходов</span></div>
    <b>≈</b><div><Glyph type="latent"/><strong>A × B</strong><span>(M × r) · (r × C)</span></div>
    <b>+</b><div><Glyph type="sparse"/><strong>L</strong><span>Выбранные пары «лист, выход»</span></div>
  </div>;
}

export function MethodsView({q}) {
  const {sectionViews,setSectionView}=useDataApp();
  const pref=sectionViews['hb-methods']??{};
  const family=byId[pref.family]?pref.family:'direct_ABL';
  const m=byId[family],step=Math.min(m.steps.length-1,Number(pref.step??0));
  const choose=id=>setSectionView('hb-methods',{family:id,step:0});
  const source=q.method_mechanisms.rows.filter(r=>r.id===family);
  return <>
    <SectionHeader id="methods-heading" title="Как устроены семь семейств"/>
    <div className="hb-levels"><span><b>Обучение</b> · как строим деревья</span><span><b>Представление</b> · что храним в листьях</span><span><b>Отбор</b> · какую модель берём при лимите</span></div>
    <p className="hb-explain-intro">Названия в легенде относятся к этим трём уровням. Например, Sketch меняет построение дерева, AB — ещё и представление приращений, а смешанный отбор выбирает готового кандидата.</p>
    <div className="hb-method-selector" role="group" aria-label="Семейство для пояснения">
      {FAMILIES.map(id=><button key={id} aria-pressed={family===id} onClick={()=>choose(id)} style={{'--hb-method-color':COLORS[id]}}><span className="hb-method-swatch"/><span>{LABELS[id]}</span><small>{byId[id].category}</small></button>)}
    </div>
    <DataComponent id="method-mechanism" queryId="method_mechanisms" title={LABELS[family]} kind="diagram" sourceRows={source} displayRows={source}>
      <p className="hb-explain-lead">{m.summary}</p>
      <div className="hb-pipeline" role="group" aria-label="Этапы метода">
        {m.steps.map((s,i)=><React.Fragment key={s.title}>{i>0&&<span className="hb-pipeline-arrow" aria-hidden="true">→</span>}<button className="hb-stage" aria-pressed={i===step} onClick={()=>setSectionView('hb-methods',{family,step:i})}><span className="hb-step-number">{i+1}</span><Glyph type={s.icon}/><strong>{s.title}</strong></button></React.Fragment>)}
      </div>
      <div className="hb-stage-detail" aria-live="polite"><b>Шаг {step+1}. {m.steps[step].title}</b><p>{m.steps[step].detail}</p></div>
      <div className="hb-explain-facts"><div><b>Что хранится</b><p>{m.storage}</p></div><div><b>Что проверено в этой серии</b><p>{m.scope}</p></div></div>
      <p className="hb-explain-takeaway">{m.difference}</p>
      <p className="hb-schematic-note">Схема механизма, без масштаба и измеренных величин. Шаги можно переключать.</p>
    </DataComponent>
    <DataComponent id="representation-primer" queryId="method_mechanisms" title="Что означают A, B и L" kind="diagram" sourceRows={q.method_mechanisms.rows} displayRows={q.method_mechanisms.rows}>
      <MatrixRepresentation/>
      <div className="hb-explain-facts"><div><b>A · координаты листьев</b><p>Вместо C отдельных чисел лист хранит r коэффициентов. Матрица B превращает их в полный вектор по выходам.</p></div><div><b>L · адресные поправки</b><p>Сохраняет только выбранные дополнительные коэффициенты. Сумма общего базиса и этих поправок образует гибрид.</p></div></div>
      <p className="hb-schematic-note">M — число листьев блока; C — число выходов; r — ранг. У модели могут быть несколько блоков со своими базисами и полный начальный участок. В схеме B записана в форме r × C; часть обучающего кода хранит её транспонированной. Порог применяется уже к полученному прогнозу.</p>
    </DataComponent>
    <DataComponent id="methods-comparison" queryId="method_mechanisms" title="Семейства рядом: обучение и хранение" kind="table" sourceRows={q.method_mechanisms.rows} displayRows={q.method_mechanisms.rows}>
      <DataTable rows={q.method_mechanisms.rows} label="Семейства: обучение и хранение" searchable={false} columns={[{field:'title',label:'Семейство'},{field:'category',label:'Что меняем'},{field:'summary',label:'Принцип'},{field:'storage',label:'Что хранится'}]}/>
    </DataComponent>
    <details className="hasboost-details"><summary>Реализация, по которой сверены схемы</summary><ul className="hb-code-sources">{[...new Set(catalog.methods.flatMap(m=>m.sources))].map(p=><li key={p}><a href={baseUrl+p} target="_blank" rel="noreferrer">{p}</a></li>)}</ul></details>
  </>;
}

function EngineBox({backend,role,reference=false}) {
  return <div className={`hb-compare-model hb-compare-${backend}`}><span>{reference?'B · ориентир':'A · измеряемая модель'}</span><strong>{ROLES[role]??role}</strong><small>{BACKENDS[backend]}</small></div>;
}

export function InferenceComparison({mode,role,referenceRole,example,sourceRows}) {
  const same=mode==='executor';
  return <DataComponent id="inference-comparison-guide" queryId="inference_summary" title={same?'Меняем только исполнение сохранённой модели':'Меняем модель, сохраняем исполнителя'} kind="diagram" sourceRows={sourceRows} displayRows={sourceRows}>
    <div className="hb-compare-pair"><EngineBox backend={same?'legacy':'packed'} role={same?role:referenceRole} reference/><div className="hb-compare-operation"><b>{same?'Одна модель':'Две модели'}</b><span>{same?'Те же деревья, коэффициенты и пороги':'Могут различаться деревья, листья, ранг и поправки'}</span></div><EngineBox backend="packed" role={role}/></div>
    <div className="hb-speed-formula"><span>Коэффициент ускорения</span><strong>время B ÷ время A</strong><span><b>&gt; 1×</b> · A быстрее <i/> <b>&lt; 1×</b> · A медленнее</span></div>
    <a className="hb-executor-jump" href="#hb-executor-scheme">Схема: как работают legacy и packed ↓</a>
    <p className="hb-explain-takeaway">{same?`Сейчас ориентир — та же «${ROLES[role]}» на legacy. При смене модели меняется и её собственный ориентир legacy. Поэтому даже полная модель может получить ускорение: сравниваются два способа вычислить её прогноз.`:`Сейчас ориентир зафиксирован: «${ROLES[referenceRole]}» на packed. Сравнение показывает эффект выбора представления и размера модели на одном исполнителе. Рядом обязательно учитываем изменение качества и памяти.`}</p>
    {same&&example&&<div className="hb-numeric-example"><b>Пример из замеров: {example.dataset}, первый сеанс</b><span>p95 запроса: {example.legacyMs.toLocaleString('ru-RU',{maximumFractionDigits:4})} мс (legacy) ÷ {example.packedMs.toLocaleString('ru-RU',{maximumFractionDigits:4})} мс (packed) = <b>{example.ratio.toLocaleString('ru-RU',{maximumFractionDigits:3})}×</b></span><small>UID с обеих сторон: {example.uid}. Итоговая карточка выше таблицы — медиана отношений трёх сеансов, затем медиана по выбранным наборам.</small></div>}
    <div className="hb-timing-scopes"><div><b>GPU p50</b><p>По CUDA-событиям: вся цепочка legacy либо одно ядро packed, включая вероятности и метки. Вход уже на GPU; выделение основных входных и выходных буферов и передача CPU ↔ GPU за пределами этого замера.</p></div><div><b>p95 запроса</b><p>Весь вызов request: подготовка входа, передача на GPU, выделения, вычисления и получение результата на CPU. Это другая граница измерения.</p></div></div>
  </DataComponent>;
}

export function ExecutorDiagram({q,outputs}) {
  const {sectionViews,setSectionView}=useDataApp();
  const route=sectionViews['hb-executor-diagram']?.route??(outputs!=null&&outputs<=128?'narrow':'wide');
  const wide=route==='wide';
  const legacySteps=wide?['Обход деревьев и сбор значений листьев по блоку','Глобальные рабочие буферы и остаток, если он есть','Декодирование через B, если нужен базис; суммирование блоков','Отдельные CuPy-операции: вероятности и метки']:['Уже объединённое ядро внутри каждого блока','При нескольких блоках: рабочий выход и суммирование','Отдельные CuPy-операции: вероятности и метки'];
  return <DataComponent id="executor-mechanism" queryId="executor_mechanisms" title="Legacy и packed: как вычисляется один прогноз" kind="diagram" sourceRows={q.executor_mechanisms.rows} displayRows={q.executor_mechanisms.rows} headerControls={<SegmentedControl ariaLabel="Поясняемый путь legacy" value={route} options={[{value:'narrow',label:'Legacy: до 128 выходов'},{value:'wide',label:'Legacy: более 128'}]} onChange={r=>setSectionView('hb-executor-diagram',{route:r})}/>}>
    <div className="hb-request-ends"><span>Вход X на CPU</span><b>→</b><span>Передача на GPU</span><b>→</b><strong>Вычисления ниже</strong><b>→</b><span>Результат на CPU</span></div>
    <div className="hb-engine-grid"><div className="hb-engine-lane"><h3>Legacy · по блокам</h3><p>Прежний оптимизированный путь{wide?' для широких выходов':'. Уже использует объединённое ядро для каждого блока'}.</p><ol>{legacySteps.map(s=><li key={s}>{s}</li>)}</ol></div><div className="hb-engine-lane hb-packed-lane"><h3>Packed · единое ядро</h3><p>Один запуск CUDA-ядра на весь батч и все блоки модели.</p><div className="hb-kernel-boundary"><span>Внутри одного запуска</span><ol><li>Группы потоков обрабатывают деревья</li><li>Собирают коэффициенты и поправки в регистрах и общей памяти блока</li><li>Декодируют базисы B, если они есть, и складывают приращения</li><li>Вычисляют вероятности и метки</li></ol></div><p className="hb-schematic-note">Глобальный промежуточный массив «батч × ранг» не создаётся. Входы, параметры и окончательные выходы всё равно занимают память GPU.</p></div></div>
    <p className="hb-explain-takeaway">Модель и её пороги сохраняются. Параметры и входы — float32, суммы и вероятности — float64 в обоих путях; прогнозы сверены с численным допуском. Packed объединяет больше стадий в одном запуске; это не дополнительное сжатие модели.</p>
    <p className="hb-schematic-note">Переключатель меняет поясняющую схему legacy, а не результаты замеров. {outputs!=null?`У выбранного набора ${outputs} выходов.`:'При выборе всех наборов в измерениях есть оба пути legacy.'} Подготовка модели и компиляция не входят в прогретые времена запроса.</p>
    <details className="hb-inline-sources"><summary>Где это реализовано</summary>{catalog.executors.map(e=><p key={e.id}><b>{e.title}:</b> {e.sources.map((p,i)=><React.Fragment key={p}>{i>0?' · ':''}<a href={baseUrl+p} target="_blank" rel="noreferrer">{p.split('/').at(-1)}</a></React.Fragment>)}</p>)}<p><a href={baseUrl+'research/external_controls/wide_probe.py'} target="_blank" rel="noreferrer">Границы GPU-замера и request в wide_probe.py</a></p></details>
  </DataComponent>;
}
