import React from 'react';

function Tree({x,y,color='currentColor'}) {
  return <g transform={`translate(${x},${y})`} fill="none" stroke={color} strokeWidth="1.8"><path d="M12 0v9L2 20m10-11 10 11M2 20v9m20-9v9"/><circle cx="12" cy="0" r="3" fill={color}/><rect x="-2" y="29" width="8" height="6" rx="1" fill={color} opacity=".28"/><rect x="18" y="29" width="8" height="6" rx="1" fill={color}/></g>;
}
function Answer({x,y}) {
  return <g transform={`translate(${x},${y})`} fill="currentColor"><rect width="8" height="16" y="13" rx="2" opacity=".35"/><rect width="8" height="29" x="12" rx="2" opacity=".7"/><rect width="8" height="22" x="24" y="7" rx="2"/><path d="m1 39 4 4 6-8m5 4 4 4 6-8" fill="none" stroke="currentColor" strokeWidth="2"/></g>;
}
function Wire({x,y,w=13}) {return <path d={`M${x} ${y}h${w}m-4-4 4 4-4 4`} fill="none" stroke="currentColor" strokeWidth="1.6"/>;}
function Native() {
  return <svg viewBox="0 0 360 152" role="img" aria-label="Четыре дерева: для каждого отдельный поиск листа и отдельное прибавление значений, затем обработка выхода">
    {[0,1,2,3].map(i=><g key={i} transform={`translate(${8+i*85},16)`}><text x="36" y="0" textAnchor="middle">Дерево {i+1}</text><rect x="0" y="12" width="32" height="57" rx="6" fill="currentColor" opacity=".09"/><Tree x={4} y={24}/><Wire x={33} y={40} w={10}/><rect x="46" y="12" width="30" height="57" rx="6" fill="currentColor" opacity=".16"/><text x="61" y="49" textAnchor="middle" className="hb-primer-sigma">+</text><text x="16" y="85" textAnchor="middle">лист</text><text x="61" y="85" textAnchor="middle">сумма</text></g>)}
    <text x="180" y="136" textAnchor="middle" className="hb-primer-svg-note">8 отдельных запусков + обработка выхода</text>
  </svg>;
}
function Legacy({wide,compact}) {
  return <svg viewBox="0 0 360 152" role="img" aria-label={wide?(compact?'Два блока деревьев, отдельное декодирование базиса и обработка выхода':'Два блока деревьев, сбор полных листьев, отдельное суммирование и обработка выхода'):'Два блока деревьев: один объединённый запуск на блок, затем обработка выхода'}>
    {[0,1].map(i=><g key={i} transform={`translate(${10+i*132},16)`}><text x="54" y="0" textAnchor="middle">Блок {i+1}</text><rect y="12" width="110" height="76" rx="8" fill="currentColor" opacity=".10"/><Tree x={12} y={25}/><Tree x={44} y={25}/><text x="86" y="51" textAnchor="middle" className="hb-primer-sigma">Σ</text><text x="55" y="78" textAnchor="middle">{wide?(compact?'сбор → базис':'сбор → сумма'):'обход + сумма'}</text><Wire x={113} y={47}/></g>)}
    <Answer x={300} y={42}/><text x="316" y="110" textAnchor="middle">выход</text>
    <text x="180" y="136" textAnchor="middle" className="hb-primer-svg-note">{wide?(compact?'По блокам + декодирование + выход':'По блокам + суммирование + выход'):'2 объединённых запуска + обработка выхода'}</text>
  </svg>;
}
function Packed({compact}) {
  return <svg viewBox="0 0 360 152" role="img" aria-label="Все четыре дерева, суммирование и обработка выхода внутри одного запуска GPU">
    <rect x="8" y="17" width="344" height="102" rx="10" fill="currentColor" opacity=".07"/><rect x="8" y="17" width="344" height="102" rx="10" fill="none" stroke="currentColor" strokeWidth="1.6"/>
    {[20,53,86,119].map(x=><Tree key={x} x={x} y={40}/>)}<Wire x={149} y={61} w={17}/><text x="205" y="64" textAnchor="middle" className="hb-primer-sigma">Σ</text><text x="205" y="95" textAnchor="middle">{compact?'базис + поправки':'общая сумма'}</text><Wire x={244} y={61} w={23}/><Answer x={294} y={39}/>
    <text x="180" y="136" textAnchor="middle" className="hb-primer-svg-note">1 запуск: деревья → сумма → готовый ответ</text>
  </svg>;
}

/** A four-tree teaching example, never a measured launch/time chart. */
export function ExecutorFlow({wide,compact,focus='packed',onFocus=()=>{}}) {
  const rows=[
    {id:'native',name:'Исходный Py-Boost',tag:compact?'Полный ориентир W':'По одному дереву',lead:'На каждое дерево — поиск листа, затем прибавление значений.',svg:<Native/>,detail:'Два разных запуска на дерево: один находит лист, другой прибавляет его значения. GPU обрабатывает весь батч, но получает много отдельных команд. После последнего дерева вероятности и метки вычисляются отдельно.'},
    {id:'legacy',name:'Legacy',tag:'По блокам деревьев',lead:'Несколько деревьев обрабатываются вместе. Результаты блоков складываются.',svg:<Legacy wide={wide} compact={compact}/>,detail:wide?(compact?'Прежний оптимизированный путь для более 128 выходов. Один блок собирает координаты и поправки, затем отдельные операции декодируют базис и суммируют результат. Вероятности и метки вычисляются после блоков. Legacy уже оптимизирован относительно исходного Py-Boost.':'Прежний оптимизированный путь для более 128 выходов. Один блок собирает полные значения листьев, затем его результат отдельно прибавляется к общему прогнозу. Базиса в этом представлении нет. Вероятности и метки вычисляются после блоков.'):'Прежний оптимизированный путь для не более 128 выходов. Деревья одного блока и его сумма обрабатываются совместно. При нескольких блоках остаются отдельные вызовы и общий накопитель; вероятности и метки считаются после них.'},
    {id:'packed',name:'Packed',tag:'Весь ансамбль',lead:'Обход деревьев, сумма и обработка выхода объединены в один запуск.',svg:<Packed compact={compact}/>,detail:compact?'Координаты достигнутых листьев суммируются внутри ядра, затем применяется базис и добавляются адресные поправки. Вероятности и метки вычисляются там же. Глобального промежуточного массива «батч × ранг» нет; параметры, вход и конечный выход всё равно занимают GPU-память.':'Все блоки полной модели проходят в одном ядре. Значения листьев суммируются внутри запуска; вероятности и метки вычисляются там же. Вход, параметры и конечный выход остаются в памяти GPU.'}
  ];
  return <div className="hb-primer">
    <p className="hb-primer-intro">Условный пример: <b>4 дерева, 2 блока</b>. Каждый CUDA-запуск работает с батчем целиком. Рамки показывают, какую работу движок объединяет.</p>
    <div className="hb-primer-grid">{rows.map(r=><button key={r.id} className={`hb-primer-card hb-primer-${r.id}`} aria-pressed={focus===r.id} onClick={()=>onFocus(r.id)}><span className="hb-primer-tag">{r.tag}</span><strong>{r.name}</strong>{r.svg}<span className="hb-primer-lead">{r.lead}</span><span className="hb-primer-prompt">Пояснить этот вариант ↓</span></button>)}</div>
    <div className="hb-primer-detail" aria-live="polite"><b>{rows.find(r=>r.id===focus).name} · что важно</b><p>{rows.find(r=>r.id===focus).detail}</p></div>
    <p className="hb-schematic-note">Это схема организации работы, не шкала скорости. Один большой запуск не гарантирует наименьшую задержку; результат проверяется по измерениям.</p>
  </div>;
}
