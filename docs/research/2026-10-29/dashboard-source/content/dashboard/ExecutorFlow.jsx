import React from 'react';

function Tree({x,y,color}) {
  return <g transform={`translate(${x},${y})`} stroke={color} strokeWidth="2" fill="none"><path d="M32 0v13L12 31m20-18 20 18M12 31 2 48m10-17 11 17M52 31 42 48m10-17 11 17"/><path d="M32 0v13L12 31l11 17" strokeWidth="4"/><circle cx="32" cy="0" r="4" fill={color}/>{[2,23,42,63].map(v=><rect key={v} x={v-5} y="48" width="10" height="8" rx="2" fill={color} opacity={v===23?1:.3}/>)}</g>;
}
function Matrix({x,y,rows=4,cols=5,label,sub,sparse=false,local=false}) {
  const cw=local?8:10,gap=3,w=cols*(cw+gap)-gap;
  return <g className={local?'hb-flow-local':'hb-flow-memory'} transform={`translate(${x},${y})`}><rect x="-9" y="-10" width={w+18} height={rows*(cw+gap)+17} rx="6" className="hb-flow-matrix-bg"/>{Array.from({length:rows*cols},(_,i)=><rect key={i} x={(i%cols)*(cw+gap)} y={Math.floor(i/cols)*(cw+gap)} width={cw} height={cw} rx="1.5" fill="currentColor" opacity={sparse?((i*7)%13===0?.9:.07):.3+((i+Math.floor(i/cols))%3)*.2}/>)}<text x={w/2} y={rows*(cw+gap)+29} textAnchor="middle" className="hb-flow-label">{label}</text>{sub&&<text x={w/2} y={rows*(cw+gap)+46} textAnchor="middle" className="hb-flow-sub">{sub}</text>}</g>;
}
function Arrow({x1,y1,x2,y2,memory=false,label}) {
  return <g className={memory?'hb-flow-memory':'hb-flow-wire'}><path d={`M${x1} ${y1}H${x2-5}L${x2} ${y2}`} fill="none" stroke="currentColor" strokeWidth={memory?2.5:1.6}/><path d={`m${x2-6} ${y2-4} 6 4-6 4`} fill="none" stroke="currentColor" strokeWidth="1.6"/>{label&&<text x={(x1+x2)/2} y={y1-10} textAnchor="middle" className="hb-flow-sub">{label}</text>}</g>;
}
function Kernel({x,y,w,h=110,label,packed=false}) {
  return <g className={packed?'hb-flow-packed':'hb-flow-kernel'}><rect x={x} y={y} width={w} height={h} rx="12"/><text x={x+13} y={y+21} className="hb-flow-kernel-title">{label}</text></g>;
}
function Output({x,y}) {
  return <g transform={`translate(${x},${y})`}><path d="M0 45h68M4 40V0" fill="none" className="hb-flow-wire" stroke="currentColor"/>{[14,30,46,62].map((v,i)=><rect key={v} x={v-7} y={40-[11,31,21,37][i]} width="10" height={[11,31,21,37][i]} rx="2" fill="currentColor" opacity=".65"/>)}<path d="m3 63 5 5 10-12m13 7 5 5 10-12M59 60l9 9m0-9-9 9" fill="none" stroke="currentColor" strokeWidth="2"/><text x="33" y="88" textAnchor="middle" className="hb-flow-sub">p + метки</text></g>;
}
function LaunchStrip({y,count,packed=false,label}) {
  return <g transform={`translate(185,${y})`} className={packed?'hb-flow-packed':'hb-flow-wire'}><text x="0" y="13" className="hb-flow-sub">Запуски</text>{Array.from({length:count},(_,i)=><rect key={i} x={75+i*20} y="0" width={packed?270:13} height="16" rx="3" fill="currentColor" opacity={packed?.8:.45}/>) }<text x={packed?365:75+count*20+8} y="13" className="hb-flow-sub">{label}</text></g>;
}

/** Code-backed dataflow. Geometry and matrix cell counts are schematic. */
export function ExecutorFlow({wide,compact}) {
  const narrow=!wide;
  return <div className="hb-flow-scroll" tabIndex="0" aria-label="Схема исполнения. На узком экране прокручивается по горизонтали."><svg className="hb-flow-svg" viewBox="0 0 1120 774" role="img" aria-labelledby="hb-flow-title hb-flow-desc">
    <title id="hb-flow-title">Py-Boost, legacy и packed: обход деревьев, буферы и запуски CUDA</title>
    <desc id="hb-flow-desc">Py-Boost запускает обход и сбор значений отдельно для каждого дерева. Legacy объединяет вычисления внутри блока при числе выходов до 128, а для широких выходов записывает промежуточные массивы и использует отдельное декодирование. Packed обрабатывает все блоки одним ядром, хранит промежуточные суммы в регистрах и общей памяти CUDA-блока.</desc>
    {[24,266,508].map(y=><rect key={y} x="0" y={y} width="1120" height="225" rx="14" className="hb-flow-lane"/>)}
    <g className="hb-flow-native">
      <text x="20" y="55" className="hb-flow-heading">Py-Boost</text><text x="20" y="77" className="hb-flow-sub">Исходный master</text><text x="20" y="98" className="hb-flow-sub">Полные листья W</text><text x="20" y="122" className="hb-flow-sub">float32</text>
      <Kernel x={180} y={45} w={150} label="1 · обход"/><Tree x={222} y={91} color="currentColor"/>
      <Arrow x1={334} y1={115} x2={372} y2={115} memory/>
      <Matrix x={392} y={84} cols={1} rows={4} label="J"/>
      <Arrow x1={418} y1={115} x2={467} y2={115} memory/>
      <Kernel x={475} y={45} w={175} label="2 · сбор + сумма"/><Matrix x={491} y={86} rows={3} cols={4} label="W" local/><text x="556" y="119" className="hb-flow-symbol">→ +</text>
      <Arrow x1={654} y1={115} x2={699} y2={115} memory/>
      <Matrix x={720} y={84} label="S ← S + W[J]" sub="батч × C"/>
      <Arrow x1={788} y1={115} x2={835} y2={115} memory/>
      <Kernel x={843} y={45} w={250} label="После последнего дерева"/><path d="M860 126c20 0 10-38 29-38h10" fill="none" stroke="currentColor" strokeWidth="2"/><Output x={984} y={87}/>
      <path d="M788 183v17H187v-33m-5 6 5-6 5 6" className="hb-flow-repeat"/><text x="479" y="192" textAnchor="middle" className="hb-flow-sub">Повтор для каждого дерева · T раз</text>
      <LaunchStrip y={222} count={16} label="… 2T запусков + обработка выхода"/>
    </g>
    <g className="hb-flow-legacy">
      <text x="20" y="297" className="hb-flow-heading">Legacy</text><text x="20" y="319" className="hb-flow-sub">По блокам модели</text><text x="20" y="340" className="hb-flow-sub">{wide?'C > 128':'C ≤ 128'}</text><text x="20" y="364" className="hb-flow-sub">суммы float64</text>
      <Kernel x={180} y={287} w={narrow?548:230} label={narrow?'Объединённое ядро одного блока':'Обход + сбор одного блока'}/>
      <Tree x={200} y={334} color="currentColor"/><Tree x={258} y={334} color="currentColor"/>
      {narrow?<><Arrow x1={330} y1={357} x2={368} y2={357}/><Matrix x={392} y={337} rows={3} cols={compact?2:5} label={compact?'Σ A[лист]':'Σ W[лист]'} local/>{compact&&<><text x="437" y="363" className="hb-flow-symbol">×</text><Matrix x={465} y={344} rows={2} cols={5} label="B" local/><text x="534" y="363" className="hb-flow-symbol">+</text><Matrix x={560} y={337} rows={3} cols={4} label="L" sparse local/></>}<Arrow x1={compact?616:474} y1={357} x2={708} y2={357}/></>:<><Arrow x1={414} y1={357} x2={446} y2={357} memory/><Matrix x={468} y={326} cols={compact?2:5} label={compact?'Z':'V'} sub={compact?'батч × r':'батч × C'}/>{compact&&<><Matrix x={532} y={326} cols={4} label="E" sparse/><text x="565" y="417" className="hb-flow-sub">батч × C</text></>}<Arrow x1={compact?590:535} y1={357} x2={619} y2={357} memory/><Kernel x={627} y={287} w={101} label={compact?'GEMM':'Копия'}/><text x="677" y="351" textAnchor="middle" className="hb-flow-symbol">{compact?'ZB+E':'V'}</text><text x="677" y="376" textAnchor="middle" className="hb-flow-sub">+ сумма</text></>}
      <Arrow x1={733} y1={357} x2={767} y2={357} memory/><Matrix x={786} y={326} cols={4} label="S" sub="батч × C"/>
      <Arrow x1={845} y1={357} x2={878} y2={357} memory/><Kernel x={886} y={287} w={207} label="CuPy · выход отдельно"/><Output x={1003} y={329}/>
      <path d="M841 429v13H187v-27m-5 6 5-6 5 6" className="hb-flow-repeat"/><text x="515" y="434" textAnchor="middle" className="hb-flow-sub">Повтор для блоков · K раз; у нескольких блоков есть рабочий S</text>
      <LaunchStrip y={464} count={7} label="… по блокам + операции выхода"/>
    </g>
    <g className="hb-flow-packed">
      <text x="20" y="539" className="hb-flow-heading">Packed</text><text x="20" y="561" className="hb-flow-sub">Все блоки вместе</text><text x="20" y="582" className="hb-flow-sub">Один запуск</text><text x="20" y="606" className="hb-flow-sub">суммы float64</text>
      <Kernel x={180} y={529} w={795} h={150} label="Одно CUDA-ядро · блок потоков на объект батча" packed/>
      <Tree x={198} y={582} color="currentColor"/><Tree x={257} y={582} color="currentColor"/>
      <g transform="translate(192,646)">{Array.from({length:12},(_,i)=><rect key={i} x={i*10} y="0" width="7" height="7" rx="1" fill="currentColor" opacity=".6"/>)}</g>
      <Arrow x1={330} y1={600} x2={366} y2={600}/>
      <Matrix x={392} y={580} rows={3} cols={compact?2:5} label={compact?'Σ A[лист]':'Σ W[лист]'} local/>
      {compact?<><text x="436" y="604" className="hb-flow-symbol">×</text><Matrix x={460} y={586} rows={2} cols={5} label="B" local/><Arrow x1={522} y1={600} x2={563} y2={600}/><text x="585" y="604" className="hb-flow-symbol">+</text><Matrix x={617} y={580} rows={3} cols={5} label="L" sparse local/><text x="541" y="658" textAnchor="middle" className="hb-flow-sub">декодирование + поправки</text></>:<Arrow x1={480} y1={600} x2={683} y2={600}/>}
      <Arrow x1={691} y1={600} x2={736} y2={600}/><path d="M752 618c20 0 12-39 31-39h9" fill="none" stroke="currentColor" strokeWidth="2"/><text x="776" y="652" textAnchor="middle" className="hb-flow-sub">вероятности</text><Output x={877} y={571}/>
      <Arrow x1={980} y1={600} x2={1017} y2={600} memory label="запись"/><Matrix x={1040} y={573} cols={4} rows={4} label="Выход" sub="батч × C"/>
      <text x="580" y="700" textAnchor="middle" className="hb-flow-sub">Промежуточные суммы: регистры + общая память CUDA-блока · глобального Z нет</text>
      <LaunchStrip y={706} count={1} packed label="все деревья + декодирование + выход"/>
    </g>
    <g transform="translate(180,757)"><rect x="0" y="-9" width="15" height="12" rx="2" className="hb-flow-memory" fill="currentColor"/><text x="23" y="1" className="hb-flow-sub">Глобальная память GPU</text><rect x="267" y="-9" width="15" height="12" rx="2" className="hb-flow-local" fill="currentColor"/><text x="290" y="1" className="hb-flow-sub">Регистры / общая память</text><rect x="580" y="-12" width="31" height="17" rx="4" className="hb-flow-legend-kernel"/><text x="620" y="1" className="hb-flow-sub">Граница CUDA-запуска</text></g>
  </svg></div>;
}
