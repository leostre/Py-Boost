import React from 'react';
import {DataComponent} from '../../data-app-public.jsx';

const engines={native:'Исходный Py-Boost',legacy:'Legacy',packed:'Packed'};
export function InferenceJourney({q,engine,onEngineChange,compact,wide,onCompactChange,onWideChange}) {
  const effectiveCompact=engine!=='native'&&compact;
  const variant=`${engine}/${effectiveCompact?'compact':'full'}/${wide?'wide':'narrow'}`;
  const frames=React.useMemo(()=>(q.inference_animation?.rows??[]).filter(r=>r.variant===variant),[q.inference_animation,variant]);
  const [position,setPosition]=React.useState(0),[playing,setPlaying]=React.useState(false),[rate,setRate]=React.useState(1),[loop,setLoop]=React.useState(true),[reduced,setReduced]=React.useState(false);
  React.useEffect(()=>{const media=window.matchMedia('(prefers-reduced-motion: reduce)');const update=()=>setReduced(media.matches);update();media.addEventListener('change',update);return()=>media.removeEventListener('change',update);},[]);
  React.useEffect(()=>{setPosition(0);setPlaying(false);},[variant]);
  React.useEffect(()=>{
    if(!playing||!frames.length)return;
    const timeout=window.setTimeout(()=>{
      if(position+1<frames.length)setPosition(position+1);else if(loop)setPosition(0);else setPlaying(false);
    },1200/rate);
    return()=>window.clearTimeout(timeout);
  },[playing,position,rate,loop,frames.length]);
  if(!frames.length)return null;
  const index=Math.min(position,frames.length-1),frame=frames[index];
  const svg=(playing&&!reduced?frame.animated_svg:frame.svg).replaceAll('dur="1.2s"',`dur="${1.2/rate}s"`);
  const image=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  function seek(next){setPlaying(false);setPosition(Math.max(0,Math.min(frames.length-1,next)));}
  return <DataComponent id="inference-journey" queryId="inference_animation" title="Путь батча: от входа до готового предсказания" kind="diagram" sourceRows={frames} displayRows={[frame]}>
    <div className="hb-journey-engine" role="group" aria-label="Исполнитель анимации">{Object.entries(engines).map(([id,label])=><button key={id} aria-pressed={engine===id} onClick={()=>onEngineChange(id)}>{label}</button>)}</div>
    <div className="hb-journey-options"><span>Листья:</span><button aria-pressed={!effectiveCompact} onClick={()=>onCompactChange(false)}>Полные W</button><button disabled={engine==='native'} aria-pressed={effectiveCompact} onClick={()=>onCompactChange(true)}>AB + L</button>{engine==='legacy'&&<><span>Путь legacy:</span><button aria-pressed={!wide} onClick={()=>onWideChange(false)}>C ≤ 128</button><button aria-pressed={wide} onClick={()=>onWideChange(true)}>C &gt; 128</button></>}</div>
    <div className="hb-journey-controls"><button className="hb-journey-play" onClick={()=>{if(!playing&&index===frames.length-1)setPosition(0);setPlaying(!playing);}}>{playing?'Пауза':'Воспроизвести'}</button><button onClick={()=>seek(0)}>Сначала</button><button aria-label="Предыдущий этап" disabled={index===0} onClick={()=>seek(index-1)}>← Шаг</button><button aria-label="Следующий этап" disabled={index===frames.length-1} onClick={()=>seek(index+1)}>Шаг →</button><label>Темп <select aria-label="Темп анимации" value={rate} onChange={e=>setRate(Number(e.target.value))}><option value={.5}>0,5×</option><option value={1}>1×</option><option value={2}>2×</option></select></label><label><input type="checkbox" checked={loop} onChange={e=>setLoop(e.target.checked)}/> Повторять</label></div>
    <div className="hb-journey-scrub"><input type="range" min="0" max={frames.length-1} value={index} onChange={e=>seek(Number(e.target.value))} aria-label="Этап анимации" aria-valuetext={frame.title}/><span data-reviewed-rows>Этап {index+1} / {frames.length}</span></div>
    <div className="hb-journey-scene" tabIndex="0" aria-label="Схема полного запроса; на узком экране прокручивается внутри карточки"><img src={image} alt={`${engines[engine]}. ${frame.title}. ${frame.detail}`} width="1320" height="760"/></div>
    <div className="hb-journey-caption" aria-live={playing?'off':'polite'}><strong data-reviewed-rows>{frame.title}</strong><p data-reviewed-rows>{frame.detail}</p></div>
    <p className="hb-schematic-note">Учебный пример многометочной классификации: два объекта; показаны 3 выхода из C = {frame.outputs}. Числа одинаковы для всех наборов бенчмарка и не являются результатами эксперимента. В этом примере W = AB + L задано точно; реальное сжатие обычно приближённое. Темп и длительность не показывают аппаратное ускорение. Пауза фиксирует этап.{reduced?' Движение маркеров отключено согласно настройке уменьшения анимации.':''}</p>
  </DataComponent>;
}
