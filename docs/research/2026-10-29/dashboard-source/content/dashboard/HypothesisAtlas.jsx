import React from 'react';
import {DataComponent,useDataApp} from '../../data-app-public.jsx';
export function HypothesisAtlas({q}) {
  const {sectionViews,setSectionView}=useDataApp();
  const rows=q.hypothesis_atlas.rows;
  const key=sectionViews['hb-hypothesis-atlas']?.idea??'basis';
  const selected=rows.find(r=>r.id===key)??rows[0];
  return <DataComponent id="hypothesis-atlas" queryId="hypothesis_atlas" title="Основные идеи: механизм, проверка и ограничение" kind="diagram" sourceRows={rows} displayRows={[selected]}>
    <p className="hb-atlas-intro">Выберите идею. Рисунки из прежних обсуждений объясняют механизм; подпись показывает, что установили эксперименты к 2 ноября.</p>
    <div className="hb-atlas-nav" role="group" aria-label="Основные гипотезы HASBoost">{rows.map(r=><button key={r.id} aria-pressed={selected.id===r.id} onClick={()=>setSectionView('hb-hypothesis-atlas',{idea:r.id})}><strong>{r.title}</strong><span>{r.status}</span></button>)}</div>
    <figure className="hb-atlas-figure"><img src={selected.image_data_url} alt={selected.alt}/><figcaption><b>{selected.title}.</b> {selected.caption}</figcaption></figure>
    <p className="hb-atlas-boundary"><b>Граница:</b> {selected.boundary}</p>
    <a className="hb-atlas-history" href="https://github.com/leostre/Py-Boost/blob/hasboost/20261030/six-direction-study/docs/research/methodology/README.md" target="_blank" rel="noreferrer">История постановок, неудачных веток и результатов →</a>
  </DataComponent>;
}
