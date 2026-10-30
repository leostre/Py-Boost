// Pure projections of the sealed, reviewed experiment snapshot. No test-time selection.
export const DATASETS = ['emotions','scene','yeast','birds','medical','genbase','bibtex','mediamill','enron','cal500','corel5k','delicious'];
export const NEW_DATASETS = new Set(['enron','cal500','corel5k','delicious']);
export const FAMILIES = ['ordinary','direct_AB','direct_ABL','posthoc','controller','sketch','mixed24'];
export const LABELS = {ordinary:'Обычный бустинг',direct_AB:'AB · прямое обучение',direct_ABL:'AB + L · прямое обучение',posthoc:'Сжатие после обучения',controller:'Регулятор бюджета',sketch:'Sketch',mixed24:'Смешанный отбор'};
export const COLORS = {ordinary:'#687b96',direct_AB:'#008d9b',direct_ABL:'#c28514',posthoc:'#8d6cb6',controller:'#3c63a2',sketch:'#b9676d',mixed24:'#3e996c'};
export const METRICS = {f1_macro:'macro-F1',f1_weighted:'Взвешенная F1',loss:'NLL',brier:'Brier'};
export const ROLES = {full200:'Полная · 200 деревьев',direct_AB200:'AB · 200 деревьев',direct_ABL200:'AB + L · 200 деревьев',ordinary25:'Обычная · 0,25 B0',ordinary50:'Обычная · 0,50 B0',ordinary70:'Обычная · 0,70 B0',ordinary100:'Обычная · 1,00 B0',direct_AB70:'AB · 0,70 B0',direct_ABL70:'AB + L · 0,70 B0',posthoc70:'Сжатие после обучения · 0,70 B0',controller70:'Регулятор · 0,70 B0',mixed2425:'Смешанная · 0,25 B0',mixed2450:'Смешанная · 0,50 B0',mixed2470:'Смешанная · 0,70 B0',mixed24100:'Смешанная · 1,00 B0'};
export const BANKS = {full200:'Полные модели',ordinary70:'Обычные · 0,70 B0',mixed2470:'Смешанные · 0,70 B0'};
export const finite = v => typeof v === 'number' && Number.isFinite(v);
export const mean = a => {const b=a.filter(finite);return b.length?b.reduce((s,v)=>s+v,0)/b.length:null;};
export const median = a => {const b=a.filter(finite).sort((x,y)=>x-y);return b.length?(b[Math.floor((b.length-1)/2)]+b[Math.ceil((b.length-1)/2)])/2:null;};
export const groupBy = (a,key) => a.reduce((g,r)=>{const k=typeof key==='function'?key(r):r[key];(g[k]??=[]).push(r);return g;},{});
export const withinCohort = (r,c) => c==='all'||(c==='new'?NEW_DATASETS.has(r.dataset):!NEW_DATASETS.has(r.dataset));
export const valid = (r,m='f1_macro') => r.uid!=null && finite(r[m]);
export const metricDelta = (a,b,m) => (m.startsWith('f1_')?100: -1)*(a-b);
export const deltaUnit = m => m.startsWith('f1_')?'п.п.':'ед.';
export const deltaDefinition = m => m.startsWith('f1_')?'Δ = 100 × (A − B), в процентных пунктах. Положительное значение — в пользу A.':'Δ = B − A. Положительное значение — в пользу A, поскольку меньшее значение ошибки лучше.';
export function pairModels(rows,a,b,ratio,metric='f1_macro',cohort='all',dataset='all') {
  const scoped=rows.filter(r=>r.ratio===Number(ratio)&&withinCohort(r,cohort)&&(dataset==='all'||r.dataset===dataset));
  const aa=new Map(scoped.filter(r=>r.family===a).map(r=>[r.case,r]));
  const bb=new Map(scoped.filter(r=>r.family===b).map(r=>[r.case,r]));
  return [...new Set([...aa.keys(),...bb.keys()])].sort().map(k=>{const x=aa.get(k),y=bb.get(k),ok=x&&y&&valid(x,metric)&&valid(y,metric);return {case:k,dataset:(x||y).dataset,seed:(x||y).seed,complete:!!ok,A:x||null,B:y||null,value_A:valid(x||{},metric)?x[metric]:null,value_B:valid(y||{},metric)?y[metric]:null,delta:ok?metricDelta(x[metric],y[metric],metric):null,bytes_A:x?.cost_bytes??null,bytes_B:y?.cost_bytes??null,uid_A:x?.uid??null,uid_B:y?.uid??null};});
}
export function summarizePairs(pairs) {
  return Object.entries(groupBy(pairs.filter(r=>r.complete),'dataset')).map(([dataset,rs])=>{const ds=rs.map(r=>r.delta);return {dataset,delta:mean(ds),lo:Math.min(...ds),hi:Math.max(...ds),pairs:rs.length,bytes_A:mean(rs.map(r=>r.bytes_A)),bytes_B:mean(rs.map(r=>r.bytes_B))};}).sort((a,b)=>b.delta-a.delta);
}
export function selectedCurves(rows,metric) {
  return Object.values(groupBy(rows,r=>`${r.family}|${r.ratio}`)).map(rs=>{const vs=rs.filter(r=>valid(r,metric)).map(r=>r[metric]);return {family:LABELS[rs[0].family],ratio:rs[0].ratio,value:mean(vs),min:vs.length?Math.min(...vs):null,max:vs.length?Math.max(...vs):null,splits:vs.length};});
}
export function exportCSV(rows,filename) {
  if(!rows.length)return;
  const keys=Object.keys(rows[0]).filter(k=>!['A','B'].includes(k));
  const esc=v=>'"'+String(v==null?'':typeof v==='object'?JSON.stringify(v):v).replaceAll('"','""')+'"';
  const blob=new Blob(['\ufeff'+[keys.map(esc).join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\r\n')],{type:'text/csv;charset=utf-8'});
  const href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
}
