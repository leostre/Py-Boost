import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {DATASETS,FAMILIES,pairModels,summarizePairs,selectedCurves,mean,median,sameFunctionExample,representationTimes} from './analytics.js';
const snapshot=JSON.parse(await readFile(new URL('../../data.json',import.meta.url),'utf8'));
const q=snapshot.queries,selected=q.selected_quality.rows;
const near=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
test('dataset and evidence cardinalities stay sealed',()=>{
  assert.equal(DATASETS.length,12);assert.equal(q.all_quality_points.rows.length,4941);assert.equal(selected.length,1008);
  assert.equal(q.executor_pairs.rows.length,520);assert.equal(q.bank_choices.rows.length,72);
  assert.equal(new Set(q.dataset_profiles.rows.map(r=>r.dataset)).size,12);
});
test('headline paired quality reconciles, with equal weight per dataset',()=>{
  for(const [fam,expected] of [['direct_AB',0.7679015625],['direct_ABL',0.6823713442]]){
    const pairs=pairModels(selected,fam,'ordinary',.25),sums=summarizePairs(pairs);
    assert.equal(pairs.length,36);assert.ok(pairs.every(r=>r.complete));
    assert.equal(sums.filter(r=>r.delta>0).length,10);near(mean(sums.map(r=>r.delta)),expected,1e-8);
    assert.ok(pairs.every(r=>r.A.cost_bytes<=r.A.limit_bytes&&r.B.cost_bytes<=r.B.limit_bytes));
  }
});
test('missing models stay missing, never zero filled; reversal preserves pairs',()=>{
  const a=pairModels(selected,'posthoc','ordinary',.25),b=pairModels(selected,'ordinary','posthoc',.25);
  assert.equal(a.filter(r=>r.complete).length,30);assert.equal(a.filter(r=>!r.complete).length,6);
  for(let i=0;i<a.length;i++){assert.equal(a[i].case,b[i].case);if(a[i].complete)near(a[i].delta,-b[i].delta);else{assert.equal(a[i].delta,null);assert.equal(a[i].uid_A,null);}}
  const identical=pairModels(selected,'direct_AB','direct_AB',.7);assert.ok(identical.every(r=>r.delta===0));
});
test('metric orientation favors lower loss; cohort and dataset filters restrict source',()=>{
  const f=pairModels(selected,'direct_AB','ordinary',.7,'loss','new');assert.equal(f.length,12);
  for(const r of f)near(r.delta,r.value_B-r.value_A);
  assert.equal(pairModels(selected,'direct_AB','ordinary',.7,'f1_macro','new','birds').length,0);
  assert.equal(pairModels(selected,'direct_AB','ordinary',.7,'f1_macro','all','birds').length,3);
});
test('curves retain explicit empty budget cells',()=>{
  const curves=selectedCurves(selected.filter(r=>r.dataset==='delicious'),'f1_macro');
  assert.equal(curves.length,FAMILIES.length*4);
  assert.ok(curves.every(r=>r.value===null?r.splits===0:r.splits>0));
});
test('inference headlines are medians of paired session ratios',()=>{
  for(const [batch,expected] of [[1,1.93771693],[1024,1.27260378]]){
    const rs=q.executor_pairs.rows.filter(r=>r.batch===batch&&r.roles.includes('direct_ABL200'));assert.equal(rs.length,12);
    near(median(rs.map(r=>r.request_p95_speedup)),expected,1e-8);
    rs.forEach(r=>near(r.request_p95_speedup,median(r.request_session_ratios)));
  }
});
test('resident bank per-task bytes and F1 reconcile to measured aggregate',()=>{
  for(const bank of ['full200','ordinary70','mixed2470'])for(const backend of ['legacy','packed']){
    const tasks=q.bank_choices.rows.filter(r=>r.bank===bank&&r.backend===backend),s=q.bank_summary.rows.filter(r=>r.bank===bank&&r.backend===backend);
    assert.equal(tasks.length,12);assert.equal(s.length,8);
    for(const r of s){assert.equal(tasks.reduce((n,t)=>n+t.device_array_bytes,0),r.device_array_bytes);near(mean(tasks.map(t=>t.f1_macro)),r.mean_task_f1);}
  }
});
test('constant diagnostic deltas use the full-minus-constant orientation',()=>{
  for(const r of q.constant_diagnostics.rows){near(r.delta_full_pp,100*(r.full200_f1-r.f1_macro));near(r.delta_ordinary100_pp,100*(r.ordinary100_f1-r.f1_macro));}
});
test('the executor example compares one UID and the same recorded session',()=>{
  const pairs=q.executor_pairs.rows.filter(r=>r.batch===1024&&r.roles.includes('direct_ABL200'));
  const example=sameFunctionExample(pairs,q.inference_summary.rows);assert.ok(example);
  near(example.ratio,pairs[0].request_session_ratios[0]);
  assert.equal(example.uid,pairs[0].uid);
  assert.equal(sameFunctionExample(pairs,[]),null);
});
test('representation baseline is explicit, packed and paired by case/batch',()=>{
  for(const reference of ['full200','ordinary70']){
    const joined=representationTimes(q.representation_comparison.rows,q.inference_summary.rows,reference);
    assert.equal(joined.length,336);assert.ok(joined.every(r=>r.uid_B&&r.request_p95_A>0&&r.request_p95_B>0));
    for(const r of joined){
      const b=q.inference_summary.rows.find(t=>t.case===r.case&&t.batch===r.batch&&t.backend==='packed'&&t.roles.includes(reference));
      assert.equal(r.uid_B,b.uid);assert.equal(r.request_p95_B,b.request_p95_ms);
      near(r.speed,reference==='full200'?r.request_vs_full_speedup:r.request_vs_small_ordinary_speedup);
      near(r.speed,r.request_p95_B/r.request_p95_A);
    }
  }
  assert.ok(representationTimes(q.representation_comparison.rows,[],'full200').every(r=>r.uid_B===null));
});
