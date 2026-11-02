"""Exact shared-vector tree export; no approximation or output-wise tree copies."""
from pathlib import Path
import hashlib,json,sys,gc
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(ref):
    assert sha(ref['path'])==ref['sha256'];return json.loads(Path(ref['path']).read_text())

def export_full(model):
    from treelite.model_builder import ModelBuilder,Metadata,TreeAnnotation,PostProcessorFunc
    c=len(model.bias);trees=model.n_trees
    builder=ModelBuilder(threshold_type='float32',leaf_output_type='float32',
        metadata=Metadata(num_feature=model.n_features,task_type='kMultiClf',average_tree_output=False,num_target=1,num_class=[c],leaf_vector_shape=(1,c)),
        tree_annotation=TreeAnnotation(num_tree=trees,target_id=[0]*trees,class_id=[-1]*trees),postprocessor=PostProcessorFunc(name='identity_multiclass'),base_scores=model.bias.tolist())
    for block in model.blocks:
        b=block.base
        if b.representation!='dense' or len(block.values):raise ValueError('This control exports full saved leaves only')
        for t in range(b.n_trees):
            nodes=b.nodes[b.node_offsets[t]:b.node_offsets[t+1]];values=b.coefficients[b.leaf_offsets[t]:b.leaf_offsets[t+1]];n=len(nodes)
            key=lambda v:int(v) if v>=0 else n-int(v)-1
            builder.start_tree()
            for i,row in enumerate(nodes):
                f=int(row[0]);builder.start_node(i)
                builder.numerical_test(abs(f)-1,float(row[1]),default_left=f<0,opname='<=',left_child_key=key(row[2]),right_child_key=key(row[3]));builder.end_node()
            for i,value in enumerate(values):builder.start_node(n+i);builder.leaf(value.tolist());builder.end_node()
            builder.end_tree()
    return builder.commit()

def cases(cfg):
    source=Path(cfg['research_source']);sys.path.insert(0,str(source));sys.path.insert(0,str(source/'native_source'))
    from experiments.block_leaf_model import BlockLeafModel
    seal=load(cfg['tuning'])
    for e in seal['entries']:
        if 'full200' not in e['roles']:continue
        m=e['model'];assert sha(m['artifact'])==m['sha256'];model=BlockLeafModel.load(m['artifact']);cc=e['record']['case'];dd=e['record']['dataset']
        assert sha(cc['split_file'])==cc['split_sha256'] and sha(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=np.concatenate([f[k] for k in ('calibration','control','selection')])
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32)
        yield e,model,x

def run(cfg):
    import cupy as cp,nvforest,treelite
    from scipy.special import expit
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True);rows=[]
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    print(json.dumps(dict(event='N1_protocol',baseline='full200 saved shared vector leaves',formats=['single','double'],source=cfg['tuning'],new_trees=0)),flush=True)
    for e,model,x in cases(cfg):
        small=x[:min(37,len(x))].copy();nan=x[:1].copy();nan[0,0]=np.nan;sample=np.concatenate((small,nan));expected=model.raw(sample)
        original=export_full(model);original.serialize(out/(e['case']+'.tl'))
        z=treelite.gtil.predict(original,sample,pred_margin=True).reshape(expected.shape)
        np.testing.assert_allclose(z,expected,atol=5e-5,rtol=1e-5)
        results=[]
        for precision in ('single','double'):
            handle=nvforest.Handle(stream=cp.cuda.get_current_stream().ptr)
            forest=nvforest.load_from_treelite_model(original,device='gpu',precision=precision,handle=handle)
            actual=cp.asnumpy(forest.predict_proba(cp.asarray(sample)));actual=actual.reshape(expected.shape)
            error=float(np.max(np.abs(actual-expected)));pe=float(np.max(np.abs(expit(actual)-expit(expected))))
            assert error<5e-5 and pe<2e-5,(e['dataset'],precision,error,pe)
            shifts=np.asarray(e['model']['assessment']['refit_control']['shifts']);diff=(actual>=shifts)!=(expected>=shifts)
            assert not np.any(diff & (np.abs(expected-shifts)>5e-5))
            results.append(dict(precision=precision,raw_error=error,probability_error=pe,decision_differences=int(diff.sum()),near_threshold_differences=int((diff & (np.abs(expected-shifts)<=5e-5)).sum()),outputs=int(forest.num_outputs),trees=int(forest.num_trees)))
            del forest,handle;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        row=dict(case=e['case'],dataset=e['dataset'],model_sha256=e['model']['sha256'],treelite_raw_error=float(np.max(np.abs(z-expected))),checks=results)
        rows.append(row);(out/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2));print(json.dumps(dict(event='N1_export',**row)),flush=True)
        del model,x,original;gc.collect()
    result=dict(status='nvforest_export_verified',rows=rows,new_trees=0,new_timing_queries=0,postprocessor='identity_multiclass; vector positions encode independent raw logits, not a softmax classification objective',
        note='Predict_proba returns identity raw vector here; common sigmoid and frozen thresholds are external. No duplicated tree per output.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='N1_complete',models=len(rows),new_trees=0,report_sha256=sha(out/'report.json'))),flush=True)

if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
