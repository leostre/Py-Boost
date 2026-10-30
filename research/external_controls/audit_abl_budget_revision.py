"""Independent stored-array, topology, metric and physical-storage BR0 audit."""
import numpy as np
from scipy.special import expit, softmax
from budget_quality_worker import locked, digest
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from audit_abl_protected import physical_cost
from audit_abl_transfer import independent_task_quality
from audit_abl_stop_rule import passed
from experiments.sparse_leaf_core import leaf_design


def independent_hessian(raw, task):
    p = (np.clip(softmax(raw, axis=1), 1e-6, 1-1e-6) if task == 'multiclass'
        else np.clip(expit(raw), 1e-7, 1-1e-7))
    return p*(1-p)*(p.shape[1]/(p.shape[1]-1) if task == 'multiclass' else 1.)


def metric_equal(a, b):
    for k in ('loss','f1_macro','f1_weighted'):
        assert abs(a[k]-b[k]) < 2e-8, (k,a[k],b[k])


def independent_deficit(q, r):
    return max(0., q['loss']/(1.05*r['loss'])-1,
        1-q['f1_macro']/(.95*r['f1_macro']), 1-q['f1_weighted']/(.95*r['f1_weighted']))


def independent_choice(rows, role, reference, budget, qualified=False):
    a = [r for r in rows if r['cost_bytes'] <= budget and (not qualified or r['control_guard'])]
    if not a:
        return None
    best = min(independent_deficit(r[role], reference) for r in a)
    a = [r for r in a if independent_deficit(r[role], reference)-best <= 1e-12]
    if role == 'selection':
        price = min(r['cost_bytes'] for r in a); a = [r for r in a if r['cost_bytes'] == price]
    loss = min(r[role]['loss'] for r in a); a = [r for r in a if r[role]['loss']-loss <= 1e-6]
    return min(a, key=(lambda r:(r['cost_bytes'], r.get('leaves',0), r['uid'])) if role == 'control' else
        (lambda r:(-r[role]['f1_macro'], r['cost_bytes'], r['uid'])))


def stored_arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def check_support_solution(target, weights, basis, support, block):
    a = block.base.coefficients.astype(float)
    free = weights.copy(); free.ravel()[support] = 0
    residual = (free*(a@basis.T-target))@basis
    scale = np.maximum(1., np.max(np.abs((free*target)@basis), axis=1))
    assert np.max(np.max(np.abs(residual), axis=1)/scale) < 6e-5
    # Independent weighted lstsq on a deterministic sample, equations on every row.
    ids = set(np.linspace(0, len(target)-1, min(16,len(target)), dtype=int).tolist())
    if len(support):
        per = np.bincount(support//target.shape[1], minlength=len(target)); ids.add(int(per.argmax()))
    maximum = 0.
    for i in sorted(ids):
        root = np.sqrt(free[i]); lhs = root[:,None]*basis; rhs = root*target[i]
        expected = np.linalg.lstsq(lhs, rhs, rcond=1e-6)[0]
        error = float(np.max(np.abs(expected-a[i]))); maximum = max(maximum,error)
        np.testing.assert_allclose(a[i], expected, atol=3e-5, rtol=4e-5)
    actual = block.rows.astype(np.int64)*target.shape[1]+block.columns
    assert set(actual).issubset(set(support))
    expected_values = (target-a@basis.T)[block.rows,block.columns].astype(np.float32)
    np.testing.assert_array_equal(block.values, expected_values)
    return maximum


def topology_mapping(original_nodes, new_nodes, origin_nodes):
    mapping = {}; retained = []
    def leaves(pointer):
        if pointer < 0:
            return [-pointer-1]
        return leaves(int(original_nodes[pointer,2]))+leaves(int(original_nodes[pointer,3]))
    def visit(old, new):
        if new < 0:
            for ell in leaves(old):
                assert ell not in mapping; mapping[ell] = -new-1
            return
        assert old >= 0 and int(origin_nodes[new]) == old
        np.testing.assert_array_equal(original_nodes[old,:2], new_nodes[new,:2]); retained.append(old)
        for side in (2,3):
            visit(int(original_nodes[old,side]), int(new_nodes[new,side]))
    visit(0,0)
    assert len(retained) == len(new_nodes)
    return np.array([mapping[j] for j in range(len(mapping))], np.int64)


def audit_mnist(row, case, source_geometry, features, labels, hcache):
    fit = row['fit']; assert digest(fit['path']) == fit['sha256']
    a = stored_arrays(fit['path']); s = source_geometry[fit['source_trees']]
    m = load_protected(row['artifact']); block = m.blocks[0]; base = block.base
    origin = s['arrays']; src = s['model']; source_base = s['flat_base']
    if row['rank'] == 9:
        assert base.representation == 'reference_class' and base.coefficients.shape[1] == 9 and not base.basis.size
        basis = np.eye(9)
    else:
        basis = base.basis[:-1].astype(float)
        np.testing.assert_array_equal(base.basis[-1], 0)
    np.testing.assert_allclose(basis, a['basis'], atol=2e-7)
    original_target = origin['reference_target']; original_weights = origin['weights']
    expected_target = []; expected_weight = []
    for t in range(base.n_trees):
        no, nn = source_base.node_offsets[t:t+2]; q, z = base.node_offsets[t:t+2]
        lo, hi = source_base.leaf_offsets[t:t+2]; x, y = base.leaf_offsets[t:t+2]
        local = topology_mapping(source_base.nodes[no:nn], base.nodes[q:z], a['origin_nodes'][q:z])
        np.testing.assert_array_equal(a['mapping'][lo:hi], local+x)
        assert y-x <= row['leaf_cap']
        for ell in range(y-x):
            which = np.flatnonzero(local == ell)
            ww = original_weights[lo:hi][which]; tt = original_target[lo:hi][which]
            total = ww.sum(0); expected_weight.append(total); expected_target.append((ww*tt).sum(0)/total)
    expected_target = np.asarray(expected_target); expected_weight = np.asarray(expected_weight)
    np.testing.assert_allclose(a['target'], expected_target, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a['weights'], expected_weight, atol=2e-11, rtol=2e-12)
    decoded = base.coefficients.astype(float) if row['rank'] == 9 else base.coefficients.astype(float)@basis.T
    normal = (expected_weight*(decoded-expected_target))@basis
    rhs = (expected_weight*expected_target)@basis
    assert np.max(np.abs(normal)/(1+np.abs(rhs))) < 8e-5
    # These positive-D systems have full column rank; check independent lstsq.
    for i in np.linspace(0,len(decoded)-1,min(16,len(decoded)),dtype=int):
        root = np.sqrt(expected_weight[i]); expected = np.linalg.lstsq(root[:,None]*basis, root*expected_target[i], rcond=None)[0]
        np.testing.assert_allclose(base.coefficients[i], expected, atol=3e-6, rtol=2e-5)
    visits = base.leaf_indices(features['train'])
    support = np.bincount(visits.ravel(), minlength=len(base.coefficients))
    assert support.min() >= 10 and support.min() == row['observed_minimum_leaf_support']
    np.testing.assert_array_equal(m.bias, (src.bias-src.bias[-1]).astype(np.float32))


def audit_media_fit(row, case, group_cache, features, hcache):
    fit = row['fit']; assert digest(fit['path']) == fit['sha256']; support = stored_arrays(fit['path'])
    model = load_protected(row['artifact']); corrections = []; maximum = 0.
    assert len(model.blocks) == len(fit['groups'])
    for i,(ref,b) in enumerate(zip(fit['groups'],model.blocks)):
        assert digest(ref['path']) == ref['sha256']
        if ref['path'] not in group_cache:
            g = stored_arrays(ref['path'])
            # Derivatives and frequencies use actual source160 predictions, never T/shifts.
            visits = b.base.leaf_indices(features['train']); design = leaf_design(visits,len(g['target']))
            np.testing.assert_allclose(g['weights'], np.asarray(design.T@hcache.hessian)+1., atol=2e-11, rtol=2e-12)
            np.testing.assert_allclose(g['frequency'], np.asarray(design.mean(0)).ravel(), atol=1e-14)
            g['_train_design'] = design
            group_cache[ref['path']] = g
        g = group_cache[ref['path']]; keys = support[f'g{i}']
        for name in ('nodes','node_offsets','leaf_offsets'):
            np.testing.assert_array_equal(getattr(b.base,name),g[name])
        if g['basis'].size:
            np.testing.assert_array_equal(b.base.basis, g['basis'].astype(np.float32))
            maximum = max(maximum, check_support_solution(g['target'],g['weights'],g['basis'],keys,b))
            decoded = b.base.coefficients.astype(float)@b.base.basis.astype(float).T
            decoded[b.rows,b.columns] += b.values
        else:
            assert len(keys) == 0 and not len(b.values)
            np.testing.assert_array_equal(b.base.coefficients,g['target'].astype(np.float32)); decoded=b.base.coefficients.astype(float)
        corrections.append(g['frequency']@(g['target']-decoded))
        assert len(keys) == row['group_quotas'][i]
        if row['random_seed'] is None and row['family'] not in ('M-S','M-BS','M-E64'):
            np.testing.assert_array_equal(keys, g['order'][:len(keys)])
        assert len(set(keys)) == len(keys) and set(keys).issubset(set(g['order']))
    np.testing.assert_array_equal(model.bias, (hcache.bias+np.sum(corrections,axis=0)).astype(np.float32))
    return maximum


class HessianContext:
    """ndarray protocol is avoided: explicit fields prevent mixing scores/bias."""
    def __init__(self, hessian, bias):
        self.hessian = hessian; self.bias = bias


def f1_labels(y, z, shifts):
    pred=z-np.asarray(shifts)>=0; denom=y.sum(0)+pred.sum(0)
    return np.divide(2*(pred*y).sum(0),denom,out=np.zeros(y.shape[1]),where=denom>0)


def check_media_diagnostics(case, cache, y, teacher):
    for item in case['method_inputs']:
        zero_row=next(r for r in case['rows'] if r['family']==item['method'] and r['variant']=='zero')
        z=load_protected(zero_row['artifact']).raw(case['_train_x'])
        expected=np.maximum(0.,f1_labels(y,teacher,item['input_policy']['shifts'])-f1_labels(y,z,item['input_policy']['shifts']))
        np.testing.assert_allclose(expected,item['deficits'],atol=1e-14)
        # Check both physical envelope branches directly from retained arrays.
        geom=[]
        for ref in item['groups']:
            g=cache[ref['path']]
            geom.append(sum(g[k].nbytes for k in ('nodes','node_offsets','leaf_offsets','zero_coefficients'))+g['basis'].astype(np.float32).nbytes)
        c=y.shape[1]
        assert item['zero_price']['coo_resident_bytes']==sum(geom)+12*c+4
        assert item['zero_price']['prepared_blocks_bytes']==sum(geom)+8*c*len(geom)+8*c+4
        for row in (r for r in case['rows'] if r['family']==item['method']):
            logical=stored_arrays(row['fit']['path']); rng=None if row['random_seed'] is None else np.random.default_rng(row['random_seed'])
            for i,ref in enumerate(item['groups']):
                g=cache[ref['path']]; keys=logical[f'g{i}']; number=row['group_quotas'][i]
                if rng is not None:
                    np.testing.assert_array_equal(keys,rng.choice(g['order'],int(number),replace=False))
                elif row['family'] in ('M-S','M-BS','M-E64'):
                    info=row['support_info'][i]
                    np.testing.assert_array_equal(np.bincount(keys%c,minlength=c),info['quotas'])
                    assert info['active']==bool(number and expected.sum()>0)
                    if not info['active']:
                        np.testing.assert_array_equal(keys,g['order'][:number])
                    else:
                        # Within each output, supported coordinates must be the
                        # first entries of the old balanced-MSE order.
                        for j in range(c):
                            actual=set(keys[keys%c==j]); order=g['order'][g['order']%c==j]
                            assert actual==set(order[:len(actual)])


def audit(entries, registry):
    checks = []; total = 0
    for entry in entries:
        case = locked(entry['path'], entry['sha256']); reg = next(r for r in registry['cases'] if r['case']==case['case'])
        assert case['registry']==reg and case['budget_bytes']==reg['budget_bytes']
        source = locked(case['original']['path'],case['original']['sha256']); data = data_before_test(source['record'])
        nc, nt, _ = data['sizes']; features=dict(train=data['train_x'], development=data['dev_x'])
        labels=dict(train=data['train_y'],calibration=data['dev_y'][:nc],control=data['dev_y'][nc:nc+nt],selection=data['dev_y'][nc+nt:])
        task=case['task']; anchor=case['anchor']; fr=reg['frozen_full_reference']
        assert anchor['uid']==fr['uid'] and anchor['sha256']==fr['sha256'] and anchor['policy']==fr['policy']
        full=load_protected(fr['artifact']); assert digest(fr['artifact'])==fr['sha256']
        z=full.raw(data['dev_x'])
        for role,xx in (('control',slice(nc,nc+nt)),('selection',slice(nc+nt,None))):
            metric_equal(independent_task_quality(labels[role],z[xx],fr['policy'],task),anchor[role])
        source_geometry={}; group_cache={}; max_lstsq=0.; hcontext=None
        if task=='multiclass':
            for src in case['source_geometry']:
                f=src['fitting']; assert digest(f['path'])==f['sha256']; a=stored_arrays(f['path'])
                m=load_protected(src['source']['artifact']); assert digest(src['source']['artifact'])==src['source']['sha256']
                from abl_budget_revision_core import decode_model
                view,_=decode_model(m); base=view.base()
                p=independent_hessian(m.raw(data['train_x']),task)[:,:-1]
                d=leaf_design(base.leaf_indices(data['train_x']),len(view.values))
                np.testing.assert_allclose(a['weights'],np.asarray(d.T@p)+1.,atol=2e-11,rtol=2e-12)
                np.testing.assert_array_equal(a['reference_target'],view.values[:,:-1]-view.values[:,-1:])
                source_geometry[src['trees']]=dict(arrays=a,model=view,flat_base=base)
        else:
            ip=case['input_snapshot']; assert digest(ip['artifact'])==ip['sha256']; source_model=load_protected(ip['artifact'])
            hh=independent_hessian(source_model.raw(data['train_x']),task)
            hcontext=HessianContext(hh,source_model.bias)
            case['_source_bias']=source_model.bias
            case['_train_x']=data['train_x']
            source_train=source_model.raw(data['train_x'])
            input_dev=source_model.raw(data['dev_x'])
            for role,sl in (('control',slice(nc,nc+nt)),('selection',slice(nc+nt,None))):
                metric_equal(independent_task_quality(labels[role],input_dev[sl],ip['policy'],task),case['input_policy_quality'][role])
        for row in case['rows']:
            assert row['new_tree_fits']==0 and digest(row['artifact'])==row['sha256']
            model=load_protected(row['artifact']); assert physical_cost(model)==row['cost_bytes']
            assert row['memory_guard']==(row['cost_bytes']<=reg['budget_bytes'])
            assert digest(row['cache']['path'])==row['cache']['sha256']; cache=stored_arrays(row['cache']['path'])
            raw=model.raw(data['dev_x']); cuts={'calibration':slice(0,nc),'control':slice(nc,nc+nt),'selection':slice(nc+nt,None)}
            for role,sl in cuts.items():
                np.testing.assert_allclose(raw[sl],cache[role],atol=2e-8,rtol=1e-11)
            for role in ('control','selection'):
                q=independent_task_quality(labels[role],cache[role],row['policy'],task)
                metric_equal(q,row[role]); assert row[role+'_guard']==passed(q,anchor[role])
                if task=='multilabel':
                    same=independent_task_quality(labels[role],cache[role],case['input_snapshot']['policy'],task)
                    metric_equal(same,row['same_input_policy'][role])
                    assert row['input_preservation_guard'][role]==passed(same,case['input_policy_quality'][role])
            for p in row['policies']:
                metric_equal(independent_task_quality(labels['control'],cache['control'],p,task),p['control'])
            alternatives=[dict(uid=f'v{i}',cost_bytes=row['cost_bytes'],control=p['control']) for i,p in enumerate(row['policies'])]
            expected=independent_choice(alternatives,'control',anchor['control'],10**15)
            assert expected['uid']==f"v{row['selected_policy_index']}"
            np.testing.assert_array_equal(model.decision_shifts,row['policy']['shifts'])
            if task=='multiclass':
                audit_mnist(row,case,source_geometry,features,labels,None)
            else:
                max_lstsq=max(max_lstsq,audit_media_fit(row,case,group_cache,features,hcontext))
        if task=='multilabel':
            check_media_diagnostics(case,group_cache,labels['train'],source_train)
        for choice in case['selections']:
            rows=[r for r in case['rows'] if r['family']==choice['family'] and r.get('random_seed') is None]
            for key,qualified in (('qualified',True),('diagnostic',False)):
                q=independent_choice(rows,'selection',anchor['selection'],reg['budget_bytes'],qualified)
                assert choice[key]==(None if q is None else q['uid'])
        assert not case['test_accessed'] and case['new_tree_fits']==0
        total+=len(case['rows']); checks.append(dict(case=case['case'],representations=len(case['rows']),status='passed',
            maximum_sampled_lstsq_difference=max_lstsq))
    assert len(entries)==len(registry['cases']) and total<=156
    return dict(status='passed',cases=checks,representations=total,new_tree_fits=0,test_accessed=False,gpu_execution=False,
        scope='independent sigmoid/softmax metrics, exact full anchors and byte costs, saved predictions, '
              'all pruning topology maps and merged atomic targets, fixed-support normal equations on all leaves, '
              'independent weighted lstsq on up to16 leaves per group, control-policy and final selection decisions')
