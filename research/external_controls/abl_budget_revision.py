"""BR0: CPU diagnostics on saved MNIST/MediaMill; no tree fit or test access."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import gc, json, sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked, digest, write, event
from abl_quality_worker import check_native_environment, memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected, block_cost
from abl_training_core import calibrated_local_variants
from abl_increment_diagnostic_core import task_quality
from abl_stop_rule_replay import qvalue
from abl_stop_rule_core import quality_failures
from abl_budget_revision_core import (DenseView, decode_model, blank, pack, hessian,
    fixed_support_projection, prune_tree, reference_basis, bounded_allocate,
    support_order, marginal_price, maximum_capacity, label_f1, deficit, select_point)
from experiments.shared_leaf_basis import functional_basis, probabilities
from experiments.uniform_threshold_core import ThresholdModel
from experiments.sparse_leaf_core import leaf_design, training_weights, coordinate_scores, score_order

MEDIA_METHODS = ('M-SEP', 'M-F', 'M-B', 'M-S', 'M-BS', 'M-E64')


def pin(ref, locks, key='path'):
    p = ref[key]
    if digest(p) != ref['sha256']:
        raise RuntimeError('BR0 frozen source changed: '+str(p))
    locks[str(p)] = ref['sha256']
    return p


def source_case(reg, locks):
    r = reg['original_case']; pin(r, locks)
    return locked(r['path'], r['sha256'])


def role_data(source):
    data = data_before_test(source['record']); a, b = np.cumsum(data['sizes'])[:2]
    return data, dict(train=data['train_x'], calibration=data['dev_x'][:a],
        control=data['dev_x'][a:b], selection=data['dev_x'][b:]), dict(train=data['train_y'],
        calibration=data['dev_y'][:a], control=data['dev_y'][a:b], selection=data['dev_y'][b:])


def initial_model(reg, locks):
    r = reg['initial_snapshot']; pin(r, locks, 'artifact')
    return load_protected(r['artifact'])


def raw_roles(model, features):
    return {k:model.raw(x) for k, x in features.items()}


def store_candidate(model, raw, labels, task, uid, metadata, cfg, anchor, budget, out, fit_ref):
    variants = calibrated_local_variants(raw['calibration'], labels['calibration'],
        raw['control'], labels['control'], task, cfg['decision_grid'], cfg['decision_shrinkage'])
    rows = [dict(uid=f'v{i}', cost_bytes=block_cost(model)['cost_bytes'], control=v['validation'])
        for i, v in enumerate(variants)]
    chosen = select_point(rows, 'control', anchor['control'], 10**15)
    index = int(chosen['uid'][1:]); v = variants[index]
    cp = dict(temperature=float(v['temperature']), shifts=np.asarray(v['decision_shifts'], np.float32).tolist())
    model = model.with_shifts(np.asarray(cp['shifts'], np.float32)); price = block_cost(model)
    control = task_quality(labels['control'], raw['control'], task, cp['temperature'], cp['shifts'])
    selection = task_quality(labels['selection'], raw['selection'], task, cp['temperature'], cp['shifts'])
    out.mkdir(parents=True, exist_ok=True); file = out/'model.npz'; model.save(file)
    cache = out/'development.npz'; np.savez_compressed(cache, **{k:raw[k] for k in ('calibration', 'control', 'selection')})
    result = dict(uid=uid, **metadata, **price, control=control, selection=selection,
        control_guard=not quality_failures(qvalue(control), qvalue(anchor['control'])),
        selection_guard=not quality_failures(qvalue(selection), qvalue(anchor['selection'])),
        memory_guard=price['cost_bytes'] <= budget, policy=cp, selected_policy_index=index,
        policies=[dict(temperature=float(w['temperature']), shifts=np.asarray(w['decision_shifts'], np.float32).tolist(),
            control={k:float(w['validation'][k]) for k in ('loss', 'f1_macro', 'f1_weighted')}) for w in variants],
        raw_control_nll=task_quality(labels['control'], raw['control'], task)['loss'],
        artifact=str(file), sha256=digest(file), cache=dict(path=str(cache), sha256=digest(cache)), fit=fit_ref)
    write(out/'metadata.json', result)
    return result


def frozen_anchor(reg, source, features, labels, locks):
    r = reg['frozen_full_reference']; pin(r, locks, 'artifact')
    model = load_protected(r['artifact']); cp = r['policy']
    assert model.n_trees == r['trees'] and block_cost(model)['cost_bytes'] == r['cost_bytes']
    result = dict(uid=r['uid'], token=r['token'], policy=cp, artifact=r['artifact'], sha256=r['sha256'])
    for role in ('control', 'selection'):
        result[role] = task_quality(labels[role], model.raw(features[role]), source['task'], cp['temperature'], cp['shifts'])
    return result


def save_view(path, view, **extra):
    np.savez(path, nodes=view.nodes, node_offsets=view.node_offsets, leaf_offsets=view.leaf_offsets,
        values=view.values, bias=view.bias, **extra)
    return dict(path=str(path), sha256=digest(path))


def mnist_case(reg, cfg, out, locks):
    source = source_case(reg, locks); data, features, labels = role_data(source)
    cid = reg['case']; directory = out/'cases'/cid; directory.mkdir(parents=True, exist_ok=True)
    anchor = frozen_anchor(reg, source, features, labels, locks)
    warm = initial_model(reg, locks); view_w, _ = decode_model(warm)
    wire, contrast_basis = reference_basis(np.asarray(view_w.design(features['train'])@view_w.values))
    basis_file = directory/'warm_basis.npz'; np.savez(basis_file, wire=wire, contrast_basis=contrast_basis)
    rows = []; sources = []; maximum_roundtrip = 0.
    for trees in (160, 200):
        uid = f'full/b{trees//40-1:02d}'; meta = source['metas'][uid]; pin(meta, locks, 'artifact')
        original = load_protected(meta['artifact']); view, _ = decode_model(original)
        assert view.trees == trees and view.task == 'multiclass' and len(view.bias) == 10
        teacher_train = original.raw(features['train']); hh = hessian(teacher_train, 'multiclass')[:, :-1]
        visits = view.base().leaf_indices(features['train'])
        design = leaf_design(visits, len(view.values))
        atom = np.asarray(design.T@hh)+1.
        target = view.values[:, :-1]-view.values[:, -1:]
        prefix_dir = directory/f't{trees}'; prefix_dir.mkdir(exist_ok=True)
        src = save_view(prefix_dir/'source_geometry.npz', view, weights=atom, reference_target=target)
        sources.append(dict(trees=trees, source=dict(artifact=meta['artifact'], sha256=meta['sha256']), fitting=src))
        for rank in (4, 6, 8, 9):
            basis = np.eye(9) if rank == 9 else wire[:-1, :rank].astype(float)
            for cap in (4, 6, 8):
                pieces = []; coeff = []; no = [0]; lo = [0]; maps = []; orig_nodes = []; targets = []; weights = []; merges = []
                for t in range(trees):
                    a, b = view.leaf_offsets[t:t+2]; u, v = view.node_offsets[t:t+2]
                    p = prune_tree(view.nodes[u:v], target[a:b], atom[a:b], basis, cap)
                    pieces.append(p['nodes']); coeff.append(p['coefficients']); targets.append(p['target']); weights.append(p['weights'])
                    maps.append(p['mapping']+lo[-1]); orig_nodes.append(p['origin_nodes'])
                    no.append(no[-1]+len(p['nodes'])); lo.append(lo[-1]+len(p['coefficients'])); merges.append(p['merges'])
                nview = DenseView(np.concatenate(pieces), np.asarray(no, np.int64), np.asarray(lo, np.int64),
                    np.column_stack((np.concatenate(targets), np.zeros(lo[-1]))),
                    (view.bias-view.bias[-1]).astype(np.float32), view.features, view.task)
                if rank == 9:
                    base = nview.base(np.concatenate(coeff), representation='reference_class')
                else:
                    base = nview.base(np.concatenate(coeff), wire[:, :rank], 'low_rank')
                model = pack([blank(base)], nview.bias)
                raw = {k:model.raw(x) for k, x in features.items()}
                actual_visits = base.leaf_indices(features['train'])
                counts = np.bincount(actual_visits.ravel(), minlength=lo[-1])
                if counts.min() < 10:
                    raise RuntimeError('saved pruning lost minimum leaf support')
                if rank == 9 and cap == 8:
                    expected = original.raw(data['dev_x'])
                    got = np.concatenate([raw[k] for k in ('calibration', 'control', 'selection')])
                    err = float(np.max(np.abs(probabilities(expected, 'multiclass')-probabilities(got, 'multiclass'))))
                    maximum_roundtrip = max(maximum_roundtrip, err)
                    if err > 2e-5:
                        raise RuntimeError('full-rank unpruned probability roundtrip failed')
                uid = f't{trees}-l{cap}-r{rank}'; work = prefix_dir/uid; work.mkdir(exist_ok=True)
                f = work/'fit.npz'; np.savez(f, target=np.concatenate(targets), weights=np.concatenate(weights),
                    mapping=np.concatenate(maps), origin_nodes=np.concatenate(orig_nodes), original_weights=atom,
                    basis=np.eye(9) if rank == 9 else wire[:-1, :rank])
                fit = dict(path=str(f), sha256=digest(f), source_geometry=src, source_trees=trees)
                row = store_candidate(model, raw, labels, 'multiclass', uid,
                    dict(family=f'N-{rank}' if rank != 9 else 'N-F', rank=rank, leaf_cap=cap,
                        observed_minimum_leaf_support=int(counts.min()), source_trees=trees,
                        merges=sum(len(m) for m in merges), new_tree_fits=0), cfg, anchor,
                    reg['budget_bytes'], work, fit)
                rows.append(row)
                event('br0_representation', case=cid, uid=uid, bytes=row['cost_bytes'],
                    control_pass=row['control_guard'], selection_pass=row['selection_guard'], memory_pass=row['memory_guard'])
                del model, raw; gc.collect(); memory_guard()
    result = dict(case=cid, branch='BR-N', task='multiclass', original=reg['original_case'], budget_bytes=reg['budget_bytes'],
        registry=reg, anchor=anchor, rows=rows, source_geometry=sources,
        warm_basis=dict(path=str(basis_file), sha256=digest(basis_file)),
        maximum_full_rank_probability_error=maximum_roundtrip, new_tree_fits=0, test_accessed=False,
        fitting_scope='static atom-weighted reconstruction of saved leaves; no new split fitting and no training rollout')
    return finish_case(result, directory)


@dataclass
class Group:
    view: DenseView
    basis: np.ndarray | None
    weights: np.ndarray
    designs: dict
    frequency: np.ndarray
    coefficients: np.ndarray
    residual: np.ndarray
    order: np.ndarray
    offset: np.ndarray


def prepare_group(view, basis, features, labels, teacher, all_hessian):
    basis = None if basis is None else np.asarray(basis, np.float32)
    designs = {k:view.design(x) for k, x in features.items()}; d = np.asarray(designs['train'].T@all_hessian)+1.
    f = np.asarray(designs['train'].mean(0)).ravel()
    if basis is None:
        a = view.values.astype(np.float32); residual = np.zeros_like(view.values, np.float32); order = np.empty(0, np.int64)
    else:
        a, _, _, _ = fixed_support_projection(view.values, d, basis)
        residual = (view.values-a.astype(float)@basis.astype(float).T).astype(np.float32)
        offset = f@residual.astype(float)
        error = np.asarray(designs['train']@residual.astype(float))-offset
        w = training_weights(labels['train'], teacher, 'multilabel', 'balanced_mse', .5)
        order = score_order(coordinate_scores(designs['train'], residual, error, 'balanced_mse', w), residual)
    decoded = a.astype(float) if basis is None else a.astype(float)@basis.astype(float).T
    offset = f@(view.values-decoded)
    return Group(view, basis, d, designs, f, a, residual, order, offset)


def group_model(group, keys=()):
    if group.basis is None:
        if len(keys):
            raise ValueError('dense protected group cannot have residual')
        a = group.coefficients; rows = cols = np.empty(0, np.int32); values = np.empty(0, np.float32)
        decoded = a.astype(float)
    else:
        a, rows, cols, values = (group.coefficients, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)) if not len(keys) else fixed_support_projection(group.view.values, group.weights, group.basis, keys)
        decoded = a.astype(float)@group.basis.astype(float).T
        decoded[rows, cols] += values
    offset = group.frequency@(group.view.values-decoded)
    base = group.view.base(a, group.basis, 'dense' if group.basis is None else 'low_rank')
    return ThresholdModel(base, rows, cols, values, np.zeros(len(base.bias), np.float32)), offset


def assemble(groups, global_bias, supports):
    blocks = []; shifts = []
    for g, keys in zip(groups, supports):
        b, correction = group_model(g, keys); blocks.append(b); shifts.append(correction)
    model = pack(blocks, global_bias+np.sum(shifts, axis=0))
    raw = {k:np.broadcast_to(model.bias, (groups[0].designs[k].shape[0], len(global_bias))).astype(float, copy=True)
        for k in groups[0].designs}
    for g, b in zip(groups, blocks):
        for k in raw:
            raw[k] += b.raw_from_design(g.designs[k])
    return model, raw


def media_case(reg, cfg, out, locks, seal_cases):
    source = source_case(reg, locks); data, features, labels = role_data(source)
    cid = reg['case']; directory = out/'cases'/cid; directory.mkdir(parents=True, exist_ok=True)
    anchor = frozen_anchor(reg, source, features, labels, locks)
    old = next(c for c in seal_cases if c['case'] == cid)
    item = next(p for p in old['paths'] if p['path'] == 'functional')
    item = next(s for s in item['snapshots'] if s['trees'] == 160)
    pin(item, locks, 'artifact'); original = load_protected(item['artifact']); view, intervals = decode_model(original)
    assert intervals == ((0,40), (40,80), (80,120), (120,160))
    teacher = original.raw(features['train']); hh = hessian(teacher, 'multilabel')
    start = initial_model(reg, locks); initial, _ = decode_model(start)
    _, b0, _ = functional_basis(np.asarray(initial.design(features['train'])@initial.values), 'multilabel')
    archive = view.slice(0, 120); active = view.slice(120, 160)
    _, updated, _ = functional_basis(np.asarray(archive.design(features['train'])@archive.values), 'multilabel')
    _, ab, _ = functional_basis(np.asarray(active.design(features['train'])@active.values), 'multilabel')
    geometry = save_view(directory/'source_geometry.npz', view, source_global_bias=view.bias)
    basis_file = directory/'bases.npz'; np.savez(basis_file, initial=b0, updated=updated, active=ab)
    # Exact full-output recoding is only an audit control, not a budget candidate.
    exact = pack([blank(archive.base()), blank(active.base())], view.bias)
    exact_error = float(np.max(np.abs(exact.raw(data['dev_x'])-original.raw(data['dev_x']))))
    if exact_error > 2e-5:
        raise RuntimeError('exact coordinate recoding changed the saved function')
    input_policy = item['policy']; teacher_f1 = label_f1(labels['train'], teacher, input_policy['shifts'])
    input_quality = {role:task_quality(labels[role], original.raw(features[role]), 'multilabel',
        input_policy['temperature'], input_policy['shifts']) for role in ('control','selection')}
    rows = []; method_inputs = []
    # Reuse array geometry, designs and zero-support factorizations across controls.
    prepared = {}
    for method in MEDIA_METHODS:
        if method == 'M-SEP':
            specs = [(view.slice(a, b), None if i == 0 else original.blocks[i].base.basis)
                for i, (a, b) in enumerate(intervals)]
        else:
            basis = updated if method in ('M-B', 'M-BS', 'M-E64') else b0
            rank = 64 if method == 'M-E64' else 50
            specs = [(archive, basis[:, :rank]), (active, ab[:, :50])]
        groups = []
        for i, (v, b) in enumerate(specs):
            key = (tuple(v.leaf_offsets), None if b is None else b.tobytes(), v.values.tobytes())
            if key not in prepared:
                prepared[key] = prepare_group(v, b, features, labels, teacher, hh)
            groups.append(prepared[key])
        zero, zero_raw = assemble(groups, view.bias, [() for _ in groups]); price = block_cost(zero)
        deficits = np.maximum(0., teacher_f1-label_f1(labels['train'], zero_raw['train'], input_policy['shifts']))
        caps = np.asarray([len(g.order) for g in groups], np.int64)
        masses = np.asarray([g.view.values.size if g.basis is not None else 0 for g in groups])
        leaves = np.asarray([len(g.view.values) for g in groups])
        capacity = maximum_capacity(reg['budget_bytes'], price['coo_resident_bytes'], price['prepared_blocks_bytes'], leaves, masses, caps)
        old_count = sum((g.view.values.size+2*len(view.bias))//20 for g in groups if g.basis is not None)
        counts = dict(zero=0, old5=min(old_count, 0 if capacity is None else capacity), capacity=0 if capacity is None else capacity)
        work = directory/method; work.mkdir(exist_ok=True)
        group_refs = []
        for i, g in enumerate(groups):
            p = work/f'group{i}.npz'; np.savez(p, nodes=g.view.nodes, node_offsets=g.view.node_offsets,
                leaf_offsets=g.view.leaf_offsets, target=g.view.values, weights=g.weights,
                basis=np.empty((0,0)) if g.basis is None else g.basis, frequency=g.frequency,
                order=g.order, residual=g.residual, zero_coefficients=g.coefficients)
            group_refs.append(dict(path=str(p), sha256=digest(p)))
        method_inputs.append(dict(method=method, groups=group_refs, capacity=capacity, masses=masses.tolist(),
            caps=caps.tolist(), leaves=leaves.tolist(), zero_price=price, deficits=deficits.tolist(),
            teacher_f1=teacher_f1.tolist(), input_policy=input_policy,
            subspace_distance=None if method == 'M-SEP' else float(np.linalg.norm(
                specs[0][1]@specs[0][1].T-b0[:,:specs[0][1].shape[1]]@b0[:,:specs[0][1].shape[1]].T)),
            old5_parent_counts=[g.view.values.size+2*len(view.bias) if g.basis is not None else 0 for g in groups]))
        for tier, count in counts.items():
            variants = [(tier, None)]
            if tier == 'capacity' and method in ('M-F', 'M-BS') and count:
                variants += [(f'random-{seed}', seed) for seed in (701,702,703,704,705)]
            q = bounded_allocate(count, masses, caps)
            for variant, seed in variants:
                supports = []; support_info = []; rng = None if seed is None else np.random.default_rng(seed)
                for g, number in zip(groups, q):
                    if seed is not None:
                        keys = rng.choice(g.order, int(number), replace=False)
                        info = dict(active=False, randomized=True)
                    elif method in ('M-S', 'M-BS', 'M-E64'):
                        keys, info = support_order(g.order, len(view.bias), deficits, int(number))
                    else:
                        keys = g.order[:number].copy(); info = dict(active=False)
                    supports.append(keys); support_info.append(info)
                model, raw = assemble(groups, view.bias, supports)
                if count and block_cost(model)['cost_bytes'] > reg['budget_bytes']:
                    raise RuntimeError('actual sparse representation exceeded unchanged cap')
                target_dir = work/variant; target_dir.mkdir(exist_ok=True)
                fit_path = target_dir/'support.npz'; np.savez(fit_path,
                    **{f'g{i}':keys for i, keys in enumerate(supports)})
                fit = dict(path=str(fit_path), sha256=digest(fit_path), groups=group_refs, geometry=geometry)
                row = store_candidate(model, raw, labels, 'multilabel', method+'/'+variant,
                    dict(family=method, tier=tier, variant=variant, random_seed=seed,
                        requested_nnz=int(count), group_quotas=q.tolist(), support_info=support_info,
                        archive_rank=None if method == 'M-SEP' else (64 if method == 'M-E64' else 50),
                        new_tree_fits=0, residual_parent_counts=method_inputs[-1]['old5_parent_counts']),
                    cfg, anchor, reg['budget_bytes'], target_dir, fit)
                row['same_input_policy'] = {role:task_quality(labels[role],raw[role],'multilabel',
                    input_policy['temperature'],input_policy['shifts']) for role in ('control','selection')}
                row['input_preservation_guard'] = {role:not quality_failures(qvalue(row['same_input_policy'][role]),
                    qvalue(input_quality[role])) for role in ('control','selection')}
                row['actual_support_matches_quota'] = row['nnz']==count
                write(target_dir/'metadata.json',row)
                rows.append(row)
                event('br0_representation', case=cid, uid=row['uid'], bytes=row['cost_bytes'],
                    control_pass=row['control_guard'], selection_pass=row['selection_guard'], memory_pass=row['memory_guard'])
                del model, raw; gc.collect(); memory_guard()
    result = dict(case=cid, branch='BR-M', task='multilabel', registry=reg, original=reg['original_case'],
        budget_bytes=reg['budget_bytes'], anchor=anchor, rows=rows, method_inputs=method_inputs,
        input_snapshot=item, source_geometry=geometry, bases=dict(path=str(basis_file), sha256=digest(basis_file)),
        input_policy_quality=input_quality,
        exact_recode_raw_error=exact_error, exact_recode_cost=block_cost(exact),
        new_tree_fits=0, test_accessed=False, fitting_scope='static re-encoding of the same functional160; not a new optimization trajectory')
    return finish_case(result, directory)


def finish_case(result, directory):
    budget = result['budget_bytes']; anchor = result['anchor']; selected = []
    for family in sorted({r['family'] for r in result['rows']}):
        rows = [r for r in result['rows'] if r['family'] == family and r.get('random_seed') is None]
        primary = select_point(rows, 'selection', anchor['selection'], budget, True)
        diagnostic = select_point(rows, 'selection', anchor['selection'], budget, False)
        from abl_stop_rule_replay import choose_final
        legacy = choose_final([r for r in rows if r['cost_bytes'] <= budget and r['control_guard']])
        selected.append(dict(family=family, qualified=None if primary is None else primary['uid'],
            diagnostic=None if diagnostic is None else diagnostic['uid'],
            legacy_qualified=None if legacy is None else legacy['uid'],
            qualified_selection_pass=primary is not None and primary['selection_guard']))
    result['selections'] = selected
    result['summary'] = dict(representations=len(result['rows']),
        feasible=sum(r['memory_guard'] for r in result['rows']),
        control_pass=sum(r['memory_guard'] and r['control_guard'] for r in result['rows']),
        control_selection_pass=sum(r['memory_guard'] and r['control_guard'] and r['selection_guard'] for r in result['rows']),
        selected_family_passes=sum(s['qualified_selection_pass'] for s in selected))
    file = directory/'case.json'; write(file, result)
    event('br0_case_complete', case=result['case'], summary=result['summary'], path=str(file), sha256=digest(file))
    return dict(case=result['case'], branch=result['branch'], path=str(file), sha256=digest(file))


def run(cfg, out):
    started = perf_counter(); environment = check_native_environment(cfg)
    if cfg['abl_budget_revision'] != dict(stage='saved_model_diagnostic', maximum_representations=156,
            tree_training_allowed=False, gpu_execution=False, test_accessed=False, tolerance=.05):
        raise RuntimeError('outside BR0 authorization')
    registry = locked(Path(__file__).with_name('br_protocol_registry.json'), cfg['br_registry_sha256'])
    if registry['authorization'] != dict(stage='saved_model_diagnostic_only',new_trees=0,gpu_execution=False,
            test_access=False,user_confirmed=True,subagents=0):
        raise RuntimeError('BR0 registry authorization changed')
    for doc in registry['copied_protocols']:
        assert digest(Path(__file__).parent/doc['path'])==doc['sha256']
    assert registry['quality_tolerance'] == .05 and len(registry['cases']) == 6
    ref = cfg['abl_stop_rule_continuation_completed']; previous = locked(ref['path'], ref['sha256'])
    assert previous['status'] == 'bounded_native_continuation_complete'
    assert locked(previous['final_audit']['path'], previous['final_audit']['sha256'])['status'] == 'passed'
    seal_ref = previous['pretest_seal']; sealed = locked(seal_ref['path'], seal_ref['sha256'])
    locks = {ref['path']:ref['sha256'], seal_ref['path']:seal_ref['sha256']}; entries = []
    event('br0_start', contract=cfg['abl_budget_revision'], new_tree_fits=0, source=ref)
    for reg in sorted(registry['cases'], key=lambda r:(not r['case'].startswith('mnist'), r['case'])):
        entries.append(mnist_case(reg, cfg, out, locks) if reg['case'].startswith('mnist') else
            media_case(reg, cfg, out, locks, sealed['cases']))
        write(out/'progress.json', dict(cases=entries, new_tree_fits=0, test_accessed=False))
        gc.collect(); memory_guard()
    from audit_abl_budget_revision import audit
    checked = audit(entries, registry); audit_file = out/'audit_report.json'; write(audit_file, checked)
    for p, sha in locks.items():
        if digest(p) != sha:
            raise RuntimeError('frozen input changed during BR0')
    if any(k.split('.')[0] in ('cupy','torch','py_boost') for k in sys.modules):
        raise RuntimeError('BR0 imported a training/GPU runtime')
    cases = [locked(e['path'], e['sha256']) for e in entries]
    count = sum(len(c['rows']) for c in cases)
    assert count <= 156
    summary = dict(representations=count, cases=6, new_tree_fits=0,
        cases_with_control_selection_solution={branch:sum(c['summary']['selected_family_passes'] > 0 for c in cases if c['branch'] == branch)
            for branch in ('BR-N','BR-M')},
        method_selected_passes={family:sum(s['qualified_selection_pass'] for c in cases for s in c['selections'] if s['family'] == family)
            for family in sorted({s['family'] for c in cases for s in c['selections']})})
    result = dict(status='saved_model_budget_revision_complete', cases=entries,
        summary=summary, source=ref, immutable_inputs=locks, audit=dict(path=str(audit_file), sha256=digest(audit_file)),
        environment=environment, elapsed_seconds=perf_counter()-started, cpu_peak_rss_bytes=memory_guard(),
        new_tree_fits=0, gpu_execution=False, test_accessed=False,
        limitations=['static representations only, no new optimization trajectories',
            'selection is used for model choice, not independent validation',
            'full reference retains historical pooled calibration; previous research data reused',
            'no conclusion about 2of3 expansion gate without online and final test stages'])
    write(out/'report.json', result)
    event('br0_complete', path=str(out/'report.json'), sha256=digest(out/'report.json'), summary=summary)


def main():
    from threadpoolctl import threadpool_limits
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); out = Path(sys.argv[2])
    with threadpool_limits(limits=1):
        run(cfg, out)


if __name__ == '__main__':
    main()
