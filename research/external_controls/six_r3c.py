"""R3: disentangle structural removal, leaf refit, and AB+L compression."""
from pathlib import Path
import gc, json, sys, time
import numpy as np
from budget_quality_worker import write, digest, event
from abl_quality_worker import check_native_environment
from six_common import load_cases, load_model, store_model, test_case, gpu_raw
from six_structural_core import keep_trees, importance_gpu, refit_gpu
from broad_core import model_prefix
from abl_transfer_study import data_before_test
from unified_training_core import coarsen
from adaptive_controller_core import candidates, Action


def run(cfg, out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    out.mkdir(parents=True, exist_ok=True); env = check_native_environment(cfg)
    _, cases = load_cases(cfg, first=False); refs = []; started = time.perf_counter()
    first=set();remaining=[]
    for case in cases:
        name=case['record']['case']['dataset']
        if name in first:remaining.append(case)
        first.add(name)
    cases=remaining
    event('R3_protocol', datasets=len(cases), new_trees=0, four_D_checkpoints_per_arm=True,
          method='independent LOP-inspired tree removal and fixed-leaf refit, not exact LOP reproduction')
    for case in cases:
        data = data_before_test(case['record']); directory = out/case['case']; directory.mkdir(exist_ok=True)
        full = load_model(next(m for m in case['models'] if m['uid']=='ordinary/d3/t200'))
        teacher = gpu_raw(full, data['train_x']); scores, _, objective = importance_gpu(full, data['train_x'], data['train_y'])
        order = np.lexsort((np.arange(len(scores)), -scores)); rank = min(64, int(np.ceil(len(full.bias)/2)))
        models = []; diagnostics = []
        for count in (50, 100, 150, 200):
            removed = keep_trees(full, np.sort(order[:count])); prefix = model_prefix(full, count)
            refitted, trace = refit_gpu(removed, data['train_x'], data['train_y'], teacher)
            coarse = coarsen(removed, data['train_x'], teacher, 4)
            coarse, coarse_trace = refit_gpu(coarse, data['train_x'], data['train_y'], teacher)
            variants = [('prefix', prefix), ('remove', removed), ('refit', refitted)]
            actions = [Action(rank, 8, 0.), Action(rank, 8, .02)]
            for candidate in candidates(None, refitted, {'train':data['train_x']}, {'train':data['train_y']}, actions, 'risk_residual', max_basis_rank=rank):
                variants.append(('refit_ABL' if candidate['residual_fraction'] else 'refit_AB', candidate['model']))
            candidate = next(candidates(None, coarse, {'train':data['train_x']}, {'train':data['train_y']},
                                       [Action(rank, 8, .02)], 'risk_residual', max_basis_rank=rank))
            variants.append(('coarse_refit_ABL', candidate['model']))
            for family, model in variants:
                models.append(store_model(model, f'{family}/t{count}', family, data, cfg, case,
                              directory/family/str(count), rank=rank, kept_trees=np.sort(order[:count]).tolist()))
            diagnostics.append(dict(trees=count, refit=trace, coarse_refit=coarse_trace))
            event('R3_checkpoint', case=case['case'], trees=count, models=len(models))
        item = dict(case=case['case'], record=case['record'], models=models, diagnostics=diagnostics,
                    pruning_scores=scores.tolist(), train_original_loss=objective, new_trees=0)
        path = directory/'development.json'; write(path, item); refs.append(dict(case=case['case'],path=str(path),sha256=digest(path)))
        write(out/'progress.json', dict(cases=refs, test_opened=False)); del full, teacher, data
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    seal=out/'pretest_seal.json'; write(seal, dict(cases=refs, new_trees=0, test_opened=False))
    event('R3_seal', sha256=digest(seal)); results=[]
    from budget_quality_worker import locked
    for ref in refs:
        case=locked(ref['path'],ref['sha256']); rows=test_case(case,case['models'],out/case['case']/'test.json')
        results.append(dict(case=case['case'],models=rows)); event('R3_test',case=case['case'],models=len(rows))
    report=dict(status='R3_complete',cases=refs,results=results,new_trees=0,environment=env,
                pretest_seal=dict(path=str(seal),sha256=digest(seal)),seconds=time.perf_counter()-started)
    write(out/'report.json',report); event('R3_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
