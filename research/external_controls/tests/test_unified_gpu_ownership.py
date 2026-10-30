from unified_gpu_ownership import probe_candidate


def event(size,matching=(101,),moving=False):
    before={101:500*1024**2,202:1024**3};during={pid:value+(size if pid in matching else 0) for pid,value in before.items()};after=before.copy()
    if moving:after[101]+=size
    return dict(bytes=size,before=before,during=during,after=after)


def test_two_unique_reversible_markers_required_and_foreign_ambiguity_rejected():
    sizes=(64*1024**2,96*1024**2)
    assert probe_candidate([event(s) for s in sizes])==101
    assert probe_candidate([event(s,matching=(101,202)) for s in sizes]) is None
    assert probe_candidate([event(s,moving=True) for s in sizes]) is None
    assert probe_candidate([event(sizes[0])]) is None
