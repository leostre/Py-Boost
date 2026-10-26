from test_cumulative_budget_projection import run_synthetic_shell
from experiments import final_ensemble_study as ensemble


def test_synthetic_same_role_shell_seals_before_any_test_scores(monkeypatch,tmp_path):
    study,cfg,parent_report,parent_output,calls=run_synthetic_shell(monkeypatch,tmp_path)
    study.cumulative_training(cfg,parent_report,parent_output)
    path=parent_output/'cumulative_development_manifest.json'
    cfg.update(cumulative_development_manifest=str(path),cumulative_development_manifest_sha256=study.sha256(path))
    calls.clear()
    out=tmp_path/'ensemble'
    out.mkdir()
    original=ensemble.calibrated_quality
    observed=[]
    def spy(y,*args,**kwargs):
        if len(y)==32:
            assert (out/'ensemble_selection_seal.json').exists()
            observed.append(True)
        return original(y,*args,**kwargs)
    monkeypatch.setattr(ensemble,'calibrated_quality',spy)
    report={}
    ensemble.final_ensemble_study(cfg,report,out)
    assert observed and not calls and report['new_fits']==0
    assert report['results'][0]['state']=='no_distinct_memory_feasible_pair'
    assert not report['serving_measured']
