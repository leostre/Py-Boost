import numpy as np
from experiments.final_ensemble_core import Component,feasible_pairs,choose_mixture
from experiments.output_calibration_core import calibrated_quality


def test_pair_cost_includes_both_actual_layouts_new_weights_and_shifts():
    a=Component('a','a.npz','a',100,120,4)
    b=Component('b','b.npz','b',110,130,5)
    assert feasible_pairs([a,b],3,278)==[(a,b,278)]
    assert not feasible_pairs([a,b],3,277)
    assert not feasible_pairs([a,Component('other','other.npz','a',100,120,4)],3,10000)


def test_convex_scores_calibration_and_no_extra_test_capability():
    y=np.array([0,1,2,0,1,2],np.int32)
    raw=5*np.eye(3)[y]
    a=Component('a','a.npz','a',100,120,4)
    b=Component('b','b.npz','b',110,130,5)
    config=dict(decision_grid=[0],decision_shrinkage=[1])
    reference=calibrated_quality(y,raw,'multiclass',np.zeros(3))
    result=choose_mixture([a,b],{'a':raw,'b':.995*raw},y,{'a':raw,'b':.995*raw},y,
                           'multiclass',3,1000,reference,config)
    assert result['state']=='accepted' and result['candidate']['payload_bytes']==278
    assert all(r['validation']['f1_macro']==1 for r in result['candidates'])
    assert len(result['candidates'])==6
    empty=choose_mixture([a,b],{},y,{},y,'multiclass',3,277,reference,config)
    assert empty['candidate'] is None and not empty['candidates']
