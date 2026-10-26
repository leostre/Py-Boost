import pytest
from hypothesis import given, strategies as st
from experiments.controlled_study import transfer_reference_settings


def row(**updates):
    result = dict(dataset='birds', seed=604, method='full_reference',
                  is_full_reference=True, boosting_steps=200, parameters=1000,
                  lambda_l2=10., candidate_id='birds/604/full_reference/t200')
    result.update(updates)
    return result


@given(st.dictionaries(st.text(max_size=10), st.floats(allow_nan=True)))
def test_scores_cannot_affect_calibration(arbitrary_scores):
    expected = transfer_reference_settings([row()], 'birds', 604)
    assert transfer_reference_settings([row(test=arbitrary_scores, validation=arbitrary_scores)], 'birds', 604) == expected


def test_final_model_selected_by_step_not_quality_or_size():
    records = [row(boosting_steps=150, parameters=750, candidate_id='prefix'),
               row(), row(method='history', parameters=10), row(seed=614, parameters=9999)]
    assert transfer_reference_settings(records, 'birds', 604)['parameters'] == 1000


@pytest.mark.parametrize('records', [[], [row(is_full_reference=False)],
                                     [row(boosting_steps=150)], [row(parameters=0)],
                                     [row(parameters=1000.5)]])
def test_invalid_reference_is_rejected(records):
    with pytest.raises(ValueError):
        transfer_reference_settings(records, 'birds', 604)
