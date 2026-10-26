"""Checks of prespecified calibration selection and paired pilot decisions."""

from experiments.revision_comparison import blocked_case_order, select_calibration, comparison_gate


def test_block_order_varies_without_dropping_any_method():
    cases = ('full', 'fixed', 'energy', 'risk')
    orders = [blocked_case_order(cases, block) for block in range(3)]
    assert len(set(orders)) == 3
    assert all(set(order) == set(cases) for order in orders)


def test_calibration_never_selects_fast_but_quality_ineligible_candidate():
    shared = dict(dataset='test', model_seed=104, split_seed=1140)
    records = [dict(shared, case_id='full', f1_weighted=.8, train_seconds=10.),
               dict(shared, case_id='fixed_50', f1_weighted=.7, train_seconds=3., sketch_outputs=50),
               dict(shared, case_id='fixed_75', f1_weighted=.8, train_seconds=7., sketch_outputs=75),
               dict(shared, case_id='energy_0.80', f1_weighted=.7, train_seconds=2., output_energy_fraction=.8),
               dict(shared, case_id='energy_0.95', f1_weighted=.799, train_seconds=6., output_energy_fraction=.95)]
    decision = select_calibration(records, .002)['test']
    assert decision['fixed_budget'] == 75
    assert decision['energy_fraction'] == .95


def test_regulator_must_beat_both_simple_competitors():
    shared = dict(dataset='test', seed=17, f1_weighted=.8)
    rows = [dict(shared, case_id='fixed', train_seconds=10.),
            dict(shared, case_id='energy', train_seconds=5.),
            dict(shared, case_id='risk', train_seconds=7.)]
    assert comparison_gate(rows, .002, .10)['continue_risk_controller'] is False
