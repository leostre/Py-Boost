"""Pure specification of the size-matching regime; legacy defaults are unchanged."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BudgetRegime:
    matched: bool
    reference_trees: int
    controller_horizon: int

    @property
    def stop_on_quality(self):
        return not self.matched

    def cap(self, reference_parameters, outputs, fraction):
        if (reference_parameters < outputs or outputs < 1
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError('invalid reference budget')
        raw = int(fraction * reference_parameters)
        return outputs * (raw // outputs) if self.matched else raw

    def endpoint(self, parameters, cap, outputs, stop_reason):
        if outputs < 1 or parameters < outputs or parameters > cap:
            raise ValueError('invalid realized endpoint')
        remaining = cap - parameters
        return dict(matched=stop_reason == 'cap' and 0 <= remaining < 2 * outputs,
                    remaining_parameters=remaining, matching_tolerance=2 * outputs)


def budget_regime(config):
    mode = config.get('study_mode', 'quality_stop')
    if mode not in ('quality_stop', 'matched_parameters'):
        raise ValueError('unknown budget study mode')
    matched = mode == 'matched_parameters'
    reference = config.get('reference_trees', config['ntrees'])
    horizon = config['ntrees']
    if (type(reference) is not int or type(horizon) is not int
            or reference < 100 or horizon < reference):
        raise ValueError('invalid reference/controller horizon')
    if matched and (reference != 200 or horizon != 500 or config['parameter_fraction'] != .75):
        raise ValueError('matched regime must follow preregistered 200/500/75% contract')
    return BudgetRegime(matched, reference, horizon)
