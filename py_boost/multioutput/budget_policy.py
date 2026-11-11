"""Simple, spectrum-free output-budget rules for controlled comparisons."""

import numpy as np


def gradient_energy_budget(energies, fraction=0.90, min_budget=1, xp=np):
    """Smallest prefix covering a specified fraction of squared-gradient mass."""
    if not 0 < fraction <= 1:
        raise ValueError('energy fraction must satisfy 0 < value <= 1')
    if energies.ndim != 1 or not 1 <= min_budget <= energies.size:
        raise ValueError('energy vector and minimum budget must be valid')
    if bool(xp.any(~xp.isfinite(energies))) or bool(xp.any(energies < 0)):
        raise ValueError('energies must be finite and non-negative')
    if fraction == 1 or float(energies.sum()) <= 0:
        return int(energies.size)
    ordered = xp.sort(energies)[::-1]
    budget = int(xp.searchsorted(xp.cumsum(ordered), fraction * ordered.sum(), side='left')) + 1
    return min(int(energies.size), max(int(min_budget), budget))
