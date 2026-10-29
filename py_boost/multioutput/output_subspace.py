"""Pure output-policy calculations for online HASBoost spectral tracking."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OutputSubspaceSummary:
    """Small serializable summary of one output-subspace update."""

    spike_count: int
    effective_rank: float
    spike_energy_fraction: float
    relative_spike_margin: float
    loading_profiles: Tuple[Tuple[float, ...], ...]
    probabilities: Tuple[float, ...]
    significant_component_indices: Tuple[int, ...]


def entropy_effective_rank(eigenvalues: Sequence[float]) -> float:
    """Return entropy effective rank for a non-negative spectrum."""

    values = tuple(max(0.0, float(value)) for value in eigenvalues)
    total = sum(values)
    if total <= 0:
        return 0.0
    probabilities = tuple(
        probability
        for value in values
        if value > 0
        for probability in (value / total,)
        if probability > 0
    )
    return exp(-sum(value * log(value) for value in probabilities))


def spectral_output_policy(
    eigenvalues: Sequence[float],
    eigenvectors: Sequence[Sequence[float]],
    null_p95: Sequence[float],
    beta: float = 0.9,
    tau: float = 1e-6,
) -> OutputSubspaceSummary:
    """Convert excess spectral energy into a uniform-protected distribution."""

    if not 0 <= beta <= 1:
        raise ValueError('beta must satisfy 0 <= beta <= 1')
    if tau < 0:
        raise ValueError('tau must be non-negative')
    values = tuple(float(value) for value in eigenvalues)
    thresholds = tuple(float(value) for value in null_p95)
    vectors = tuple(tuple(float(value) for value in row) for row in eigenvectors)
    if len(values) != len(thresholds):
        raise ValueError('observed and null spectra must have the same length')
    if not vectors or any(len(row) != len(values) for row in vectors):
        raise ValueError('eigenvectors must have shape outputs by components')

    spikes = tuple(value > threshold for value, threshold in zip(values, thresholds))
    excess = tuple(max(0.0, value - threshold) for value, threshold in zip(values, thresholds))
    importance = [tau for _ in vectors]
    for output_index, row in enumerate(vectors):
        importance[output_index] += sum(
            row[component] ** 2 * excess[component]
            for component, selected in enumerate(spikes)
            if selected
        )
    total = sum(importance)
    if total <= 0:
        importance = [1.0 for _ in vectors]
        total = float(len(vectors))
    dimension = len(vectors)
    probabilities = tuple(
        (1 - beta) / dimension + beta * value / total
        for value in importance
    )
    spike_indices = tuple(
        component for component, selected in enumerate(spikes) if selected
    )
    loading_profiles = tuple(
        tuple(
            row[component] * sqrt(excess[component])
            for component in spike_indices
        )
        for row in vectors
    )
    observed_energy = sum(max(0.0, value) for value in values)
    spike_energy_fraction = (
        sum(excess) / observed_energy if observed_energy > 0 else 0.0
    )
    relative_spike_margin = max(
        (
            excess[component] / max(thresholds[component], tau, 1e-12)
            for component in spike_indices
        ),
        default=0.0,
    )
    return OutputSubspaceSummary(
        spike_count=sum(spikes),
        effective_rank=entropy_effective_rank(values),
        spike_energy_fraction=spike_energy_fraction,
        relative_spike_margin=relative_spike_margin,
        loading_profiles=loading_profiles,
        probabilities=probabilities,
        significant_component_indices=spike_indices,
    )


def diversity_aware_topk(
    probabilities: Sequence[float],
    loading_profiles: Sequence[Sequence[float]],
    budget: int,
    diversity_weight: float = 0.25,
) -> Tuple[int, ...]:
    """Greedily select relevant outputs while suppressing spectral duplicates."""

    relevance = tuple(float(value) for value in probabilities)
    profiles = tuple(tuple(float(value) for value in row) for row in loading_profiles)
    if not relevance:
        raise ValueError("probabilities must not be empty")
    if len(profiles) != len(relevance):
        raise ValueError("loading profiles must align with probabilities")
    widths = {len(row) for row in profiles}
    if len(widths) != 1:
        raise ValueError("loading profiles must form a rectangular matrix")
    if not 1 <= budget <= len(relevance):
        raise ValueError("budget must be within the output dimension")
    if not 0 <= diversity_weight <= 1:
        raise ValueError("diversity_weight must satisfy 0 <= value <= 1")
    if any(value < 0 for value in relevance):
        raise ValueError("probabilities must be non-negative")

    maximum = max(relevance)
    scaled = tuple(value / maximum if maximum > 0 else 1.0 for value in relevance)
    profile_matrix = np.asarray(profiles, dtype=float)
    norms = np.linalg.norm(profile_matrix, axis=1)
    normalized = np.divide(
        profile_matrix,
        norms[:, None],
        out=np.zeros_like(profile_matrix),
        where=norms[:, None] > 0,
    )
    similarities = np.abs(normalized @ normalized.T)

    selected = []
    remaining = set(range(len(relevance)))
    while len(selected) < budget:
        def score(index: int):
            redundancy = (
                float(np.max(similarities[index, selected])) if selected else 0.0
            )
            value = (
                (1 - diversity_weight) * scaled[index]
                - diversity_weight * redundancy
            )
            return value, relevance[index], -index

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)
