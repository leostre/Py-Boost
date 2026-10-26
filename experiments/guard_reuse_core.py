"""Sufficient, not necessary, proof of strict/soft trajectory equivalence."""
from experiments.shared_leaf_basis import quality_guard
from experiments.uniform_threshold_core import choice_key


def all_pass_equivalent(record):
    """No failure means neither soft policy changes choice, state, or RNG use.

    This is exact trace reuse, not a claim of independent replication. It needs
    a completed strict trajectory and unchanged data/algorithm configuration;
    the caller locks that manifest and the auditor checks the artifact arrays.
    """
    cycles = record.get('cycles', [])
    if len(cycles) != 5 or record.get('new_built_trees') != 200:
        return False
    for block, cycle in enumerate(cycles):
        if cycle.get('block') != block or cycle.get('state') != 'accepted':
            return False
        projection = cycle.get('projection', {})
        chosen = projection.get('chosen')
        if not chosen or not quality_guard(chosen['validation'], projection['proposal_control']):
            return False
        eligible = [r for r in projection['candidates'] if quality_guard(r['validation'], projection['proposal_control'])]
        if not eligible or min(eligible, key=choice_key) != chosen:
            return False
        if (cycle.get('accepted_trees') != (block+1)*40 or
            cycle['payload_bytes'] > record['budget']['payload_bytes'] or
            cycle['parameters'] > record['budget']['parameters']):
            return False
    return True
