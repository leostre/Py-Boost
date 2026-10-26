# H-B3S: frozen Birds final-selection audit (2026-10-21)

User-approved Sol/xhigh, no agents, no tree retraining, relative quality tolerance 5%.
The Russian preregistration in the Hasboost vault is the full protocol.

- Seeds 2204/2214/2224; strict periodic_fixed and joint_adaptive trajectories.
- Load only confirmed checkpoints in their existing final candidate lists.
- Copy weights and previously recorded threshold vectors, never refit anything.
- Full reference, role partition, 75% COO bytes AND stored coefficients unchanged.
- 2x2 metadata-only lists: all checkpoints / latest two; all existing threshold
  fractions / fraction 1 only. Maximum sizes 20/8/5/2, never quality-pruned.
- Point selector: original three quality gates, then macro/loss/bytes/coefficients/id.
- Paired90: original point gates AND >=90% of 500 paired bootstrap draws passing
  all three relative quality gates. Same preference order. Failure is not success.
- Seeds for bootstrap = outer_seed+700000; internal halves = outer_seed+800000.
- All pools share draws; 100 split-half repetitions in both directions, and exact
  leave-one-out on the original selection role only. No role mixing/stratification.
- Retain all 19 labels for multilabel macro-F1, including zero-support labels.
- Recompute frozen rows numerically, then replay their original precision for
  full-sample choices (avoid changing a winner through platform one-ULP loss noise).
- First node: development only, seal choices/inputs. Child: verify seal and replay
  decisions before exploratory reused-test scoring. Print all 48 decision metrics.
- Diagnose BOTH false acceptance and refusal; also available-candidate coverage.
  Test-oracle coverage is diagnostic, never a selector. No post-test rule changes.
- No new datasets, GPU work, environment changes or new training; CPU one thread.
- Stop after this bounded check, deadline 2026-10-21T14:00Z. No broad follow-ups.
