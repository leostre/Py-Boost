# H6 numerical and synthetic training gate

Approved executor: Sol / xhigh, no subagents. Fixed OpenResearch command and
environment are unchanged. This node answers correctness only, not usefulness.

Run property and example tests for the curvature-scaled signal, explicit EMA
first/second moments, positive-semidefinite temporal variance identity, projector
overlap, coordinate-safe multibasis serialization, and frozen-prefix equality.
Synthetic GPU tests cover multilabel and multiclass training, independent
training-state replay, a no-refresh control and repeated fit state reset.
Retain the existing compact training regression suite. No scientific dataset
fits, threshold selection, or test quality claims are allowed at this gate.

Use shared A100 only when at least 2048 MiB is free, observe capacity, do not
alter the environment or foreign workloads. Bound this gate by 900 seconds.
Failure handling and immutable-node rules remain those of the parent project.
The separately saved H6 protocol defines the subsequent scientific study.
