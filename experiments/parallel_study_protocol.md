# H-PAR first fixed comparison

Reuse first40 actual compact OPT1hybrid trees, not the full200 model. Four
cycles of two independent20tree continuations, average increments with prefix
counted once. Four variants: serial40lr.1; serial40lr.05; parallel2x20lr.1
without recompression; same with shared-basis sparse5% recompression eachcycle.
All new fits rows/features.8, samefirstcyclebranch0seed asserialfull-step,
whose first20 native values/topology must match exactly.

Union PCA of centered training functions F+Delta1,F+Delta2; rankhalf, contrast
for multiclass. Decode old blocks transiently, leaf weights training visits+1,
weighted projection/support/refit as OPT1. Preserve training prediction mean by
one global bias. Log train loss before/after: no monotonicity guarantee.
Recompression affects subsequent real gradients. Store all checkpoints/jobs.

Two persistent spawn workers with independent CUDA RNG; use simultaneous jobs
only with>=8192MiB free, otherwise samebranches sequentially with reason logged.
Each worker>=2048MiB free. This algorithm comparison is not a controlled claim
of hardware speedup. Record fit interval overlap, wall time, worker sum time,
merge time and individual ownpoolpeaks; summed peaks are NOT process/context
memory or time-aligned combined peak. No foreign workloads/environment changes.

3datasets5oldsplits4newvariants=60fulltrajectories,9600newtrees,360nativejobs.
Full/fixed/higher-rank OPT1 controls reused. Sameold threshold folds, choices
allsealedbeforetests, same5%quality and75%actualcost guards, no testsselection.
Historical tests exploratory. Fixedcommand/env,5400sec cap, no parameter search.
