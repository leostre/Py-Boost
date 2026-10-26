# H-OPT1: training feedback versus fixed-basis posthoc

Protocol common to Yeast,MediaMill,Helena sibling nodes,registered before the
first dataset run. Each dataset has its five OLD sealed H-LR1 train/validation/
test splits and the same OLD H-LR4 calibration/selection partition. Hash old
dataset,split,development and selection/cohort manifests before/after. All test
metrics are evaluated only after all five seeds' model/threshold choices seal.
These test sets have been used in research; findings are exploratory,not a
new independent confirmatory evaluation. No MoA or new data/environment.

## Fixed settings and methods

Four scientific fits per split,200trees,depth5,lr.1,lambda1,minleaf10,full rows/
features,deterministic native histogram and leaf reduction,no early stop. Same
20tree full warmup; fingerprint topology,coefficients and training state before
intervention must match EXACTLY across all four fits. Randomize fit order by
seed. Construct functional PCA basis on warmup training predictions only,using
FP64 fitting and native softmax contrast space. Rank r=floor(.5*(C-1)) for
multiclass or floor(.5*C) for multilabel,minimum1. Basis remains FIXED here.

1. full: native full updates retained,plus diagnostic per-leaf Hessian sums.
2. online_rank: D-weighted leaf projection,rank r,no sparse residual.
3. online_hybrid: rank r,at most floor(.05*M_tree*C) residual coordinates/tree,
   weighted support/refit rule from passed OPT-P.
4. online_more_rank: no residual,r+ceil(3*.05*C),capped at contrast dimension.
   This is a NOMINAL MEMORY control for12byteCOO entries versus4byte factors;
   it is NOT equal coefficient count or exact equal realized memory. Report
   realized per-model bytes/counts and avoid matched-cost causal claims.

Native G/H and full-output tree structure search remain identical algorithms.
Transient full leaf targets are projected then discarded; no hidden full old
leaf matrices retained in compact modes. Next G/H are computed from the actual
compact prediction update. This isolates optimization feedback,not histogram
work reduction. Per-output loss curvature is diagonal native approximation,
not the full softmax Hessian. Full-rank identity neutrality is the OPT-P test;
do not claim contrast-rank(C-1) weighted projection equals native diagonal fit.

Warmup is compressed at step20; preserve the mean warmup contribution via one
training-only C-vector bias correction. No later bias refitting. Derive three
posthoc models(rank/hybrid/more_rank) from full-fit targets and stored diagonal
leaf curvatures,using the EXACT SAME warmup basis,rank,rule and warmup-only bias
correction as each online method. Verify basis,prefix coefficients and sparse
support/values match. Thus later differences can reflect feedback rather than
a different posthoc algorithm. Never use final ensemble PCA in this comparison.

FP32 serialized factors,FP64 projection/reconstruction; checked PSD pseudoinverse
relative cutoff1e-12 for degenerate sparse-refit systems. Native ensembles keep
their original accumulation dtypes; export parity tolerance2e-5 is reported.

## Calibration and evaluation

All seven models(full +3 online +3 posthoc) use the previous identical discrete
threshold grid and shrinkage0/.25/.5/1. Fit on old calibration fold,choose maximum
macro-F1 on old selection fold,ties by loss then smaller shrinkage. No ranks,
residual budget or fit parameters chosen by test. Preserve all selection variants
and uncalibrated test metrics. Save each standalone model,verify CPU-only reload
in a separate process without importing training classes/CuPy. Test raw and
calibrated quality,loss ratio,macro/weighted F1 retention to full; quality guard
loss<=1.05,F1s>=.95 on BOTH selection and test for joint success.

Record parameters and actual numerical array bytes incl topology,one basis,
COO two indices+value,bias,thresholds; uncompressed NPZ size separately. For
multiclass full denominator use exact reference-class representation(C-1),
not inflated dense output storage. Joint75% success additionally requires both
counts and bytes<=.75. Report per-seed costs and quality,not just success count.
Five overlapping splits are not five independent tasks. This fixed rank is a
mechanism test,not optimal-rank search; poor results do not refute all low-rank
training or tensor-history methods. Compare online vs its posthoc counterpart,
hybrid vs rank,and higherrank with actual costs shown.

## Resources and limitations

One dataset per node,five splits20scientific fits +8 tiny synthetic regression
fits,shared A100,private CuPy training pool measured via tracked allocator.
Count all fitting/posthoc/calibration costs. Full retains CPU curvature arrays
solely to materialize matched posthoc controls; report these diagnostic bytes
separately. Exported payload is NOT whole fitted Python object or all training
memory. Private-pool peak excludes CUDA context and hidden library allocations;
do not claim process-total memory reduction. Stage timers synchronize around
gradient/build/statistics/projection/update/basis work for all fits; this is an
instrumented research prototype,not an optimized native library speed contest.
Foreign activity observed,no foreign process changes. Minimumfree2048MiB at
start,monitor256MiB; fixed command and registered environment.5400s wall cap,
standard orx repair limits. Archive raw fit histories,curvatures,sealed models,
splits and reports. Stop this node when its dataset question is answered;then
complete both other preregistered datasets as siblings before choosing the next
algorithm. Next topics(history source,projected structure,parallel merge) are
not claims established by this run.
