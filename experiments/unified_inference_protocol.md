# H-K3 unified inference, after H-K3P profiling

Build on frozen H-K3P run ed9a62ea-62fd-4ce4-b4c8-2fee0c43e09f: gather
dominates, so fusing the relatively small decoder/finish is not expected to give
an order-of-magnitude speedup. Preserve H-K2 accumulated as unchanged control.

One CTA per object, same 32/128-thread power-of-two policy. Tree routing,
FP64 coefficient accumulation, unique sparse-coordinate accumulation and tree
order remain unchanged. A rank vector and residual use dynamic shared memory;
one decode after the last tree, then stable sigmoid/softmax directly into the
only global N-by-C FP64 result. Full/reference-class use the same unified
kernel without decoder work. No N-by-r global array, separate residual or
probability array. FP32 decoder cache is transposed once from C-by-r to r-by-C
and replaces the old copy; no FP64 decoder needed. Count preparation, stored
arrays, dynamic/static shared and kernel local/register resources separately.
No claim that a single kernel launch means no validation kernels: the existing
isinf/any validation and host synchronization are retained equally for all roles.

Exactly the frozen 45 models and15 splits, batches1/32/256/all, previous ranks,
residuals, thresholds, feature rows. No fitting, labels, model selection or
precision reduction. Paired randomized9 rounds with2 warmups for accumulated
versus unified, for device and host modes. Preserve original CSR independent
oracle, atol2e-5 with exact calibrated decision agreement. Unit/generated tests
require2e-14 probability error on small cases and cover NaNs, threshold equality,
reference class, padded lanes, rank1/full rank, raw values, extremes, invalid
inputs, empty batches, source mutation and output ownership. Allocation spy
must show exactly one cp.empty result and no GEMM per request. Test varied
outputs across warp boundaries. No new dependencies or environment changes.

GPU-only timing and isolated own-CuPy-pool cold/warm memory at N1/full, not
total device process memory. Record kernel attributes including local bytes
(local bytes alone cannot identify dynamic spill traffic). Shared A100 allowed,
observe foreign processes without changes; require>=2048MiB at worker start
and>=256MiB during run. Input65 hashes and fixed environment checked before and
after; immutable source snapshot and fixed registered command. Keep same
walltime5400s and worker timeout240s, orx repair caps. Archive raw records/logs,
paired speed ratios, memory including decoder cache and quality parity. Whether
this wins or loses, freeze the node before proceeding to training research.
