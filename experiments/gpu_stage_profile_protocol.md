# H-K3P: profile before implementing unified kernels

Parent: frozen H-K2, commit 48853995a1665db0f96c9da2e53bb0e5369521be.
Owner: current primary session, Sol/xhigh; no delegation.

Use exactly the previous 15 splits and 45 saved full/rank/hybrid models,
unchanged ranks, residual coordinates, thresholds and test indices. Read test
features only; independent saved CSR predictor is the numerical oracle. No
fitting or selection. GPU only, same A100/environment/command, shared capacity
per explicit user permission; foreign contexts are observed, never terminated.

For each model and batches 1,32,256,all: two warmups and nine repeated rounds.
Randomize role and instrumented/device/host request order with the fixed seed.
Instrument the unchanged H-K2 operators using CUDA events around gather, GEMM
and finish, without synchronizing between those operators. Record validation
and allocation wall time; separately measure original uninstrumented device and
host request wall time. Stage intervals are GPU-stream timeline intervals,
including Python-dispatch idle gaps; they are NOT hardware-active instruction
time. An event pair with no GEMM is reported as zero decoder work for full.
Event creation and recording perturb the instrumented request; do not subtract
its stages from the independent request or assert a strict Amdahl bound.

Run the existing guarded GPU tests before the first case. Verify prediction
agreement (atol2e-5, no relative tolerance), exact calibrated decisions, all 65
frozen hashes before/after, and environment lock via the fixed entry point.
Print effective configuration, per-case samples and final dataset summaries.
Archive raw worker records, manifest and log. Freeze this node once it answers;
only then create a child implementing unified kernels. No Nsight installation
or environment modification. Stop on GPU capacity guard or ordinary orx repair
limits. No conclusion about pure routing versus coefficient-gather cost is
possible here, because those operations remain one existing gather kernel.
