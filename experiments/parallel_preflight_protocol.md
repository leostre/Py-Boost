# H-PAR correctness gate

Only synthetic tests: checkpoint prefix is counted once, increments have zero
bias, average formula and sparse duplicates decode correctly; full-rank
recompression preserves probabilities and mean; native continuation uses actual
checkpoint states rather than restarting. Launch two real independent process
workers with spawn (never fork a live CUDA runtime), verify identical checkpoint
hash, different PID/seeds, exported increments and averaging. Observe capacity
on shared A100, at least8192MiB free for this tiny concurrent gate. No real-data
fits or usefulness conclusions. Fixed environment/command, 900sec bound.

Poststudy safety child additionally checks rejection of checkpoint-free native
predict/predict_staged, exact decimal ceil(3*f*C) nominal budget, the C=100
full/online more-rank/posthoc prefix regression, and the old compact suite.
Historical Helena higher-rank artifacts remain rank65 and unchanged, with
actual costs reported. The future corrected nominal control is rank64. Main
H6/PAR sources rank49 and all frozen scientific models are unaffected.
