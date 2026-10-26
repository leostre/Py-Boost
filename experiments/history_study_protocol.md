# H6: fixed preregistered first comparison

Three datasets, five old splits each, 200 trees, three new sources per split:
current G/H, EMA signal then Gram, EMA Gram. Beta=.8; observations every five
trees, first refresh before tree41, then every20. Rank half the contrast width,
residual floor(5% reachable leaf/output coordinates) with the same weighted
projection/refit as OPT1. No window/rank/normalization/RMT threshold search.

Reuse exact full/fixed hybrid/more-rank OPT1 artifacts and old development/test
splits. Hash-lock reports, inputs, models and dense trajectory file.
Native warmup20 fingerprint and compact prefix40 must equal fixed hybrid.
Older blocks retain their own bases; count every basis and packed array.
The multibasis CPU predictor is a reference, not an H-K3 latency claim.

Signal Z=-G/sqrt(H+lambda/n), multiclass contrast, Frobenius normalization.
Full frozen trajectory uses native FP32 summation and loss derivatives,
verified against canonical prediction. All estimators see identical training
signals. Compare future energy at lags5/10/15/20, not just current reconstruction.
Record projector overlap across time and two same-time object halves.

Closed-loop fits change subsequent gradients. Seal all threshold choices on old
calibration/selection folds before tests across all15cases. Reused controls must
recover prior threshold choices. Quality: loss<=1.05 full; both F1>=.95 full on
selection and test. Budget75 for coefficients and arrays, counting all bases.
Count estimator and diagnostic state separately. Historical tests are exploratory.
45 scientific fits, fixed command/environment, shared A100>=2048MiB free,
5400sec cap, no foreign process control. Not CP/Tucker, temporal-order modeling,
Adam, or a GaLore theorem for boosting.
