# H-OPT-P: compact training correctness, before real-data comparison

Sole primary owner Sol/xhigh,no agents. Previous completed H-K3 remains frozen.
Fixed environment/registered command,shared A100 permitted,minfree2048MiB,
monitor256MiB,no foreign termination,900s preflight cap. Existing datasets,
models and splits are not changed or used by this synthetic gate.

Implement research-only GradientBoosting subclass retaining compact topology,
rank coefficients and sparse corrections,not a hidden full teacher. Full-output
G/H,structure histograms and the transient native leaf fit are retained for
OPT1 mechanism isolation; no claim of reducing histogram work or total training
memory. Native public model implementation is untouched.

Leaf target v=lr*(-G_leaf)/(H_leaf+lambda) is fitted in basis B under positive
diagonal D=H_leaf+lambda. Solve B.T D B a=B.T D v. Select at most floor(.05*M*C)
sparse coordinates by weighted squared residual,then refit a with selected
coordinates' weight zero and set L there to v-Ba. Use FP64 fitting and actual
FP32 saved factors. This optimizes the local diagonal-Hessian surrogate,not
macro-F1 and not a globally optimal sparse support. Count coordinates and indices.

Real-data follow-up planned warmup20,rank floor(.5*contrast_dimension),residual5%;
fixed functional basis on training warmup predictions. Global mean preservation
uses training only. A more-rank control adds ceil(3*.05*C) directions up to the
contrast limit as an approximate MEMORY control; measure actual final costs,
do not claim exact equality of realized parameters or bytes. Same-basis posthoc
rank/hybrid controls derived from full teacher will isolate feedback effects.
Separate parameter-count comparisons from memory comparisons.

This node itself owes ONLY correctness: property tests on weighted projection,
support bounds,non-worsening surrogate,identity-rank recovery,rotation invariance,
input immutability,invalid data,leaf remapping,serialization/byte accounting.
GPU synthetic multiclass/multilabel six-tree fits: native versus custom full,
identity-rank neutral,plus compact hybrid; verify probabilities2e-5 and exported
artifact replay. Compact trainer must discard full leaf coefficients and warmup
memberships after basis initialization. All previous sparse/shared-basis tests
also run. Eight small fits,no scientific dataset fits. Save guarded resource
samples and final test gate evidence. Freeze when answered; scientific quality
comparison and history/parallel hypotheses belong to children,not to this gate.
