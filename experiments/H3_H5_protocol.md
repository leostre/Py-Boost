# Bounded causal compactness study

The command, environment lock, datasets and paired splits remain fixed. Scientific
selection uses validation only. Test is evaluation, never an input to any controller.

H3 main: full G/H, 50% of the validation-selected full H1 coefficient reference;
fixed depth 3, schedule 3/5/7, adaptive plateau check every20 after40 with floor0.002.
All actual coefficients are independently counted after fitting and bounded.

H3 alternative1: floor0.01 only, same50%cap and15pairedfits. Motivated by no depth
growth in the original Helena/MediaMill traces. Alternative2:75%cap with fixed,
schedule,and earlier adaptive. Motivated by4/15 validation CE+twoF1 passes per
main controller. No further H3 tuning in this session.
One permitted diagnostic clarification was registered after the two alternatives:
all15 H1 validation-guarded compact choices useL2=1, unlike the inherited large-model
L2=10 on many pairs. At75%cap repeat all three controls changing onlyL2to1. Anchor
and cap stay immutable. This is a documented regularization/compactness interaction,
not test-driven tuning. This is the last H3 training branch in the session.

H4: fixed512training-row probe, separate centered trace-normalized G/H profiles.
Underlying schedule mode,L2=1,75%cap chosen before H4 from H3 validation final
models:12/15 guarded passes vs10/15 for fixed andadaptiveL2=1. Repeat nonspectral
schedule as direct control; staticTopKhalf,snapshot,andhistory use identicalmode.
Static TopKhalf control vs snapshot vs four-profile history. Stable event requires
four checks (earliest step60), temporal shift/noise thresholds1.5 and2.
Before H4 scientific fitting, latest consecutive shift also must be<=1.5noise:
median alone falsely accepts [0,0,large jump]. This diagnosed CPU counterexample
motivates the pre-run refinement; it is not a test-outcome-driven threshold change.
Dependent epochs never multiply sample count. MP-inspired diagnostic is NOT an
established RMT significance test. Original G/H frames are saved objects x outputs
x epochs. This is covariance aggregation, NOT CP/Tucker decomposition.

H5: Birds,MNIST12000,RT-IoT2022,new seeds604/614/624/634/644. GenBase excluded
before training due26exactsemanticproxies for20labels andrarepositives; this is
not established loaderleakage. MNIST is the existing registry task independently
audited for finitevalues,classbalance,duplicates and valid splits.
Frozen universal reference
depth5,lambda10,t200; canonical maximum-leaf cap fraction set before fitting.
Primary nonspectral policy selected on prior validation final-model passes, then
coefficient reduction, then validation loss. Prefer simplicity in close ties.
Primary schedule,L2=1 had12/15 prior guarded passes. H5 main fixes allL2=10
as originally preregistered; one previously declared sensitivity repeats all
four variants atL2=1. Thresholds/windows/modes unchanged on newtasks.
H4 history11/15 ties snapshot11/15 with lower medianvalidationloss, but loses to
nonspectral12/15; carry history only as a diagnostic, not selected bestpolicy.
Spectral candidate may remain a diagnostic without being called the best policy.

H5 bounded alternative2 (registered before its training): independent artifact
audit confirmed that the canonical maximum-leaf cap does not imply25% reduction
against the realized full model. All15 main schedule finals miss that count
criterion; some RT-IoT models are larger than the full control despite respecting
the nominal cap. Diagnosis uses counts and training/validation, not test.
Calibrate cap to75% of final200-tree full_reference(lambda10) actual coefficients
from completed main H5 report, SHA256
c281eba6834e50fef5cdc6361f271382ce038a58dac91935ce59f31bed3fd1ee.
Repeat fixed/schedule/history atlambda1 on identical15splits (45fits). No changes
to thresholds, windows, or controller logic; no selection between score values
for cap. Keep original results and compare against both full lambda references,
with validation-only anchor choice when combined. Final H5 training branch.

CE5% and additional validation/test macro+weightedF1>=95% evaluations stay separate.
Five overlapping splits are descriptive stability checks, not independent datasets.
Final online models and validation-chosen prefixes are reported separately.
Full fit time is not the training time of a retrospectively selected prefix.
History retained bytes are not peak GPU memory. No MoA, new paid compute or transfer
to SamplingZoo/Fedot.Industrial. User's new-run limit12GPUhours, one GPU job at a time.
