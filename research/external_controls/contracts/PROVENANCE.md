Mechanically exported, unmodified pure modules from HASBoost commit 5fdca6e.
Source archive SHA256: 0714ffb1907218e550a3ed348370ae54f862d22a8a33544e637d345ba0f0946a.
No a100_runner import, no package installation, no changes to the frozen source.
Contracts retain their original module names under experiments/.

Correction: calibrated_quality lazily imports quality from a100_runner. The local
adapter contains ONLY that exact function (AST parity tested), its NumPy/sklearn
imports and the mechanically exported compactness_core normalization. It does
not import the original launcher's iterstrat/runtime/environment dependencies.
An explicit experiments/__init__.py isolates these contracts from editable local
installations of the original repository.
