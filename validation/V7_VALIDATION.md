# v7 selective adoption validation — 2026-11-08

The supplied ZIP was inspected as source, not executed. It contains no saved
execution outputs. Its archive hash and cell/line findings are recorded in
`docs/V7_AUDIT_AND_ABLATIONS.md`.

`python -m unittest discover -s tests -v`: **33 tests passed** on Python 3.12.5,
torch 2.9.1+cpu and NumPy 2.3.5. This includes the previous 26 tests and seven
v7 regression tests:

- Filtered subject sampling returns original global rows and excludes validation.
- ConvEEG shape, gradients, input preservation and inference batch independence.
- Explicit augmentation with no-op/RNG preservation when disabled.
- Margin applies to all identity positives, zero margin matches the existing
  multi-positive loss, and candidate permutations preserve the objective.
- Nonzero text supervision is rejected rather than enabled by a cache file.
- Ensemble selection rejects target/test rows and missing models; scores align
  by explicit query/gallery IDs before averaging.
- Each new ablation config differs from corrected A0 in exactly one factor.

The end-to-end synthetic test additionally trains ConvEEG through the shared
runner, verifies that held-out training EEG is unnecessary, and exports only
source validation score bundles. Primary three-arm synthetic training and
epoch checkpoint/resume equivalence still pass.

No CUDA/Colab execution or real THINGS-EEG2 experiment was performed. No new
accuracy, SOTA, predicted improvement or actual Compute Unit consumption is
reported. All paid-run approval flags remain false and no full-run command is
executed by the pilot notebook. Historical notebooks/results are unchanged.
