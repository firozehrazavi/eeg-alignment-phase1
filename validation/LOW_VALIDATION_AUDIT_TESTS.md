# Root-cause audit validation

29 selected CPU tests passed. CUDA_VISIBLE_DEVICES was empty. No optimizer step,
training loop, feature extraction or final-test evaluation test was run. Private
Drive arrays/checkpoints were unavailable. Synthetic backward tests verify gradient
connectivity only; they do not retrain any model or update weights.

Test selection:
- test_validation_root_audit (8): prefix read boundary, nested selection, refusal
  before test fields, read-only training-array integration, pool/split independence,
  exact-image vs concept counterexample, positive semantics, affine commutation,
  temperature/rank counterexample, uncertainty and actual batch partition.
- test_validation_root_notebook (1): default-denied gate, CPU enforcement, absence
  of training/evaluation/model-download entry points, compilation.
- ObjectiveTests: unique loss/gradient equivalence, duplicate identities,
  rectangular positive indexing, missing positives, permutation invariance (5).
- MetricTests (3), ProtocolTests (6): all use synthetic identities only.
- PatchTests: residual gradient, zero-branch mean, chunk-gradient equivalence,
  matched EEG initialization, fit-only moments/provenance guard (5).
- V7Tests.test_global_stratified_rows_exclude_validation (1).

Source audit examined training/model/preparation/metric code and notebook06b SOURCE
cells only. Historical percentages were taken from existing PHASE1_AUDIT.md,
not reopened result CSVs. No current final-test identities, labels or results read.

Report: docs/LOW_VALIDATION_ROOT_CAUSE_AUDIT.md.
Read-only Colab: notebooks/15_validation_root_cause_cpu.ipynb, pinned diagnostic code
ee22fa0b9ee83924a95fea82a27708f20b902bb9. Private-runtime diagnostics remain to run;
no causal ranking based on unobserved private outputs is claimed.
