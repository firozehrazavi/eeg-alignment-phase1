# Training audit and separately gated test

Notebook13 reads private Drive checkpoints on CPU; no artifacts are available to
the assistant. It checks 15 history entries (1..5 for each arm), three epoch5 latest
checkpoints, best_state and best_epoch, and timing uniqueness if the log exists.
The trainer replaces latest checkpoints each epoch; 15 independent epoch snapshots
cannot be verified because they were never saved. Tied validation maxima select
the earliest epoch (strict greater-than update in training).

Validation candidates are all images of held-out training concepts, not one prototype
per concept. With 200 training concepts and a 10% holdout, typically 20 concepts x10
images =200 candidates; the audit computes the actual count from provenance.
Chance Top-1=100/N percent. Stored source_val_top1 is already percent. Only validation
Top-1 was recorded, not validation Top-5/MRR. Loss is the mean of batch losses.
Timing/peak memory fields are printed only if present. These are not test results.

Notebook14 defaults to CPU filesystem preflight only. It does not deserialize
catalog/provenance or inspect test labels until explicit approval. Following approval,
read_catalog checks train/test identity/concept/content-hash disjointness; the
concept split is reconstructed and compared to saved fit/validation rows. Recorded
input hashes and token manifest identity must match. Upstream normalization remains
benchmark_preprocessed: these checks do not establish strict inductive LOSO.

Evaluation never calls run/train, optimizer, backward or a vision model. It loads
embedded best_state, recomputes exactly the train-fit EEG/visual normalizers, freezes
weights, and uses inference_mode. All arms use FP32 cosine scoring, 200 identical
candidate identities, batch8 and patch chunk1. This common FP32 evaluation differs
from mixed-precision training and is recorded explicitly. Top1/Top5 are percentages;
MRR is mean reciprocal 1-based rank (0..1), with the same identity-based tie policy.
MRR extends the existing rank calculation; it was not a training selection metric.

Results/scores/identity bind selected checkpoint hashes, dataset, evaluator and test
payload hashes in a new output directory; logs have their own new sibling directory.
Completed identical evaluations are idempotent; interrupted output is never overwritten.
Choose a new output name for retry after failure. Time/CU guards and persistent logs
are active. Full token payload scanning is not part of preapproval preflight.

Local tests cover read-only audit, missing/duplicate epochs, tie selection, known
retrieval scores, approval-before-read and existing training regressions. Actual
private data results and full-scale final inference remain unexecuted here. This
implements EEG-image evaluation only; tri-modal EEG-image-text work remains planned.
