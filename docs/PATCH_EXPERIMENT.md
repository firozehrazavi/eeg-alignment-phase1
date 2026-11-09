# InternViT patch-token experiment

This is an implementation of an **RPA-inspired InternViT variant**, not the
official RPA model or a reproduced accuracy claim. Reference:
https://arxiv.org/abs/2609.31698 . RPA uses CLIP ViT-H/14 intermediate tokens;
this experiment deliberately retains the repository's stronger existing
InternViT target and EEG projection architecture. Text, semantic negatives and
SCORE remain separate future ablations. No original notebook or CSV is changed.

## Start here

Open `notebooks/10_rpa_patch_pilot.ipynb` from PR #2 in Colab Pro+.
Start its CPU-only section A first. It inventories existing images/pooled caches,
downloads and verifies only sub-01 EEG, and resumes interrupted preparation on
Drive. See `CPU_PILOT_INPUTS.md`; section B requires separate GPU approval.
It retains Colab's driver-compatible torch/torchvision, pins auxiliary dependencies,
records the full environment, runs tests, checks real inputs, prepares
one subject/200 training concepts, extracts layer-28 tokens and trains three
arms for five epochs. Every test query still competes against all 200 images.
The notebook has not been executed on a Colab GPU in this environment. GPU model
and VRAM are detected at startup; no Pro+ accelerator is assumed to be guaranteed.

### Pro+ approval and Compute Unit budget

The user's reported balance is 500 CU. **Live balance and CU/hour are not
available through a supported Python billing API used by this implementation.**
The notebook requires both observed values from Colab's Resources panel and
displays them; it never guesses a rate from the GPU name. Google documents that
resource availability and limits vary: https://research.google.com/colaboratory/faq.html .

Default planning limits are 25 CU (5% of 500) and two connected hours per session.
These are conservative limits, not predictions. Four approval flags default to
false: 32-image extraction probe; remaining extraction; one training epoch per
arm without test evaluation; and completion of the five-epoch pilot. The training
profile uses the final LR schedule and resumes its exact epoch checkpoints.
There is no full-run launch cell. Full ten-subject/three-seed execution requires
pilot review and separate user approval.

CU estimate = entered CU/hour × connected elapsed hours, plus previous pilot
sessions' estimated spend. A 50% allowance accompanies probe-based projections.
For illustration only, a two-hour pilot at rates of 5, 10 or 15 CU/hour would
consume 10, 20 or 30 CU; these are sensitivity examples, not GPU price quotes.
Actual projected cost remains unknown until the runtime rate and probe timings
are supplied. Recheck Resources after each stage and when reconnecting.

The Drive JSONL ledger logs estimated CU and memory. A subprocess supervisor
checks limits every 30 seconds, and the extractor/trainer checks at batch
boundaries. These are approximate limits: setup, idle time, rate changes, other
notebooks and time after a stopped cell can differ from the estimate. A stopped
cell **does not disconnect the GPU**. Disconnect/delete it after saving; the
notebook includes an optional explicit disconnect cell. No paid run was started
from this local environment.

The default three-arm comparison is:

| Arm | Visual target/head | EEG architecture |
|---|---|---|
| baseline | Original `ImageSide`: means at 20/24/28/32/36, initial routing logits -2/-1/0/-1/-2 | Original `EEGProject`, 1024 dimensions, dropout .3 |
| mean | L28 token mean followed by a linear 3200-to-1024 head | Same |
| patch | L28 native grid, residual spatial bottleneck, GAP, linear head | Same |

The adapter is Conv1x1(D,256), GN32, GELU; Conv3x3(256,256), GN32,
GELU; Conv1x1(256,D), GN32, GELU; add input, spatial mean, linear projection.
Its native grid is inferred and validated, never spatially downsampled.
Outputs are L2-normalized in the contrastive objective and retrieval scoring.
The mean control uses precisely the patch cache's token mean, not a different
backbone. Both use the same train-only per-feature affine normalization.

The baseline **architecture** is unchanged. The corrected protocol is not an
exact replay of historical 78.8%: concept splitting, source-only normalization,
batch 64 and cosine scheduling differ from historical image-level validation,
session normalization, batch 1024 and OneCycleLR. All three new arms use the
same corrected settings. Historical 78.8% remains a separately recorded result.
Do not interpret a five-epoch subset pilot as a full-data performance estimate.

## Inputs missing from this local checkout

There are no real EEG `.npy`, visual `.pt`, or patch-token arrays in this
checkout. Drive has not been mounted or inspected here. The notebook requires:

| Required path in Colab | Purpose |
|---|---|
| `/content/drive/MyDrive/THINGS-EEG2/images_set/image_metadata.npy` | Authoritative stimulus/concept rows |
| `/content/drive/MyDrive/THINGS-EEG2/images_set/training_images/...` | Original images referenced by metadata |
| `/content/drive/MyDrive/THINGS-EEG2/images_set/test_images/...` | 200 original test images |
| `/content/drive/MyDrive/THINGS-EEG2/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_training.npy` | Pilot EEG before additional z-scoring |
| `/content/drive/MyDrive/THINGS-EEG2/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_test.npy` | Pilot test EEG, all repetitions |
| `/content/drive/MyDrive/eeg_results/cache/meta_v6.pt` | Historical cache image order and relative paths |
| `/content/drive/MyDrive/eeg_results/cache/feat6_internvit6b.pt` | Five historical mean-feature arrays |

Full evaluation additionally needs the two EEG files for **sub-02 through
sub-10** at the same directory pattern. If raw data is staged at the old
`/content/EEG_Data` location, change `RAW` in the notebook; no file is assumed
to exist. The original image paths are resolved from `meta_v6.pt` and checked
against the authoritative metadata. Duplicate image IDs/content and overlap
between training/test concepts are rejected.

The NEW token outputs are:
`/content/drive/MyDrive/eeg_results/patch_phase1/pilot_sub01_200concepts_v1/tokens_L28/{train.npy,test.npy,train_done.npy,test_done.npy,manifest.json}`.
They cannot be reconstructed from historical pooled vectors. In 06b, both
`feat6_internvit6b.pt` and `feat6_internvit6b_parts/*.pt` retain only CLS and
mean embeddings. This conclusion is from extraction code; actual Drive cache
contents cannot be inspected locally.

## Extraction and resources

Extract a **single fixed layer**, once per image. Pin the model revision and
use its pinned `CLIPImageProcessor`, as in 06b. Disable FlashAttention, truncate
the encoder after layer 28, capture that block with a hook and do not request
all hidden states. `hidden_states[28]` corresponds to block index 27 output.
Use BF16 backbone inference when supported, otherwise FP16; store FP16 tokens
and start with image batch 1. Record extraction precision and refuse a mixed-
precision cache resume on incompatible hardware.
No gradient, quantization or spatial pooling is used during extraction.
Only the optional flash_attn static import check is bypassed while loading;
the original import checker is restored immediately.

The extractor writes mmap arrays and completion masks, flushing token rows
before marking them complete. Interrupted extraction resumes missing rows.
Changing revision, layer, catalog or preprocessing requires a new cache.
Sampled pooled L28 vectors must match cached token means within 2% relative
L2 error before training; mismatches fail rather than silently changing the
baseline. This is a compatibility check, not proof of original model revision.
The historical cache lacks immutable revision provenance. If it fails, stop
and reconcile the revision/processor; do not loosen tolerance to obtain a run.

Resource estimates below are **not measured GPU benchmarks**:

| Resource | Pilot (2,200 images) | Full (16,740 images) |
|---|---|---|
| One native 1024-token, D=3200, FP16 cache | 13.4 GiB | 102.2 GiB |
| Extraction GPU | Plan 12–20 GiB at batch 1; L4 24 GB/A100 | Same VRAM; more time |
| Adapter training GPU | Initial planning range 4–10 GiB at batch 64; measure with adaptive chunks/AMP | Same batch-dependent VRAM |
| CPU RAM | High RAM; plan 25–40 GiB for loading model/raw pickle arrays | High RAM; raw subjects loaded sequentially |
| Disk beyond source EEG/images | Plan 40–60 GiB for weights, caches and staging | Plan at least 140–170 GiB; more if keeping two full cache copies |

Formula: images × tokens × width × 2 bytes. The actual grid/width are recorded
in the manifest. Extraction and training run in separate processes, so the
backbone does not occupy training VRAM. GroupNorm permits checkpointed adapter
chunks without changing batch semantics; the full contrastive batch stays 64.
This is not gradient accumulation, which would change the negative set.
Patch inputs stay on CPU and transfer one checkpoint chunk at a time. Train
with BF16 autocast where supported, otherwise FP16 autocast plus GradScaler;
CPU uses FP32. Contrastive logits/normalization and retrieval scoring are FP32.
OOM retries halve only the adapter microchunk, preserving contrastive batch,
precision and dropout RNG. Chunk=1 failure stops with a clear message. GPU
memory telemetry includes free, allocated, reserved and peak allocated VRAM.
The notebook records measured peak training GPU allocation. A GPU run is
still required to validate these estimates and pinned Colab compatibility.

Drive mmap is persistent but slow; stage the completed pilot locally before
training. Verify available disk first. For a full cache, avoid keeping both
copies on the runtime if capacity is insufficient. Do not place generated
arrays or checkpoints in the Git checkout.

## Evaluation contract

- One concept split seed across subjects/arms, disjoint validation concepts.
  Pilot selection is deterministic from authoritative concept IDs.
- EEG channel statistics fit source training concepts only, streamed in FP64.
  Visual normalization fits training concepts only. Target/test batch moments
  are never used for plain cosine retrieval.
- LOSO never opens held-out training EEG. All source subjects validate on held
  concepts; their mean score selects the checkpoint. Architecture and training
  settings are fixed a priori; this runner does not perform hyperparameter
  searches. Future searches require source-only inner subject folds.
- Three arms share initial EEG parameters, batch permutations, split, epochs,
  optimizer, LR schedule and dropout RNG stream. Each selects its own checkpoint
  on the same source validation data. Validation gallery size is the number of
  held-out training images; it is explicitly not the 200-way test metric.
- Testing uses unique 200-image candidates, explicit stimulus IDs, Top-1 and
  Top-5. Averaged EEG repetitions are one query per test image. No hidden
  per-query target label adaptation occurs.
- `diagonal` preserves symmetric CLIP loss. `multi_positive` is a distinct
  objective ablation using exact stimulus IDs across source subjects. Never
  conflate adapter effects with changed objectives. Semantic similarity does
  not create positives in this visual-only experiment.
- Distributed preprocessed EEG may use target/session whitening. The converter
  therefore declares `benchmark_preprocessed`, and outputs
  `loso_zero_calibration_benchmark_preprocessed`, **not strict inductive LOSO**.
  Strict mode requires upstream target-independent preprocessing, an explicit
  audit record and `regime: strict`; the declaration must be backed by evidence.
- CSLS is optional and emits separate `*_transductive_csls_*` rows. No SCORE,
  calibration labels or transductive recovery is folded into plain accuracy.

## Full schedule and resuming

Prepare a NEW directory with `phase1.patch_prepare --subjects sub-01 ... sub-10
--concepts 0`. Extract its tokens once. The six JSON configurations cover the
pilot, full within/LOSO diagonal, full within/LOSO multi-positive and strict
LOSO multi-positive. Full configurations use seeds 0/1/2 and 60 epochs. Ten
subjects schedule 30 runs per arm for within and 30 runs per arm for LOSO.
An optional `targets` list restricts tested subjects without removing sources.
Do not launch full configurations until the pilot passes and the user approves.
The same 17 posterior channels and all supplied time samples are used for all
arms; this differs from historical 63-channel LOSO and the RPA paper protocol.

Run `python -m phase1.patch_experiment --dataset DATASET.json --config CONFIG.json
--output NEW_DIR`. A completed run contains provenance, source/input hashes,
split rows, checkpoints, score matrices, histories, JSON/CSV results and a
`COMPLETE` marker. `--resume` checks provenance and skips completed arms. Atomic
`*_latest.pt` epoch checkpoints restore model, optimizer, scheduler, AMP scaler,
best state, CPU/CUDA RNG and batch-permutation generator. An interrupted epoch
restarts from the last complete epoch; there is no mid-batch resume. Precision
changes are rejected on resume. `--profile-one-epoch` saves one epoch per arm,
does not evaluate tests and writes `PILOT_PROFILE_COMPLETE` instead of `COMPLETE`.
Cache contents are checked for finiteness and sampled
pooled agreement, not fully byte-hashed on every training run (100+ GiB).
Preserve the extraction manifest and immutable cache with experiment outputs.

Summarize seeds within each subject first, then report mean and SD across ten
subjects; retain paired per-subject differences, all seeds, and failure logs.
Never select seeds or tune against outer test scores. Labeled calibration and
text ablations are deliberately outside this implementation.

## Local validation

`python -m unittest discover -s tests -v` runs existing evaluation/objective
tests plus adapter shape/formula/gradient checks, zero-residual mean equivalence,
checkpoint chunk gradient equivalence, matched EEG initialization, streaming
normalization, strict provenance rejection, 10-fold scheduling, and an entire
three-arm synthetic LOSO experiment repeated deterministically. The test deletes
held-out training EEG and checks refusal to overwrite completed runs and refusal
of incomplete/mismatched token caches. Tests also cover CU arithmetic, cap
enforcement, simulated OOM retries, and equivalence of uninterrupted training
to profiling then resuming complete optimizer/RNG state. These are software tests, not accuracy
evidence. No real THINGS-EEG2 training was executed locally.
