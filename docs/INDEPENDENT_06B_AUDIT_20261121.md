# Independent historical 06b and tri-modal audit

Audit base: `3562609d9af4e1347ebcc21a9d344827ed23766e`, supplied branch
`research/trimodal-baseline-transfer-20261120`. Fixes are isolated on
`research/06b-independent-audit-20261120`; `main` is outside this change.

This is a code, protocol, and input-readiness audit. No full real-data 60-epoch
reproduction, full matched A/B/C/D training, GPU training, or official-test
evaluation was performed for this audit. CPU execution evidence and regression
results must be reported separately; synthetic tests are not real-data preflight
evidence. Neither recovered 78.8% performance nor SOTA is established.

Verified locally on Python 3.12 / PyTorch 2.9.1+cpu / NumPy 2.3.5:
**124/124 regression tests passed** (`python -m unittest discover -s tests -v`).
The tests include actual notebook CPU cells on synthetic fixtures, interrupted
CPU resume, BF16-to-FP32 head compatibility, frozen text-score reconstruction,
and rejection of altered/missing artifacts. The original Drive `train_008.pt`
was separately inspected: all 14 tensors have the expected finite
`float16[156,3200]` schema. This is only one chunk, not a complete real-data
preflight. See [execution evidence](../validation/independent_06b_20261121/evidence.json)
and [CPU regression log](../validation/independent_06b_20261121/cpu_regression.log).
Full real-data CPU validation remains pending in mounted-Drive Colab.

## Original protocol and reproduction scope

The reference is the [original 06b notebook at the audit base](https://github.com/firozehrazavi/eeg-alignment-phase1/blob/3562609d9af4e1347ebcc21a9d344827ed23766e/notebooks/06b_internvit_sota.ipynb),
especially code cells 4–6. The new runner reproduces its selected within-subject
`route_c` recipe for **sub-01, seed 0**, rather than rerunning the layer sweep,
all subjects, all seeds, or LOSO.

| Component | Historical recipe preserved |
|---|---|
| EEG | Existing float16 `sub-01_train_z16.npy`; posterior indices 46–62; all cached time samples |
| Visual targets | InternViT mean-pooled layers 20/24/28/32/36, width 3200; route logits `[-2,-1,0,-1,-2]` |
| Model | Flattened EEG → 1024-wide linear/residual MLP with dropout 0.3 and LayerNorm; five trainable linear image projections and global softmax routing |
| Loss | Symmetric diagonal contrastive cross-entropy; normalized embeddings; learned scale initialized at `1/0.07`, capped at 100 |
| Optimization | AdamW, learning rate and weight decay both `1e-4`; 60 epochs; batch 1024; OneCycleLR with `pct_start=0.05`; fresh row permutation each epoch |
| Selection | Last image of each ordered block of ten; 14,886 optimizer rows, 1,654 validation rows; validation every five epochs; earliest strictly best validation score |
| Precision | Historical BF16 backbone autocast on CUDA capability ≥8; otherwise FP32 computation; EEG storage remains FP16 |

The architecture classes match the original notebook by AST comparison. At the
expected 250 samples, the input dimension is 4250. There are 15 optimizer updates
per epoch, including a final 550-row batch, hence 900 updates. No extra gradient
clipping or different normalization is introduced into this historical runner.

The [recorded within-subject CSV](https://github.com/firozehrazavi/eeg-alignment-phase1/blob/3562609d9af4e1347ebcc21a9d344827ed23766e/results_within_v6b.csv)
contains 30 `route_c` rows: ten subjects × three seeds. Recomputing those existing
rows gives mean Top-1 **78.8333315%** and Top-5 **97.4166638%**. These are historical
record arithmetic, not newly measured predictions. Sub-01 seed 0 records about
80.0% test Top-1 and 27.9322863% validation Top-1 at epoch 25. The seed-0 mean
validation sweep selects layer 28, consistent with the fixed route used here.
The original within-subject path saves result JSONs but no model weights; its
LOSO checkpoint files are not substitutes for the missing within-subject weights.

## EEG channels, time axis, and image rows

The original notebook and reproduction select the same ordered channels:
`P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2`.
The [upstream preprocessing source](https://github.com/ncclab-sustech/EEG_Image_decode/blob/main/EEG-preprocessing/preprocessing_utils.py)
currently defines the same 63-channel order and requests ordered channel
selection. It also crops the first 50 EEG samples while preserving the original
time metadata, and orders training rows by stimulus condition. This supports the
code convention; current upstream source is not a pinned provenance record for
the particular Drive cache bytes. Do not reorder channels or crop EEG again on
the strength of a guessed convention.

In original 06b, feature extraction traverses `meta_v6.pt`'s `train_rel` sequence
without shuffling and saves consecutive chunks of 2048 images. The reproduction
concatenates train chunks 000–008 in that order. The final chunk must contain 156
rows. Legacy chunk tensors and the EEG array contain no per-row stimulus IDs.
Matching shapes, finite values, filenames, and newly recorded SHA256 hashes prove
structural compatibility and subsequent file stability; they do **not** independently
prove EEG-to-image identity or original channel labels. Basename indexing in the
original metadata builder also cannot rule out collisions without source-image
evidence. No collision, row permutation, or channel mismatch has been demonstrated.

Remaining provenance work is to bind the existing cache hashes to authoritative
ordered training stimulus IDs and source channel metadata. Preserve uncertainty
instead of manufacturing an identity manifest from the same assumptions.

## Validation isolation

The historical split holds out images from concepts also present during fitting;
it is not a held-out-concept experiment. Moreover, the [original cache preparation](https://github.com/firozehrazavi/eeg-alignment-phase1/blob/3562609d9af4e1347ebcc21a9d344827ed23766e/04_trimodal_crosssubject.ipynb)
averages repetitions and fits each channel's EEG mean/std over the entire
training partition and time axis. Original 06b fits visual mean and sample std
over all 16,540 training images, adding `1e-4` to std. Both include validation
images in preprocessing. The historical test EEG was normalized separately
within its test split. Those historical facts preclude describing the result as
strictly inductive or fully isolated validation.

The reproduction deliberately retains this recipe. Changing it to fit-only
normalization would create a different experiment. Its loader reads only training
EEG and the nine train feature chunks: it does not need test EEG, test feature
chunks, the merged feature cache, or metadata containing test entries. Upstream
preprocessing history remains a separate assurance question even when this
runner's own reads are training-only.

## Matched A/B/C/D interpretation and caption blocker

The four arms start from the same verified baseline checkpoint and new-head
initialization, with the same image split, schedule, batches, and budget:

| Arm | Additional semantic training signal |
|---|---|
| A | Image contrastive control; detached-EEG caption readout |
| B | Visual-residualized caption features |
| C | Random features scaled to residual feature variance |
| D | Residual caption features shuffled across different fit concepts |

All arms train the caption readout. Direct EEG-to-caption scoring uses EEG and
caption candidates, without candidate image features. Exact caption duplicates
are handled as shared positives. The added text normalization and residualizer
fit optimizer rows only. The warm-start comparison uses clipping and added
heads, and receives a fresh 60-epoch schedule: arm A is a matched continuation
control, not the historical run itself. B versus C/D tests the semantic signal
within this implementation; it does not establish language independent of images.
Generated captions depend on their source images.

Epoch selection and reported scores use the same training validation gallery.
These are development-selection scores, not an untouched final holdout. A single
subject and seed cannot support generalization, significance, or SOTA claims.

Execution additionally requires three independently prepared training-only files:

* `captions_train_only.npy`: finite floating embeddings, shape `[16540,512]`.
* `caption_ids_train_only.json`: 16,540 exact-caption IDs; repeated IDs require
  identical embeddings and must follow the same training image order.
* `caption_provenance_train_only.json`: generator, encoder, encoder revision,
  normalization description, both partitions `official_training`, and
  `used_test: false`.

These checks validate supplied declarations, not the truth of an unverified
provenance statement. Authoritative training image IDs, generation settings,
model revisions, and input/output hashes are still needed to establish lineage.
The mixed legacy caption cache is not an acceptable substitute for this path.
Do not relabel it training-only or invent provenance to bypass the gate.

## Confirmed engineering fixes and reproducibility limits

The fixes preserve the architecture and scientific recipe:

* Resume signatures now include effective device, autocast precision, and
  numerical backend settings. Requested CUDA device capability is used instead
  of silently inspecting the default device.
* Small-batch CPU timing no longer claims to estimate the batch-1024 schedule.
* Tri-modal backbone outputs are converted back to FP32 before the FP32 heads,
  preventing mixed-precision matrix dtype failures.
* Tri-modal preprocessing is saved and hashed with dimensions and row splits;
  resumed completed arms require their checkpoint and score artifacts to exist
  and match recorded hashes.
* The Colab bootstrap verifies reviewed source hashes before project imports and
  refuses stale imported modules or an unsuitable existing checkout.

An epoch-boundary checkpoint retains optimizer, scheduler, best selection state,
and RNG state. Signature agreement is necessary for reproducible resume, not a
guarantee of bitwise equality across GPU hardware, drivers, or library kernels.
GPU reproducibility remains unmeasured. Preserve input manifests, environment
records, preprocessing, best/latest checkpoints, logs, and score hashes together.

## Resource requirements

At 250 EEG samples, constructor-derived lower bounds are:

| Resident quantity | Bytes |
|---|---:|
| 21,793,798 model parameters, FP32 | 87,175,192 |
| Parameters + gradients + two FP32 Adam states | 348,700,768 |
| Five full FP32 visual target matrices | 1,058,560,000 |
| Selected posterior EEG, FP16 | 140,590,000 |
| One 1,654 × 1,654 FP32 validation score matrix | 10,942,864 |

These exclude activations, temporary normalization copies, original loaded chunk
tensors, allocator overhead, and checkpoint copies. They are not GPU capacity
guarantees. The nine feature chunks contain about 1.482 GB of tensor payload;
the full 63-channel EEG cache contains about 0.521 GB. Allow additional host RAM
for loading and normalization, and several GB of output space for atomic
checkpoint writes and four arms. CPU preflight needs no InternViT model download
or feature extraction. The CPU profile performs one warm-up and one measured
update at batch 32; it cannot predict GPU time or full-training CPU duration.

## Precise next execution step: CPU-only Colab

1. Open [`notebooks/06c_checkpointed_06b_and_trimodal.ipynb`](../notebooks/06c_checkpointed_06b_and_trimodal.ipynb)
   from the verified commit on `research/06b-independent-audit-20261120`. Use a
   fresh Colab runtime with **Hardware accelerator: None**. Keep
   **`RUN_BASELINE=False` and `RUN_TRIMODAL=False`**.
2. Run the first code cell, headed **Bootstrap the audited sources**. Mount the
   original Drive account. It must report the intended branch/commit and pass
   source verification before importing project code.
3. Use `CACHE=/content/drive/MyDrive/eeg_results/cache`. Require the original
   `sub-01_train_z16.npy` and
   `feat6_internvit6b_parts/train_000.pt` through `train_008.pt`.
   Run the cell beginning `from phase1.historical_06b import create_manifest`.
   Review the complete manifest, shapes, finiteness, hashes, and
   `official_test_opened: false` in `cpu_preflight.json`.
4. Run the next code cell, headed **CPU-only representative update**, and inspect
   `cpu_profile.json`. Stop after this cell. Do not run GPU training or any
   official-test notebook. The default output is
   `/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec/`.

A failed manifest/hash check is a blocker to diagnose, not permission to overwrite
original caches. Passing these CPU cells establishes input compatibility and a
working small CPU update. It does not resolve legacy semantic provenance or
authorize a GPU run; GPU execution remains a separate future step.
