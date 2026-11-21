# Checkpointed historical 06b reproduction

For the independent audit, measured verification scope, remaining provenance
blockers, and the next CPU-only Colab step, see
[the 2026-11-21 audit](INDEPENDENT_06B_AUDIT_20261121.md).

`notebooks/06c_checkpointed_06b_and_trimodal.ipynb` is the single Colab entry
point for a train-only `sub-01`, seed-0 reproduction followed, optionally, by a
matched warm-start A/B/C/D comparison. It does not evaluate or open official
test EEG, test image features, or the combined train/test feature cache.

## Required baseline inputs

Use the original cache directory from 06b, normally
`/content/drive/MyDrive/eeg_results/cache/`:

| File | Required contents |
|---|---|
| `sub-01_train_z16.npy` | Historical float16 EEG cache, shape `[16540, 63, time]`, same image order as the 06b train feature chunks. |
| `feat6_internvit6b_parts/train_000.pt` through `train_008.pt` | Nine original train-only chunks. Each includes mean- and CLS-pooled layers 12, 16, 20, 24, 28, 32, 36; each selected `(<layer>, "mean")` tensor has width 3200 and the chunk's expected row count. |

No `*_test_z16.npy`, `test_*.pt`, `meta_v6.pt`, or merged
`feat6_internvit6b.pt` is required or loaded. The train cache paths, dimensions,
finite values, and SHA256 hashes are checked by the CPU preflight. It assumes
the historical cache's documented row alignment between `sub-01_train_z16`
and `train_*.pt`; these legacy arrays do not themselves carry per-row IDs.

The reproduction selects posterior channels in the exact 06b order
`P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2`; uses mean-pooled
InternViT layers 20/24/28/32/36; applies 06b's train-partition visual mean and
sample standard deviation plus `1e-4`; initializes route logits to
`[-2,-1,0,-1,-2]`; and uses the original 1,024 batch, 60 epochs, AdamW
(`lr=1e-4`, `weight_decay=1e-4`) and OneCycleLR (`pct_start=0.05`). Validation
is the last of each ten ordered train images per concept, exactly as in 06b.
No new EEG standardizer is fitted: the historical train-only `z16` input is
used unchanged.

The CPU preflight checks 14,886 fit rows, 1,654 validation rows, 900 optimizer
updates and model/target memory lower bounds. The notebook then profiles one
actual CPU update before enabling the separately gated GPU run. A latest
checkpoint is written atomically each epoch; it includes optimizer, scheduler,
best validation state and RNG state. `--resume` verifies input hashes, source
code, environment, device/precision/backend settings and protocol. A CPU
interrupted-resume regression is not evidence of bit-exact CUDA replay.
The batch-32 CPU profile reports measured update time; it does not extrapolate
that time to the batch-1,024 schedule. The final `*_best.pt` is a plain 06b
`Aligner.state_dict()` and `run.json` records subject, seed, layers, dimensions,
selected epoch, input hashes and checkpoint hash.

## Additional tri-modal inputs

Tri-modal execution is separately gated and refuses to start before the
checkpointed baseline completes. It requires three **train-only** files in the
cache directory:

* `captions_train_only.npy`: `[16540, 512]` finite floating-point embeddings,
  aligned with the historical training image row order.
* `caption_ids_train_only.json`: 16,540 exact-caption identity strings; exact
  duplicate captions must share one ID and one identical embedding.
* `caption_provenance_train_only.json`: nonempty `caption_generator`,
  `caption_encoder`, `encoder_revision`, `embedding_normalization`,
  `source_image_partition="official_training"`, `partition="official_training"`,
  and `used_test=false`.

The mixed legacy `feats7_blip_caps_clip_b32.pt` and mixed caption JSON are
intentionally not read. Obtain these files from an independently produced
training-only caption pipeline; do not extract test entries from the mixed
cache. Text normalization and the visual-to-text residualizer fit only on the
optimizer rows. A/B/C/D start from the same verified 06b checkpoint and same
new-head initialization, share the historical schedule, and report both image
retrieval and duplicate-aware direct EEG-to-caption retrieval on the training
validation gallery. Caption scoring consumes EEG and caption candidates only;
it does not use candidate image features.

Tri-modal outputs preserve a separate hashed `preprocessing.pt` containing
model dimensions, selected EEG channel indices, frozen image/text normalizers,
residualizer state and fit/validation rows. Keep it with each plain
`*_best.pt`, `run.json`, and validation-score artifact. Resume rejects missing
or changed completed checkpoints, scores, or preprocessing. Reconstruct the
model from `model_dimensions`, load its state, select the saved channels from
the historical **already-z16** EEG, and normalize candidate text using the
saved moments. `tests/test_trimodal_06b.py` verifies this direct-text path
without refitting or image inputs. These files are not the separate
pre-z-score bundle format accepted by `phase1.joint_inference`.

This is a reproduction/engineering workflow, not a result or SOTA claim. It
does not recreate weights that the historical notebook failed to save, and it
does not establish performance until the real full-size inputs are available
and the run completes.
