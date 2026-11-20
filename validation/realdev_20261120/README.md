# Verified real THINGS-EEG2 training-subset pilot — 20 November 2026

**Scope:** within-subject `sub-01`, 200 previously explored training concepts (2000 images); 1600 train / 200 tune / 200 concept-disjoint development holdout. No official test EEG/image array values used; the legacy caption cache is a mixed train/test serialization that was deserialized, although only train entries were selected.

**Method:** matched A/B/C/D controls; 5 epochs, one seed, 17 channels × 250 samples EEG, five fixed InternViT layers, frozen 512-D CLIP text embeddings of BLIP-generated image captions. End-to-end backbone starts randomly initialized. This is NOT historical 06b reproduction.

| Arm | Control | Image Top-1 (200-way dev) | Direct exact-text Top-1 (200-row dev; 193 distinct captions) | Image Top-5 | Text Top-5 |
|---|---|---:|---:|---:|---:|
| A | image-only backbone; text readout for fair diagnostics | 5.0% | 0.0% | 16.5% | 2.0% |
| B | true aligned tri-modal residual | 5.0% | 0.0% | 16.5% | 2.5% |
| C | pseudo semantic residual | 5.0% | 0.0% | 16.5% | 2.5% |
| D | cross-concept shuffled residual | 5.0% | 0.0% | 16.5% | 2.5% |

**Observed comparison:** A/B/C/D each found 10/200 exact images and 0/200 exact caption strings at Top-1; the proposed tri-modal treatment has **NO observed Top-1 improvement** over image-only or the shuffled/pseudo controls in this pilot. A vs B image Top-1 predictions agree on 199/200 cases, vs C 200/200, vs D 198/200.

**Interpretation:** Technical execution is established, but the scientific claim is NOT supported. Do not tune this repeatedly examined holdout. A longer or stronger model cannot be promised to beat the RPA SOTA from these numbers. Correct next step is to preserve this negative pilot, compare train-budget/06b backbone compatibility on *separate* training-only selection, and design a pre-registered clean evaluation.

**Data integrity:** Input SHA256 hashes checked against original Drive preparation_state; all model-fit/evaluation rows from the official training partition; the mixed text cache file also serializes unused test caption vectors; caption strings loaded from BLIP cache and embeddings selected by the saved original training indices; exact duplicate captions grouped as valid retrieval positives (1924 unique of 2000, 55 duplicate groups). Original published score 78.8% is for a different official protocol and must not be compared to this 5% directly.

**Reproducibility:** See `prepare_real_dev.py`, `configs/joint_realdev_cpu.json`, `tests/test_joint_caption_duplicates.py`, `evidence.json`, and GitHub research branch. Results include output score matrices and model checkpoints in local experiment output directories; large tensors/checkpoints are intentionally not committed to GitHub.

**Limitations:** Development holdout is historically exposed (not a virgin test). One subject and one seed. Caption teacher is image-generated. Five epochs with random initialization is undertrained relative to 60-epoch 06b. No final official test, LOSO, multi-seed or paper-ready novelty claim.

**Direct inference functionality (verified):** The trained B checkpoint was loaded with `phase1.joint_inference --modality text` using only held-out EEG and 200 text-feature candidates. The emitted 200×200 score matrix was **bitwise identical** (maximum absolute difference 0.0) to the training-run saved holdout text scores; **no image gallery is read in the inference code path**. This is an engineering verification, NOT a demonstrated useful text-retrieval accuracy.

**Regression tests:** The modified joint model, pipeline, inference and duplicate-caption tests passed 24/24 locally (see `validation/joint_regression_tests_24.log`). This is separate from the original 103 tests on the previous code commit.
**Preparation metadata update:** Repreparing with the strengthened mixed-caption-cache warning produces identical EEG/visual/text numeric `.npy` files. The `train.json` hash changes only because new provenance fields were added; the recorded run used the original manifest.
