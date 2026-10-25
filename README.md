# EEG–Image–Text Alignment: Phase-1 Baseline (THINGS-EEG2)

Preliminary experiments for a PhD proposal on reliable multimodal EEG decoding, semantic alignment and cross-subject generalization.
Everything runs on a free Google Colab T4 GPU.

## Main results (sub-01, 3 seeds, mean ± std)

| Setting | 200-way Top-1 | 200-way Top-5 |
|---|---|---|
| Previous prototype (2,000 samples, 1 conv layer, 5 epochs) | 0.0% | – |
| **New baseline: EEG → Image** | **22.7 ± 1.3%** | **55.2 ± 2.5%** |
| New baseline + text regularizer (λ = 0.1) | 20.8 ± 0.8% | 53.8 ± 1.7% |
| Chance | 0.5% | 2.5% |

![before / after](figures/fig1_before_after.png)

**Cross-subject gap** (model trained on sub-01, tested on sub-02; 100-way on the held-out half of the test set, chance 1%):
within-subject 31.3% → zero-shot on sub-02 7.7% → moment matching 10.3% → Procrustes calibration with 100 labelled trials 13.0%.

![cross-subject](figures/fig3_cross_subject.png)

**Coordinate recovery (SCORE-style Procrustes).** After the SVD transpose fix (`R = U @ Vh`), a full random rotation (22.5% → 0%) is recovered to 22.5% with ~250 labelled landmark pairs. Label-free MNN+CSLS recovery helps for moderate rotations (45–60°) but fails at 90°.

![score](figures/fig2_score.png)

## What changed from the prototype
- Training on all 16,540 training images (repetitions averaged), instead of 2,000.
- Temporal–spatial convolutional EEG encoder (ShallowConvNet/NICE family) with a learnable temperature and symmetric InfoNCE.
- Channel-wise normalization with **training** statistics.
- Model selection on a held-out validation split (one image per training concept, 1,654 images); the test set is used once.
- Multiple seeds; Top-1/Top-5 reported against chance.
- Procrustes fixed (`U @ Vh`, not `U @ Vh.T`); rotations applied in the source-subject EEG space.
- Chunked preprocessing to float32 to stay within Colab RAM.

## Reproduce
Open `eeg_alignment_v2.ipynb` in Colab with a T4 GPU and run cells 1–8 (~40 min). Cell 9 (all 10 subjects + 10×10 cross-subject matrix) is optional (~1–1.5 h).
Data: THINGS-EEG2 preprocessed 250 Hz release (Hugging Face `LidongYang/EEG_Image_decode`) — not included in this repository.
Image/text encoder: CLIP ViT-B/32 (OpenAI weights), used as a replaceable candidate.

## Limitations
Single source subject and single target subject so far; cross-subject numbers use only 100 evaluation images (1 image = 1 percentage point); 3 seeds. Text used only class-name prompts, not rich descriptions.

## Related work
- MindAlign (arXiv 2605.24523) — tri-modal EEG–image–text alignment
- SAMGA (arXiv 2604.17782) — subject-aware multi-granularity visual targets
- SATTC (CVPR 2026) — label-free test-time calibration for cross-subject retrieval
