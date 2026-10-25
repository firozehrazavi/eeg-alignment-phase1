# EEG–Image Alignment — Phase 1 Baseline (THINGS-EEG2)

Preliminary PhD work (supervisors: Dr. Leila Safari, Dr. Mohsen Afsharchi).
Goal of this phase: a **reproducible** EEG→image zero-shot retrieval baseline and a first measurement of the cross-subject gap. Nothing here is a final method; the visual/text encoder is a replaceable target.

## Setup
| Item | Value |
|---|---|
| Data | THINGS-EEG2, preprocessed 250 Hz, 63 ch × 250 samples, repetitions averaged |
| Split | 16,540 train images (1,654 concepts); validation = 1 image per training concept; test = 200 unseen concepts |
| EEG encoder | temporal–spatial conv (ShallowConvNet/NICE family) + projection |
| Target | frozen CLIP ViT-B/32 (openai) image features; optional text features (`"a photo of a {concept}"`) |
| Loss | symmetric InfoNCE, learnable temperature; optional text term with weight λ |
| Training | AdamW, lr 3e-4, batch 256, ≤ 40 epochs, early stopping on val Top-5 (patience 10) |
| Normalization | channel z-score with **training-set** statistics |
| Hardware | free Colab T4 (~3–4 min per subject) |

Chance level: 200-way Top-1 = 0.5 %, Top-5 = 2.5 %; 100-way Top-1 = 1 %.

## Results

### 1. Within-subject, sub-01 (3 seeds) — `results/results_within_subject_run1.csv`
| Variant | Top-1 (%) | Top-5 (%) |
|---|---|---|
| Previous prototype (2,000 samples, 1 conv layer, 5 epochs) | 0.0 | – |
| EEG→Image | 22.7 ± 1.3 | 55.2 ± 2.5 |
| EEG→Image + text (λ = 0.1) | 20.8 ± 0.8 | 53.8 ± 1.8 |

The text regularizer gives no gain in this setting.
A second independent run with the same seeds (`results_within_subject_run2.csv`) gave 22.8 ± 2.0 / 55.0 ± 3.5 (image-only) and 21.2 ± 0.3 / 53.8 ± 1.6 (+ text): same conclusion, about ±1 point run-to-run variation from non-deterministic cuDNN. The notebook now sets cuDNN to deterministic mode.

![before / after](figures/fig1_before_after.png)

### 2. All 10 subjects (seed 0) — `results/results_all_subjects.csv`
Top-1 = **28.6 ± 6.6 %**, Top-5 = **62.5 ± 7.8 %** (range 15.0 % sub-05 to 39.0 % sub-08).

![ten subjects](figures/fig4_ten_subjects.png)

### 3. 10 × 10 cross-subject matrix (no adaptation) — `results/results_cross_matrix.csv`
Model trained on subject *i*, tested on subject *j*. Diagonal = **27.2 %**, off-diagonal = **4.5 ± 2.0 %**.

![cross matrix](figures/fig5_cross_matrix.png)

> The diagonal differs from Section 2 because here every test set is normalized with **its own** test statistics (needed for unseen subjects). Normalization alone moves sub-05 from 15.0 % to 24.0 %, so normalization is part of the cross-subject problem.

### 4. sub-01 → sub-02 adaptation (100-way, evaluation half of the test set) — `results/results_cross_subject_sub01_sub02.csv`
| Method | Top-1 (%) |
|---|---|
| within sub-01 | 31.3 |
| zero-shot on sub-02 | 7.7 |
| moment matching (label-free) | 10.3 |
| Procrustes, 100 labelled trials | 13.0 |

![cross-subject](figures/fig3_cross_subject.png)

### 5. SCORE-style sanity check (synthetic rotation of sub-01 embeddings) — `results/results_score.csv`
- Full random rotation: 22.5 % → 0 %; recovered to 17.5 % with 50 labelled pairs and fully (22.5 %) with 250.
- Procrustes bug fixed: `R = U @ Vh` (the prototype used `U @ Vh.T`).
- Label-free MNN + CSLS refinement helps at moderate angles (60°: 9.0 % → 15.5 %) but fails at 90°.

![score](figures/fig2_score.png)

## Limitations
- Cross-subject adaptation numbers use 100 evaluation images (1 image = 1 percentage point).
- 10-subject results use one seed; text uses only class-name prompts.

## Not comparable to published LOSO numbers
Our 4.5 % trains on **one** subject and tests on another. LOSO results (e.g. MindAlign ≈ 15 %, SAMGA 34.4 %) train on 9 subjects. LOSO is the next experiment.

## Reproduce
Open `phase1_baseline.ipynb` in Colab (GPU: T4) and run all cells top to bottom. Data is downloaded from the Hugging Face dataset `LidongYang/EEG_Image_decode`. Seeds are fixed and cuDNN is set to deterministic mode.

## Next steps
1. Leave-one-subject-out training on 9 subjects.
2. Compare calibration methods (few-shot Procrustes, moment matching, label-free test-time calibration such as SATTC).
3. Ablate the visual target (CLIP B/32 vs larger encoders) and richer text descriptions.
