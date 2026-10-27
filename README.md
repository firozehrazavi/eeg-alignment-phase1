# EEG-Image Alignment: Phase 1 Baseline (THINGS-EEG2)

Preliminary PhD work (supervisors: Dr. Leila Safari, Dr. Mohsen Afsharchi).
Aim of this phase: a reproducible EEG-to-image zero-shot retrieval baseline, and a measurement of how well it transfers to new subjects. The image encoder is treated as a replaceable target, not a fixed design choice.

## Updates

| Date | Added |
|---|---|
| 2026-10-27 | `04_trimodal_crosssubject.ipynb`: ViT-H/14 target, caption alignment, CNN vs transformer, LOSO with calibration, test-time refinement (CSLS), single-trial training; results sections 9-13; figures 9-14 |
| 2026-10-26 | `01_baseline.ipynb`, `02_loso.ipynb`, `03_layer_ablation.ipynb`; results sections 1-8; figures 1-8 |

## Notebooks

| Notebook | Content |
|---|---|
| `01_baseline.ipynb` | Within-subject baseline, text regularizer, SCORE-style sanity check, sub-01 to sub-02 calibration, all 10 subjects, 10 x 10 cross-subject matrix |
| `02_loso.ipynb` | Leave-one-subject-out training (9 subjects) and Procrustes calibration on the held-out subject |
| `03_layer_ablation.ipynb` | Visual target taken from different CLIP blocks; within/cross-subject and k-way accuracy |
| `04_trimodal_crosssubject.ipynb` | CLIP ViT-H/14 target, BLIP captions as a third modality, CNN vs transformer + CNN encoder, leave-one-subject-out with calibration on the new subject |

Notebooks 01-03 run on a free Colab T4; notebook 04 was run on a Colab L4. Data: THINGS-EEG2 preprocessed 250 Hz release (Hugging Face `LidongYang/EEG_Image_decode`), not included here.

## Setup

| Item | Value |
|---|---|
| Data | THINGS-EEG2, 63 channels x 250 samples, repetitions averaged |
| Split | 16,540 training images (1,654 concepts); validation = 1 image per training concept; test = 200 unseen concepts |
| EEG encoder | temporal-spatial convolution (ShallowConvNet/NICE family) + projection |
| Target | frozen CLIP ViT-B/32 image features (text features optional) |
| Loss | symmetric InfoNCE with learnable temperature |
| Training | AdamW, lr 3e-4, batch 256, early stopping on validation Top-5 |

Chance: 200-way Top-1 0.5%, Top-5 2.5%; 100-way Top-1 1%.

## Base papers and what was taken from each

| Paper | Link | Used in | What was taken |
|---|---|---|---|
| NICE, Song et al., ICLR 2024 | [arXiv 2308.13234](https://arxiv.org/abs/2308.13234) | 01-04 | temporal-spatial CNN encoder; contrastive EEG-image alignment; 200-way zero-shot protocol |
| ATM, Li et al., NeurIPS 2024 | [arXiv 2403.07721](https://arxiv.org/abs/2403.07721) | 01-04 | preprocessed THINGS-EEG2 release; ViT-H/14 target (04); channel-attention layer before the CNN (04, ATM-style, not a reproduction) |
| THINGS-EEG2, Gifford et al., NeuroImage 2022 | [doi](https://doi.org/10.1016/j.neuroimage.2022.119754) | 01-04 | dataset |
| CLIP, Radford et al., ICML 2021 | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) | 01-03 | ViT-B/32 image and text features |
| OpenCLIP, Cherti et al., CVPR 2023 | [arXiv 2212.07143](https://arxiv.org/abs/2212.07143) | 04 | ViT-H/14, `laion2b_s32b_b79k` |
| BLIP, Li et al., ICML 2022 | [arXiv 2201.12086](https://arxiv.org/abs/2201.12086) | 04 | one caption per image (`blip-image-captioning-large`) |
| MindAlign, 2026 | [arXiv 2605.24523](https://arxiv.org/html/2605.24523v1) | 01, 04 | text as an auxiliary contrastive term, weight 0.1 |
| ENIGMA, 2026 | [arXiv 2602.10361](https://arxiv.org/html/2602.10361) | 04 | small subject-specific layer trained on a short calibration set |
| AdaBN, Li et al. | [arXiv 1603.04779](https://arxiv.org/abs/1603.04779) | 04 | BatchNorm statistics re-estimated on the new subject, no labels |
| CSLS, Conneau et al., ICLR 2018 | [arXiv 1710.04087](https://arxiv.org/abs/1710.04087) | 01, 04 | hubness-corrected similarity at test time |

## Settings: notebooks 01-03, notebook 04 and the base papers

| | NICE (2024) | ATM (2024) | 01-03 | 04 |
|---|---|---|---|---|
| EEG encoder | TSConv | MLP + TSConv + channel attention | TSConv + projection | TSConv + projection; transformer layer + TSConv |
| Parameters | 102 k (encoder) | 3.18 M | 1.10 M | 2.63 M; 3.07 M |
| Visual target | CLIP ViT-L/14 | CLIP, 1024-d | CLIP ViT-B/32, 512-d | OpenCLIP ViT-H/14, 1024-d |
| Text | - | - | class-name prompt | BLIP caption |
| Training trials per subject | 16,540 (averaged) | 66,160 (single trials) | 16,540 (averaged) | 16,540 (averaged) |
| Optimizer, lr | Adam, 2e-4 | Adam, 3e-4 | AdamW, 3e-4, wd 1e-4 | AdamW, 3e-4, wd 1e-4 |
| Batch | 1000 | 1024 | 256 | 256 |
| Epochs | 200 | not stated | ≤ 40, early stop on val Top-5 (LOSO ≤ 20) | same |
| Precision, GPU | - | - | fp32, T4 | bf16, L4 |
| Within Top-1 (%) | 13.8 | 28.6 | 28.6 | 28.8; 35.0 with CSLS |
| LOSO Top-1 (%) | 6.2 | 11.9 | 11.2 | 10.3; 25.0 with 4,000 calibration images + refinement |

Base-paper values are as reported in the papers.

## Summary: notebooks 01-03 vs notebook 04

| Question | 01-03 | 04 |
|---|---|---|
| Within-subject Top-1 | 27.2 (own-statistics normalisation) | 28.8 |
| EEG to text retrieval | class-name text, no gain | EEG to caption 15.4% Top-1 (chance 0.5%) |
| New subject, no calibration | 11.2 | 10.3 |
| New subject, calibrated | Procrustes, no clear gain | 18.3 (fine-tuning, 4,000 images); 25.0 with refinement |
| Test-time refinement | MNN + CSLS in the rotation test only | CSLS: within 28.8 to 35.0, all 10 subjects improve |
| More complex encoder | - | transformer layer: lower within (24.8), unstable calibration |

## Results

### 1. Within-subject, sub-01, 3 seeds (`results/results_within_subject_run1.csv`)

| Variant | Top-1 (%) | Top-5 (%) |
|---|---|---|
| Previous prototype (2,000 samples, 1 conv layer, 5 epochs) | 0.0 | - |
| EEG to image | 22.7 ± 1.3 | 55.2 ± 2.5 |
| EEG to image + text (λ = 0.1) | 20.8 ± 0.8 | 53.8 ± 1.8 |

The class-name text regularizer gives no gain. A second run with the same seeds (`results_within_subject_run2.csv`) gave 22.8 ± 2.0 / 55.0 ± 3.5 and 21.2 ± 0.3 / 53.8 ± 1.6, the same conclusion.

![before / after](figures/fig1_before_after.png)

### 2. All 10 subjects, seed 0 (`results/results_all_subjects.csv`)

Top-1 28.6 ± 6.6%, Top-5 62.5 ± 7.8% (range 15.0% for sub-05 to 39.0% for sub-08).

![ten subjects](figures/fig4_ten_subjects.png)

### 3. 10 x 10 cross-subject matrix, no adaptation (`results/results_cross_matrix.csv`)

Model trained on subject i, tested on subject j. Diagonal 27.2%, off-diagonal 4.5 ± 2.0%.

![cross matrix](figures/fig5_cross_matrix.png)

Here every test set is normalized with its own statistics (needed for unseen subjects), which is why the diagonal differs from section 2. Normalization alone moves sub-05 from 15.0% to 24.0%.

### 4. Leave-one-subject-out (`results/results_loso.csv`)

Trained on 9 subjects, tested on the held-out one (3 folds so far).

| Held-out subject | Top-1 (%) | Top-5 (%) |
|---|---|---|
| sub-01 | 10.0 | 31.0 |
| sub-05 | 10.5 | 31.5 |
| sub-08 | 13.0 | 35.0 |
| Mean | 11.2 | 32.5 |

Training on nine subjects raises transfer from 4.5% (one training subject) to 11.2%. On the 100-way evaluation half, Procrustes calibration with 25, 50 or 100 labelled trials of the new subject gives no clear gain after LOSO (17.7% before; 19.0%, 17.7% and 17.7% after), whereas it helped in the one-subject setting (section 5). A rotation alone is not enough once the model is trained on many subjects.

![loso](figures/fig6_loso.png)

Reported values come from the cited papers and use larger encoders and visual targets; they are reference points, not a controlled comparison.

### 5. sub-01 to sub-02 calibration, 100-way (`results/results_cross_subject_sub01_sub02.csv`)

| Method | Top-1 (%) |
|---|---|
| within sub-01 | 31.3 |
| zero-shot on sub-02 | 7.7 |
| moment matching (label-free) | 10.3 |
| Procrustes, 100 labelled trials | 13.0 |

![cross-subject](figures/fig3_cross_subject.png)

### 6. SCORE-style sanity check (`results/results_score.csv`)

- Procrustes fix: R = U @ Vh (the prototype used U @ Vh.T).
- Full random rotation: 22.5% to 0%; recovered to 17.5% with 50 labelled pairs and fully with 250.
- Label-free MNN + CSLS helps at moderate angles (60°: 9.0% to 15.5%) but fails at 90°.

![score](figures/fig2_score.png)

### 7. Visual target layer, sub-01, seed 0 (`results/results_layers.csv`)

| Target | Within Top-1 | Within Top-5 | sub-01 model on sub-02, Top-1 |
|---|---|---|---|
| block 4 | 18.5 | 51.0 | 3.5 |
| block 6 | 22.0 | 59.0 | 6.5 |
| block 8 | 18.5 | 51.0 | 3.5 |
| block 10 | 22.0 | 54.0 | 6.0 |
| block 12 | 25.0 | 61.0 | 6.0 |
| final (projected) | 20.5 | 56.0 | 3.5 |

The last block before CLIP's image-text projection gives the best within-subject accuracy. Cross-subject differences are within noise (one image = 0.5%). One seed only; to be repeated.

![layers](figures/fig7_layers.png)

### 8. Choosing among a few objects

Expected accuracy when the candidate set is the true image plus k-1 random others (computed exactly from retrieval ranks). Block-12 target:

| Candidates | Chance | Within sub-01 | sub-01 model on sub-02 |
|---|---|---|---|
| 2 | 50% | 96.5% | 84.3% |
| 5 | 20% | 88.2% | 62.2% |
| 10 | 10% | 78.8% | 45.4% |

![k-way](figures/fig8_kway.png)

## Notebook 04: stronger target, tri-modal alignment, calibration, test-time refinement

Changes from notebooks 01-03: visual target CLIP ViT-H/14 (LAION-2B, 1024-d); one BLIP-large caption per image encoded with the CLIP text encoder; two EEG encoders, the same CNN (2.63 M parameters) and a channel-attention transformer layer in front of it (3.07 M parameters, ATM-style); channel z-score with each recording's own statistics; bf16 mixed precision.

### 9. Within-subject, 10 subjects (`results/results_within_v2.csv`)

| Encoder | Target | Text | Top-1 (%) | Top-5 (%) | EEG to caption Top-1 (%) | EEG to class name Top-1 (%) |
|---|---|---|---|---|---|---|
| CNN | ViT-H/14 | none | 28.8 ± 5.5 | 61.0 | 14.1 ± 3.0 | 10.4 |
| CNN | ViT-H/14 | caption, λ = 0.1 | 28.1 ± 5.3 | 60.7 | 15.4 ± 2.8 | 11.9 |
| transformer + CNN | ViT-H/14 | none | 24.8 ± 8.1 | 55.5 | 12.4 ± 3.9 | 11.6 |

Mean ± SD over 10 subjects, 200-way, chance 0.5%. The same EEG embedding retrieves the seen image and its caption. Training with captions raises EEG-to-caption retrieval (14.1 to 15.4) at a small cost in image retrieval (28.8 to 28.1). The transformer layer lowers accuracy and increases spread across subjects at this data size.

![within vs papers](figures/fig11_within_vs_papers.png)

### 10. Leave-one-subject-out with calibration (`results/results_loso_v2.csv`)

Trained on 9 subjects; the held-out subject is calibrated with k of its own training-session images (training concepts only; the 200 test concepts stay unseen). AdaBN uses the same subject's unlabelled EEG.

| Method (CNN) | k | Top-1 (%) | 5-way (%) |
|---|---|---|---|
| none | 0 | 10.3 ± 2.0 | 69.9 |
| AdaBN, no labels | 0 | 10.7 ± 2.0 | 70.0 |
| channel adapter (63 x 63) | 4,000 | 14.7 ± 1.9 | 77.8 |
| fine-tuning | 1,000 | 15.7 ± 5.5 | 76.7 |
| fine-tuning | 4,000 | 18.3 ± 2.8 | 79.5 |

Mean ± SD over sub-01, sub-05, sub-08. Fine-tuning with 4,000 images raises Top-1 in every held-out subject (11.5 to 16.5, 8.0 to 17.0, 11.5 to 21.5). The transformer + CNN encoder reaches 16.5 ± 9.6 with the same calibration, with much larger spread across subjects. Label-free AdaBN alone gives no clear gain.

![calibration](figures/fig9_calibration.png)

![new subject](figures/fig10_new_subject.png)

Reported values are zero-shot results from the cited papers with different encoders and training data; they are reference points, not a controlled comparison.

### 11. Choosing among a few objects, new subject

![k-way new subject](figures/fig12_kway_new_subject.png)


### 12. Test-time refinement, no labels (`results/results_refine.csv`)

The trained encoder is unchanged. Two corrections of the similarity scores, computed only from the subject's unlabelled training-session EEG (no test labels, no test EEG beyond the query itself):

- centering: subtract the subject's mean EEG embedding and the mean training-image embedding, then re-normalise;
- CSLS (Conneau et al., ICLR 2018): `score = 2·cos(z, y) − r(y)`, where `r(y)` is the mean similarity of image `y` to its 10 nearest EEG embeddings of that subject. This penalises "hub" images that are close to many EEG embeddings.

| Setting | Plain | Centering | CSLS | Centering + CSLS |
|---|---|---|---|---|
| Within-subject, 10 subjects | 28.8 ± 5.5 | 29.7 ± 6.0 | 35.0 ± 6.1 | 35.4 ± 6.6 |
| New subject, no calibration (3) | 10.3 | 10.5 | 11.7 | 12.3 |
| New subject, fine-tuning 1,000 images (3) | 15.7 | 15.2 | 19.2 | 19.3 |
| New subject, fine-tuning 4,000 images (3) | 18.3 | 17.3 | 23.3 | 25.0 |

Top-1 (%), 200-way. CSLS improves every one of the 10 subjects within-subject (+2.0 to +10.0). On a new subject it helps most after calibration: with 4,000 calibration images the new-subject accuracy (25.0) approaches the within-subject accuracy of the same three subjects without refinement (26.3). The "plain" column repeats the earlier runs exactly (same seeds).

![csls within](figures/fig14_csls_within.png)

![refinement new subject](figures/fig13_refinement_loso.png)

### 13. ATM training recipe: single trials and batch 1024 (`results/results_single_trials.csv`)

ATM trains on all single trials (66,160 per subject) with batch 1024; notebooks 01-04 use the average of the 4 repetitions (16,540) and batch 256. Single trials are scaled with the statistics of the averaged training data so that they match the test scale.

| sub-01 / 05 / 08, Top-1 (%) | Plain | CSLS | Centering + CSLS |
|---|---|---|---|
| averaged repetitions, batch 256 | 26.3 | 33.8 | 35.0 |
| single trials, batch 1024 | 27.0 | 35.8 | 36.8 |

Four times more training samples and a four times larger batch change accuracy by less than one image in 200 without refinement, and by about 2 points with CSLS. The training recipe is not what separates our baseline from ATM; the test-time correction is the larger effect.

## Limitations

- LOSO on 3 of 10 folds and one seed; layer ablation on one subject and one seed.
- Calibration still needs 1,000-4,000 labelled images to help; 250 gives no gain.
- Test-time refinement uses the subject's unlabelled training-session EEG, so it needs a short unlabelled recording of the new subject.
- 100-way calibration numbers use 100 test images (one image = 1%).

## Next steps

1. LOSO on all 10 folds with 3 seeds.
2. Calibration with fewer labelled images (label-free test-time refinement, subject layers trained jointly with the shared encoder).
3. Posterior channel selection and a blurred visual target (the main gains reported by UBP, CVPR 2025).
4. Fixation-locked EEG with eye tracking (Phase 2).
