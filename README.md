# EEG to Image Retrieval on THINGS-EEG2

PhD project by Firouzeh Razavi. Supervisors: Dr. Leila Safari and Dr. Mohsen Afsharchi.

The goal is to find which image a person was looking at, using only their EEG. We test on THINGS-EEG2 with 10 subjects. For each test EEG the model has to pick the right image out of 200, so random guessing gives 0.5%. All numbers below are Top-1 accuracy.

## Baseline

I rebuilt the baseline following the 2026 papers. The image side uses features from the middle layers of InternViT-6B, which stays frozen. The EEG side is a small MLP that is trained to match these features with a contrastive loss.

| | Same subject | New subject | New subject with label-free correction |
|---|---|---|---|
| Earlier version, 28 Oct | 28.8 | 11.2, 3 folds only | - |
| DINOv2 features, notebook 06 | 70.5 | 21.6 | 35.3 |
| InternViT features, notebook 06b | 78.8 | 26.2 | 46.4 |
| Shallow Alignment, 2026 paper | 82.6 | 22.4 | - |
| SAMGA, 2026 paper | 91.3 | 34.4 | - |
| SAMGA + SCORE, 2026 paper | - | 26.2 | 48.8 |

Notes:
- Same subject means training and testing on one person. The result is averaged over 10 subjects and 3 seeds.
- New subject means training on 9 people and testing on the one left out. All 10 folds were run.
- The label-free correction uses the unlabelled test EEG of the new person to adjust the embedding. It follows the SCORE method.
- Numbers from papers are as reported by their authors. Their setups are not exactly the same as mine.
- Middle layers work better than the last layer. Layer 28 out of 45 was the best single layer.

## The main problem: new users

A model trained on other people works much worse on a new person: 79% for a known person, but only 26% for a new one. This gap appears in all recent methods.

What I found:
- Better choice of matching points for the label-free correction helps very little. Even perfect points add only 4%.
- Any simple linear correction stops at about 50%.
- Using a few labelled images in the usual way can make things worse. With only 10 images, plain Procrustes drops to almost zero for some subjects.

## Short calibration for a new user

The new user looks at k images before using the system. We know which images were shown, so these give us labelled examples for free. Each image takes about 0.8 seconds, so 100 images take a bit more than one minute. Only the EEG of these k images is used, not the test EEG.

I tested a simple method called SHRINK. It changes the model very little when there are only a few images and more when there are many.

| Calibration time | 0 | 8 s | 20 s | 40 s | 1.3 min | 3.3 min | 13 min |
|---|---|---|---|---|---|---|---|
| Number of images | 0 | 10 | 25 | 50 | 100 | 250 | 1000 |
| No calibration | 26.4 | | | | | | |
| Plain Procrustes | | 31.0 | 29.9 | 33.8 | 40.6 | 48.9 | 54.3 |
| SHRINK | 37.4 | 38.5 | 40.0 | 42.2 | 44.0 | 48.6 | 53.2 |
| SHRINK plus unlabelled test EEG | 43.6 | 43.8 | 44.3 | 45.4 | 46.4 | 49.6 | 53.1 |

What this shows:
- Plain Procrustes is worse than no labels for 9 of 10 subjects when only 10 to 25 images are used.
- SHRINK never gets worse than no labels from 25 images on.
- Between 25 and 100 images, SHRINK is clearly the best. At 50 and 100 images it is better for all 10 subjects.
- With many images SHRINK is too careful, and plain Procrustes is better at 1000 images.
- The value at 0 images is already higher than 26.4 because the scoring step corrects for images that match too many queries.

Similar ideas have been published before: Ouali et al., ICCV 2023, and Cao et al., Signal Processing 2027. Notebook 09b compares them directly on the same data and the same calibration images.

| Number of images | 5 | 20 | 50 | 100 | 200 | 1000 |
|---|---|---|---|---|---|---|
| No change, moment matching only | 37.3 | 37.5 | 38.3 | 38.7 | 39.3 | 40.0 |
| Plain Procrustes | 1.5 | 5.9 | 13.3 | 23.0 | 32.8 | 53.2 |
| Ouali 2023, strength chosen by cross-validation | 37.1 | 36.9 | 39.8 | 41.9 | 44.9 | 54.5 |
| Cao 2027, fixed strength 0.25 | 37.3 | 39.1 | 41.4 | 42.6 | 44.4 | 47.8 |
| SHRINK | 37.4 | 38.4 | 41.0 | 44.0 | 47.2 | 53.6 |

- Between 20 and 200 images SHRINK is 1.2 to 2.3 points better than Ouali and wins for 6 to 8 of the 10 subjects.
- With 1000 images Ouali is slightly better.
- With 100 images the weakest subject gets 35.8 with SHRINK and 32.2 with Ouali.
- The part that matters is letting the strength grow with the number of images. Changing only this part gives almost the same result as full SHRINK.
- These differences are small. They are not yet significant after correcting for testing many values of k, so I report them as a consistent trend, not a final claim.

## Knowing when to trust the answer

The system can also say how sure it is. I use how much the best image wins over the second one, and whether four different models agree.

| Answer only the most confident | 25% | 50% | 75% | all |
|---|---|---|---|---|
| Accuracy | 76.2 | 62.7 | 53.7 | 44.7 |

So when the system is confident, it is right about three times out of four. Adding the agreement between models helps in 9 of 10 subjects.

## Notebooks

| Notebook | What it does |
|---|---|
| 01 to 04 | earlier prototype with CLIP |
| notebooks/06 | baseline with DINOv2 features |
| notebooks/06b | baseline with InternViT features |
| notebooks/07 | how to pick matching points for the label-free correction |
| notebooks/08 | limits of the correction, first calibration tests, confidence |
| 09 | fair calibration test for new users |
| 09b | comparison with Ouali 2023 and Cao 2027 on the same data |

Result tables are in the results folder. Everything runs on Google Colab.

## Next steps

- Make the baseline stronger with ideas from SAMGA: two training stages, separate layer weights, a convolutional EEG encoder for new subjects, and a loss that uses several subjects at once.
- Run the calibration test again on the improved model.
- Add Euclidean Alignment as a classic baseline.

## Main references

SAMGA, arXiv 2604.17782. Shallow Alignment, arXiv 2601.21948. SCORE, arXiv 2608.19134. SATTC, CVPR 2026. Ouali et al., Black Box Few-Shot Adaptation for Vision-Language Models, ICCV 2023. Cao et al., Signal Processing 250, 2027. Gifford et al., THINGS-EEG2, NeuroImage 2022.
