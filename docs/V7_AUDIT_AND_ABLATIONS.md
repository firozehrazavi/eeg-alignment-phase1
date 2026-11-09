# Supplied v7: audit and controlled idea adoption

Inspected `10_sota_upgrade.zip` from the user, SHA256
`b360992d8a76fd8dbdc25dc3cfe294fddec21763030adcb1fd73c70d3a8e6b07`.
It contains one notebook, `10_sota_upgrade.ipynb`. All eight code cells have
null execution counts and no stored output blocks. It provides no observed
accuracy evidence. Predictions such as 90–93% within-subject or 52–60% LOSO
are not results and are not adopted here. The archive was read, not executed.

The RPA-inspired **patch-token** experiment remains primary. v7 consumes pooled
InternViT means, not native patches, so its ConvEEG ideas do not replace the
patch adapter. The existing historical 78.8% baseline and all CSVs remain intact.

## Findings and disposition

References below use zero-based notebook cell index and one-based source line.

| Finding | Evidence | Correction or restriction |
|---|---|---|
| LOSO validation examples train the model | Cell 5 line 84 uses `np.ones(len(X))` whenever pooled X has more than N_TR rows. Cell 7 explicitly validates `VAL + i*N_TR`. | Build source-major training rows solely from concept-disjoint fit image IDs. `source_training_rows` rejects overlapping fit/validation rows and target subjects. The runner never opens held-out training EEG. |
| Filtered sampler positions are mistaken for global rows | Cell 5 lines 101, 111–113: `stratified(ids[rows],...)` returns positions in the filtered array, then `Xs[b]` indexes the unfiltered tensor. This affects pooled pretraining with a filtered mask; LOSO's all-true mask hides the second bug while retaining the first. | `stratified_rows` returns original allowed row IDs and visits each exactly once. Regression tests include holes in both subjects' allowed rows. |
| Validation is not concept-disjoint | Cell 2 line 75 retains the last image in each concept; the other images from that concept train the model. | Reuse authoritative concept IDs and the corrected common concept split. No inferred `N_TR // n_concepts` indexing. |
| Normalization leaks evaluation distribution | Cell 3 loads `{subject}_{split}_z16.npy`; prior repo audit shows train/test session statistics were fitted separately. Cell 3 line 5 normalizes visual targets over all training rows, including validation. | The new preparation path uses pre-z-score distributed EEG and train-concept visual statistics. Distributed upstream whitening still needs audit: classify as `benchmark_preprocessed`, not strict induction. Strict mode refuses that provenance. This cannot be repaired merely by relabeling old z16 arrays. |
| Pooled-source false negatives | Cell 5 `nce` uses diagonal CE even when the same stimulus appears across subjects. A margin is subtracted only on the diagonal. | Corrected v7 controls use exact-ID multi-positive loss. Optional margin subtracts from **every matching identity**, with rectangular/permutation-safe positives. Primary patch diagonal comparison remains an explicitly named historical-objective arm. |
| A0 changes when a text cache happens to exist | Cell 1 line 54 sets LAM_TEXT=.3; cell 2 enables HAS_TEXT from a file; cells 6/7 set the training weight automatically. | No text file is discovered or loaded by the experimental runner. Nonzero `text_weight` is rejected. Text must remain a separately implemented and approved semantic experiment. |
| A0 is not an exact historical replay | Cell 5 creates fixed batch memberships and shuffles batch order, whereas 06b reshuffles rows each epoch. Evaluation cadence also differs; optional text and later selection of the best test-scoring ladder config add further confounds. | A0 here means the original MLP/route_c architecture under the **corrected** protocol, not a promised 78.8% replay. Saved configs, splits, seeds and provenance define comparability. Existing AST architecture/loss tests remain unchanged. |
| Retrieval identities and ties are unsafe | Cell 3 line 30 compares each diagonal score with strict `>`; all ties score Top-1 and reordered galleries give wrong answers. | Reuse explicit-ID 200-way Top-1/Top-5 metrics, unique image/content checks, missing-positive rejection and deterministic tie handling. |
| Stage ladder changes several factors | S2 changes LR, batch size, margin and augmentation together; A4 adds other-subject data, augmentation, MMD and centroid terms. | Separate Conv-only, noise-only and margin-only configs. Keep every other field identical to corrected A0; tests verify exact configuration differences. |
| Ensemble membership silently varies | Cell 8 skips missing files and averages whatever exists. No validation-selected weights are fitted; summary highlights best test accuracy. | Export source-validation score bundles; select among predeclared weights using source validation only. Require every member/source and align both query/gallery IDs before combining. Reject target/test selection records and mixed score types. |
| Claimed MMD term is not MMD squared | Cell 5 line 42 uses `kxx + kyy - kxy`, missing the factor 2 on the cross expectation. | Do not port this term. It would need its own corrected implementation and ablation. Subject-centroid penalties and per-subject image routing are also deferred. |
| Uncontrolled automatic full runs | Cells 6/7 call all within and LOSO configurations at top level; output existence alone controls skipping. | Do not execute the supplied notebook. The Pro+ notebook has explicit default-false paid-stage approvals and no full-run launch cell. |

## What was implemented

`phase1/v7_controls.py` adds a sample-local ConvEEG variant: odd-kernel temporal
convolution, depthwise spatial convolution, separable temporal convolution,
one temporal attention block, adaptive pooling, and the existing residual EEG
projection head. GroupNorm replaces v7's BatchNorm; no test-batch running
statistics are fitted. This is an explicit variant of the idea, not a faithful
v7 architecture reproduction. Flattening EEG in A0 preserves its information;
the convolutional encoder offers a different inductive bias, not guaranteed gain.

The module also adds globally indexed source training/stratified sampling,
explicit EEG augmentation, an identity-aware additive-margin objective, and
source-validation ensemble selection. The corrected runner exposes these only
through explicit configuration. The default patch pilot still uses the original
MLP, no augmentation, no margin and no text.

| New config | Only change versus corrected A0 | Status |
|---|---|---|
| `v7_A0_corrected.json` | Control: MLP, pooled route_c, multi-positive, no text/augmentation/margin | Prepared; not run on real EEG |
| `v7_A1_conv_only.json` | EEG encoder = ConvEEG | Prepared; not run on real EEG |
| `v7_B1_noise_only.json` | Training-only Gaussian noise, SD .2 after source-fitted z-score | Prepared; not run on real EEG |
| `v7_C1_margin_only.json` | Margin .15 from epoch 4 of 5, applied to all positives | Prepared; not run on real EEG |

All four use one subject, seed 0, the same five-epoch optimizer/schedule/batch
and 200-way test gallery. Each arm needs a distinct output directory. They use
the common prepared dataset and runner; the visual arm is `baseline` (pooled),
even though the shared preflight checks the primary pilot's token-cache identity.
The full token cache already exists after the primary patch pilot; no second
extraction is needed for these controls.

The margin-only schedule is a narrow coarse-to-fine **ablation**, not SAMGA's
complete training method. Cutout and channel-drop functions are unit tested but
have no automatic combined recipe. Additional ablations should change one
factor at a time. Selection criteria must be specified before inspecting test
results; the default maximum epoch count and all numerical settings are fixed
for this pilot, not selected from the target test set.

## Ensembling without test selection

With `save_validation_scores: true`, the runner saves
`*_source_validation.pt` (subject -> explicit-ID cosine score bundle) and
`*_test_bundle.pt` separately. Supply the former to `select_score_ensemble`,
along with a fixed candidate list such as the two single models and their
equal-weight average. For LOSO, its `sources` must be exactly the outer source
subjects and `target` must be excluded. Never pass test bundles as validation.
Freeze the resulting weights, then use `apply_score_ensemble` on test bundles.
Source-validation Top-1 used for selection is not a test accuracy claim.

No search over outer target scores, automatic best-test ensemble, hidden missing
members, transductive score mixture or automatic seed selection is implemented.
The utility is tested; no real-data ensemble has been fitted or evaluated.

## Deferred until after the primary real-data pilot

Source-subject pretraining before within-subject fine-tuning is potentially
useful but adds data and compute. Do not combine it with ConvEEG/augmentation as
the first test. A future separate ablation must hold the fine-tuning recipe
fixed, exclude validation concepts from every pretraining source, and distinguish
additional training compute from architecture effects. In strict LOSO, the
outer target must be excluded from both training and hyperparameter/checkpoint
selection; any pretraining selection requires inner source subject folds.
That pretrain/fine-tune orchestration is **not implemented** in this change.

The next real experiment remains the approval-gated RPA-inspired patch pilot.
After it passes, review validation/resource evidence and request approval for
corrected A0 versus Conv-only. Noise and margin arms follow separately if useful.
Do not launch the ten-subject/three-seed schedules without user approval.

No real EEG or feature cache is accessible in this checkout, so no real pilot
was executed. Required Drive inputs are listed in `PATCH_EXPERIMENT.md`.
All reported validation in this change is synthetic software testing, not a
retrieval improvement or evidence for the suggested 90–93% / 52–60% ranges.
