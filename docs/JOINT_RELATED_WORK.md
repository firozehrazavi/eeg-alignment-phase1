# Joint EEG–Image–Text: literature and comparison boundaries

Checked 2026-11-17 against primary papers available on that date. These are author-reported results, not replications. This targeted review is not an exhaustive novelty certification. It extends the completed repository audit; it does not reopen historical evaluation data.

## Benchmark comparison

Top-1/Top-5 below are percentages. The common task is a 200-image gallery from held-out THINGS concepts. A common gallery size alone does not establish matched preprocessing, model selection, repetition averaging, or encoder capacity.

| Method | Within subject | LOSO | Mechanism and protocol evidence |
|---|---:|---:|---|
| RPA | 95.4 / 99.8 | 35.5 / 67.4 | Adapts full intermediate OpenCLIP H/14 patch grids before pooling; 10 subjects, three runs; 4 training and 80 test repetitions averaged. Uses 17 posterior channels; within-subject window 0–700 ms, LOSO 0–1000 ms. **Layer selection used test accuracy.** [RPA v2, Table 1; Appendices A and D](https://arxiv.org/html/2609.31698v2). |
| SAMGA | 91.3 / 98.8 | 34.4 / 64.8 | InternViT-6B layers 20/24/28/32/36; global routing plus training-time subject residuals; shared encoder and staged MMD/contrastive alignment. Five seeds, 10 subjects; 17 within-subject and 63 LOSO channels; test repetitions averaged. [SAMGA v2, §§3–4](https://arxiv.org/html/2604.17782v2). |
| DTHA | 92.4 / 99.5 | 25.6 / 55.0 | Multi-blur CLIP and shallow DINOv2 teacher branches, similarity fusion, branch and fused contrastive losses. Ten subjects, seeds 2025/42/1234; 5% random training validation. Direct language inputs are absent. [DTHA v1, §§III–IV, Table II](https://arxiv.org/html/2611.12798v1). |
| NVOL | 78.1 / 96.7 | 12.7 / 29.2 | Intermediate-layer fusion; with CSLS: **86.4 / 98.5**, LOSO **19.9 / 44.1**. CSLS uses a global query–gallery matrix, including query-neighborhood statistics. [NVOL v1, §III-B, Table II](https://arxiv.org/html/2609.02582v1). |
| ProtoEEG | 64.0 / 91.5 | 22.0 / 48.4 | RN50 family; soft assignments over 64 learned prototypes, prototype contrast, distribution alignment and instance anchoring. Five-seed subject means over 10 subjects. CLIP-family results are lower and reported separately. [ProtoEEG v1, Table 1, Appendix A](https://arxiv.org/html/2610.33065v1). |
| AttrEcho | 94.84 / 99.7 | Not established here | Explicit regional visual attributes, signed residual feature corrections and learned regional aggregation. Also reports 82.06% Top-1 with 1,600 candidates containing same-concept distractors. [AttrEcho v1, §§3–4](https://arxiv.org/html/2611.09123v1). |
| MindAlign | 54.1 / 83.4 | Not established here | Masked EEG pretraining followed by EEG–image and image–text contrastive objectives; text is a mild shared-space regularizer. [MindAlign v1, §3.3 and abstract](https://arxiv.org/html/2605.24523v1). |

**RPA comparability:** Its task matches the historical 200-way, 80-repeat-averaged evaluation in broad terms. Appendix D explicitly states that layer 11 was chosen by maximum EEG test Top-1 over a depth sweep. Validation-selected checkpoints in Appendix A.4 do not remove that upstream test selection. Its reported uncertainty is the standard deviation of three participant-averaged run means, not a confidence interval across participants. Therefore 95.4% is a useful published reference, but not a verified untouched-test target. RPA is the highest point estimate among the primary reports verified here; this is not a universal SOTA certification. [RPA v2, Appendices A.4 and D](https://arxiv.org/html/2609.31698v2).

Additional qualifications:

- SAMGA is particularly relevant to the existing five-layer InternViT cache. Its source sections inspected here do not establish a fully nested subject-independent hyperparameter-selection procedure. [SAMGA v2, §4.2](https://arxiv.org/html/2604.17782v2).
- DTHA says the official test split is reserved for final evaluation, yet its teacher-depth discussion uses representations of the 200 test images to motivate shallow DINO layers. This is test-image-informed analysis, not demonstrated test-EEG label optimization. Exact repetition aggregation should be confirmed from a pinned implementation before a controlled reproduction. [DTHA v1, §§IV-A and IV-C3](https://arxiv.org/html/2611.12798v1).
- NVOL's subject-specific single-layer maxima are selected from the reported layer sweep. An independent development split for that selection was not established from the inspected paper. Its CSLS row must be distinguished from query-independent cosine ranking. [NVOL v1, §§III-B and IV-C3](https://arxiv.org/html/2609.02582v1).
- ProtoEEG's main training description specifies 100 epochs, batch 128 and dimension 512; Appendix A.3 specifies 300 epochs, batch 1024 and dimension 1024. These inconsistencies and unverified repetition aggregation prevent treating its paper alone as an exact reproduction specification. [ProtoEEG v1, main experimental setup and Appendix A.3](https://arxiv.org/html/2610.33065v1).

## Closest three-modal and interaction precedents

| Prior work | Established contribution | Boundary for this project |
|---|---|---|
| BraVL | Joint brain–visual–linguistic latent inference through a mixture of products of experts, with intra/inter-modality mutual-information regularization. [Du et al., TPAMI 2023](https://arxiv.org/abs/2210.06756). | Three-modality joint learning, missing-modality latent inference, and a joint probabilistic formulation are already established. |
| MindAlign | Weighted EEG–image and image–text contrastive training; EEG–text alignment emerges through the image representation. [§3.3](https://arxiv.org/html/2605.24523v1). | Adding text regularization or claiming a shared tri-modal space is insufficient novelty. |
| SeeEEG | Its indexed primary abstract describes EEG alignment with image/text embeddings and retrieval of image–text pairs for generation. Full PDF access failed during this review. [ICCVW 2025 primary PDF](https://openaccess.thecvf.com/content/ICCV2025W/HCV/papers/Kim_SeeEEG_Semantic-aware_EEG-based_Multi-Modal_Retrieval-Augmented_Generation_for_High-Fidelity_Visual_Brain_ICCVW_2025_paper.pdf). | Multimodal retrieval and text-supported EEG decoding are established; a full formula comparison remains open. |
| NeuroAlign, EEG submission | The indexed OpenReview paper describes dynamic EEG/visual/language interaction, BLIP-2 captions, image–text contrast, KL alignment and adaptive semantic guidance. Full text was blocked by browser verification. [OpenReview primary PDF](https://openreview.net/pdf?id=VExNUuQAWq). | An unresolved close predecessor: do not claim the first dynamic semantic interaction. Do not confuse it with the separate MICCAI 2026 fMRI paper sharing the name. |
| Tensor Fusion Network; Low-rank Multimodal Fusion | Explicit high-order interactions and efficient factorized multimodal products. [TFN, EMNLP 2017](https://aclanthology.org/D17-1115/), [LMF, ACL 2018](https://aclanthology.org/P18-1209/). | A sum of products of three projected vectors is an established CP-style tensor parameterization. Rebranding it as a new three-modal mechanism is not defensible. |
| AttrEcho | Tests attribute information beyond CLIP through partial RSA and injects regional attributes as residual visual corrections. [§§3.2–3.4](https://arxiv.org/html/2611.09123v1). | Complementarity beyond a frozen visual representation and residual preservation are already motivated in EEG decoding. A text innovation mechanism must distinguish its conditioning, learning objective and direct text inference. |

## Defensible hypothesis, with limits

The research question supported by this review is narrower than “can text help EEG?”: **does text contain EEG-predictable structure not captured by this fixed visual cache and predictor class, and can a conditional interaction exploit it while preserving image retrieval?** This is a proposed hypothesis, not an established finding or novelty claim.

For a training-only visual-to-text predictor \(m\), the residual \(r=t-m(v)\) removes what that predictor explains. This does **not** prove information beyond the image: if captions are generated from pixels, their information can already be present in the original image. Linear residualization is orthogonality to a linear model on its fitting sample, not statistical independence or conditional mutual-information estimation. Fit preprocessing and nuisance prediction within training folds; cross-fitting avoids evaluating fitted residuals as though they were out-of-sample discoveries. A high residual norm can reflect caption noise or predictor underfitting.

A factorized score \(\sum_k a_k(e)b_k(v)c_k(r)\) has non-additive three-way dependence when all factors are active. Its mathematical interaction is testable, but the factorization itself is prior art. The plausible contribution would be the **specific conditional objective, treatment of redundant caption information, baseline-preserving learning, and evidence that meaningful text changes generalization beyond matched controls**. These elements need an explicit ablation and a fuller comparison with NeuroAlign before a strong novelty statement.

Direct EEG→caption retrieval must rank texts without candidate-image features. Computing \(t-m(v_{candidate})\) at inference violates that requirement. Valid designs include an EEG-only text head trained by the joint mechanism, or a clearly disclosed marginalization over training-derived latent visual anchors. These are implementation alternatives, not additional proposed research methods. Direct retrieval must be evaluated separately from image-assisted caption ranking, and equivalent captions need semantic-positive accounting.

The decisive language control compares the same parameterization, budget, splits and initialization with correct versus identity-shuffled text and with a capacity-matched nonsemantic substitute. Improvement over the unmodified baseline alone does not attribute a gain to language. Report image and text galleries separately; average seeds within each subject before subject-level paired uncertainty. Keep previously examined development concepts identified as such, and record benchmark exposure rather than calling the official test pristine.

No new EEG result, numerical improvement, or SOTA claim is established by this review.

## Engineering verification and resumable comparison

`JointAligner.load_06b_backbone_state` accepts the plain `Aligner.state_dict()`
format emitted by the 06b source and validates keys, shapes, and finite values
before transfer. CPU tests compare transferred EEG and image embeddings exactly.
This verifies architecture/state-dict compatibility, **not** loading a trained
historical checkpoint: no 06b checkpoint tensor file is present in this
repository. Do not initialize a clean split from weights trained on its held-out
images.

The single Colab entry point uses the matched A/B/C/D training-only comparison
and a stable Drive output directory. Its `--resume` checkpoints are epoch-boundary
continuations: model, optimizer, scheduler, selection state and RNG state are
restored only when the config, manifest hash, source-code hashes, and Python /
PyTorch / NumPy versions match.
Re-run setup and training cells after a runtime restart; the manifest is reused
only when present, and input hashes are checked by the training loader. The
development holdout remains for final reporting, never tuning; no official test
data is loaded by the comparison.
