# Low validation accuracy: root-cause audit (training/validation only)

## Decision and evidence boundary

**NO-GO for final-test evaluation or further training until the alignment and
validation diagnostics below are reviewed.** This is a diagnostic gate, not a claim
that 9% is unusable or that more epochs will fix it. No optimizer/model/protocol
change is implemented in this audit. The only additions are documentation, a
read-only CPU diagnostic module/notebook, and synthetic tests.

User-observed: Baseline best epoch5 =8.5%, Mean epoch4 =9%, Patch epoch3 =9%,
200 validation candidates, 15 histories verified. Training loss decreases strongly;
exact loss trajectories were not supplied here. Private Drive arrays and checkpoints
are not accessible locally. No final-test notebook, identity list, score file or
current final-test result has been opened. Historical comparison numbers below are
quoted from the existing audit documentation, not newly read CSVs. Notebook06b was
inspected as source text only (no outputs, execution or data loading).

Training runtime pinned code: 1958251c2af98d8d065db9b84a7801b433effa40. Source
references below use repository-relative files and one-based lines; notebook
references are zero-based cell / one-based source line, independent of JSON layout.

## Prioritized findings

| Priority / severity | Finding | Status | Impact / evidence | Recommended action |
|---|---|---|---|---|
| 1 / high | Historical 78.8% and current 9% measure different protocols/data budgets | Verified design mismatch | 06b c2:27, c5:55–80 versus patch_experiment:135–202; historical score is a documented outer evaluation, current value is validation | Preserve definitions; compare only matched arms, not headline percentages |
| 2 / high | Validation asks exact-image discrimination among ten siblings per unseen concept | Verified code; pool multiplicities to confirm on Drive | core:42–72; patch_experiment:192–202; 20 concepts x10 images implies a hard within-concept problem | Report same-concept error concentration as a diagnostic alongside unchanged exact-image Top-1 |
| 3 / high | Small selected pilot and large trainable heads | Verified capacity/data imbalance; overfitting is a hypothesis | 1800 fit images versus 8.68–21.79M trainable parameters; five epochs | Inspect train/validation gap and temperature before changing budget/architecture |
| 4 / high | EEG-to-stimulus ordering is assumed upstream, not independently certified by EEG labels | Verified assurance gap, NOT a demonstrated mismatch | patch_prepare:78–95,141–150; EEG dict supplies data/channel/time, not checked trial stimulus IDs | Independently reconcile training-only upstream row manifest; do not permute rows speculatively |
| 5 / medium | Only 145 optimizer steps; small negative pool and rapidly decaying LR | Verified design; underfitting or schedule mismatch unproven | patch_experiment:220–268, config patch_pilot | Predeclare a one-factor schedule/batch study only after diagnostics; no automatic extension |
| 6 / medium | Final-test catalog/token means are touched before the old train-only branch | Verified access-boundary weakness, not evidence of test-guided tuning | patch_experiment:87–105,139–145,286 | Future separate training-only loader should not read those artifacts; do not run it now |
| 7 / medium | Patch token normalization may amplify within-image spatial variance | Hypothesis, not a proven defect | patch_experiment:136–139,181–184; patch_adapter:20–31 | Measure token/mean RMS and normalization floors on training rows; do not silently change normalization |
| 8 / medium | Best-of-five, single-subject/seed scores have high uncertainty | Verified limitation | 17 vs18 successes; no paired predictions provided | Obtain paired validation predictions and concept-cluster uncertainty; no superiority claim |
| 9 / medium | Loss can improve through temperature and memorization without new retrieval ranks | Mechanism verified synthetically; contribution in this run unknown | core:15–20; patch_adapter:50–53 | Inspect checkpoint temperature, validation ranks, train-only geometry; more epochs are not a diagnosis |
| 10 / low | No recorded gradient norms, LR-by-step, train full-gallery retrieval or per-query validation bundles | Verified instrumentation gap in this config | patch_experiment:264–285, config save_validation_scores absent | Recover what is available from checkpoints; propose future lightweight telemetry separately |

No confirmed off-by-one, wrong diagonal target, detached visual gradient, or
train/validation row overlap was found in this runner. This does not prove private
array alignment. Equality of filenames/counts and a successful training run are not
independent stimulus/EEG identity evidence.

## 1. Row lineage and preprocessing

`patch_prepare.prepare` (43–151) obtains ordered image filenames/concept strings
from original image metadata, checks filename sequence against meta_v6.pt, selects
200 concept IDs by seeded permutation, and preserves original ascending image-row
order in `selection['train']` (87). For selected original row j:

1. Raw EEG row j contains repetitions x63 channels x250 samples.
2. `x[rows].mean(1,dtype=float32)[:,POST17]` (150) averages the four training
   repetitions, selects channels46..62, and writes the same selected-row order.
3. Each pooled layer indexes `x[selection[s]]` (120–123); catalog row is created
   from metadata filename/concept at that same j (94–95).
4. Extraction iterates catalog row indices; its hook removes CLS and retains the
   1024x3200 patch grid (`patch_cache.extract`,106–145). Completion flags are per row.
5. Training row r maps to image r%n (`patch_experiment`,168–170,247) with n=2000.
   For sub-01 only, fit rows are already 0..1999, so modulo does not shift them.
6. Batch EEG and visual paths use the same `images` sequence. Targets arange(B)
   are correct for the shuffled unique-stimulus batch. Validation uses the same
   val ordering for EEG, image embeddings and explicit identity list.

Image IDs remain filename strings; concept IDs are metadata strings, not inferred
contiguous arithmetic labels. `validate_catalog` rejects repeated IDs/content
hashes; `source_training_rows` excludes validation and rejects malformed partitions.
Historical image indexing used basename `setdefault` (06b c2:12–22): a basename
collision could have selected an unintended file. Current duplicate-content checks
catch some consequences, not every semantic mismatch. Likewise, cached feature
rows are trusted to have been produced in their advertised filename order. The
sampled L28 token-mean comparison checks vision consistency but cannot prove EEG
rows correspond to those images. No current evidence establishes a collision.

The 17 channels are P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2
(`patch_prepare`:17; `pilot_inputs`:19). Input is 17x250 =4250 values. The documented
legacy time vector is 300 points -0.2..0.996s; upstream cropped EEG by50 samples but
retained metadata. Effective EEG is 0..0.996s, no second crop. See
[time-axis evidence](CPU_TIME_AXIS_FIX.md) and `pilot_inputs.validate_eeg_times`:125.
Normalization fits per-channel moments over **fit images and all time samples**
(`fit_eeg_norm`:61–77), after repetition averaging. It does not fit on validation.
Visual per-dimension mean/std fits only fit rows (`Standardizer`:138; runner:136).
No augmentation is active. Raw repetition averaging removes trial variability; it
is not single-trial training or augmentation. Whether averaging4 is sufficient for
fine image discrimination is a hypothesis requiring training-only repeat analysis.
Upstream whitening scope is not certified; label this benchmark_preprocessed.

## 2. Architecture, loss, optimization

`models.EEGProject`:8–15 is flatten -> Linear(4250,1024), then residual
GELU -> Linear(1024,1024) -> Dropout(.3), then LayerNorm1024. It is not ConvEEG,
attention, a pretrained EEG encoder or tri-modal alignment. Temporal/spatial sample
information is retained by flattening, but no explicit local convolution bias exists.
All arms initialize the EEG module identically; seed is reset before training
(runner:210–223). Visual heads are trainable, not fixed targets in the learned space.

| Arm | Exact visual path | EEG params | Visual params | Total incl. temperature |
|---|---|---:|---:|---:|
| Baseline | Five normalized means L20/24/28/32/36, each Linear3200->1024; learned softmax routing initially -2,-1,0,-1,-2 | 5,404,672 | 16,389,125 | 21,793,798 |
| Mean | Normalize L28 tokens using pooled fit moments -> mean1024 patches -> Linear3200->1024 | 5,404,672 | 3,277,824 | 8,682,497 |
| Patch | Same normalized L28 tokens -> spatial residual adapter -> GAP -> Linear3200->1024 | 5,404,672 | 5,517,184 | 10,921,857 |

Counts were calculated locally from constructors, with no training. Patch versus
Mean is the nearest controlled contrast; Baseline additionally changes visual
layers, routing and parameter count. `ResidualPatchAdapter` (9–31) reshapes to
Bx3200x32x32; 1x1 conv3200->256/GN32/GELU, 3x3 conv256->256/GN32/GELU,
1x1 conv256->3200/GN32/GELU; add input, spatial mean, then projection. Branch is
randomly initialized, not zero-initialized identity. GroupNorm is sample-local;
checkpoint chunks do not change candidate negatives. Input CPU->GPU transfer stays
in the autograd path. Existing synthetic gradient/chunk tests support this; they do
not establish healthy real-data gradient magnitudes.

Affine normalization commutes with token mean: Mean is a valid same-layer pooling
control (confirmed synthetic test). Patch sees within-image fluctuations scaled by
std of *image means*, which can be much smaller than token spatial std. GN/residual
structure partly responds to this, but neither instability nor superiority follows
without measuring training-only statistics. New diagnostics report variance floors,
validation EEG RMS and sampled token-to-mean normalized RMS ratios.

`clip_loss` (core:15–20) is symmetric diagonal InfoNCE over L2-normalized1024-vectors;
scale starts1/.07=14.286, is learned and capped100. Batch64 setting actually partitions
1800 rows into29 batches of62/63 via `np.array_split` (runner:241–243), 145 steps/arm
in five epochs. There are61/62 negatives per direction, not199 or1023. Null uniform
loss is about ln62=4.127, not a retrieval accuracy. AdamW lr1e-4, wd1e-4, clip norm1;
CosineAnnealingLR T_max5 steps once per epoch, with training LRs approximately
[1e-4,9.045e-5,6.545e-5,3.455e-5,9.549e-6]. No text, margin or augmentations activate
under this config. BF16 on L4; loss and scoring normalize in FP32. Checkpoint
selection is strict `score > best` (274), choosing earliest epoch on ties.

Exact-image multi-positive handling exists (`core.multi_positive_loss`:23) but is
inactive. In this within-subject pilot each averaged stimulus occurs once per epoch,
so the original cross-subject duplicate-ID false-negative defect does **not** apply.
Exact-ID multi-positive and diagonal loss coincide on unique aligned batches, as
local tests show. Same-concept *different* images are negatives by the exact-image
objective. With ten images/concept, a random size62 batch has an expected
61*9/1799=0.305 other same-concept images per anchor. These are semantically related,
but not incorrect labels for exact-image retrieval. Do not relabel all siblings as
positives to improve the reported metric. A semantic auxiliary loss is a different,
future ablation that must leave the primary exact-image metric unchanged.

## 3. Validation meaning and uncertainty

`concept_split` (core:64–72) holds out20 of200 concepts, normally200 images (ten each),
leaving180 concepts/1800 fit images. `validation` (runner:192–202) embeds all200
validation EEG averages against all200 corresponding image vectors. Scores are
normalized cosine products; there is no prototype reduction, CSLS or temperature
factor in ranking. `retrieval_metrics` (core:42–61) matches explicit stimulus IDs and
breaks ties by lexicographic gallery ID. All three arms have the identical pool,
rows, split and scoring; the heads differ. Units are percent, not fraction.

This is **exact-image retrieval among held-out concepts**, not 20-way concept
classification. A model that perfectly identifies the concept but cannot rank ten
siblings can yield10% exact-image accuracy (synthetic counterexample included),
while same-concept top-image accuracy is100%. That example does NOT prove this is
what happened. The optional CPU diagnostic measures both, with the latter clearly
labeled auxiliary and its chance baseline5%, leaving the original metric intact.

User results: Baseline17/200; Mean18/200; Patch18/200. One query is0.5 percentage
points. Naive Wilson95% intervals: 17/200 -> [5.37,13.19]%; 18/200 -> [5.77,13.78]%.
These are descriptive binomial calculations only: queries share20 concept clusters,
checkpoints were selected over five epochs, and only one subject/seed was run.
They are not calibrated post-selection intervals, population/LOSO uncertainty or
a valid paired significance test. Equal hit counts may conceal different errors.
Optional diagnostics output paired discordances and concept-cluster bootstrap
intervals conditional on the selected models. Even those do not adjust best-epoch
selection or estimate between-seed/subject variability. No reliable Patch advantage
is established. Above-chance observed counts do not identify their cause.

## 4. Why decreasing loss does not settle the cause

The objective rewards within-batch alignment on seen stimuli. Validation ranks
unseen-concept images in a larger, sibling-rich pool. Training loss may fall through
better memorization, larger temperature, changed visual heads or genuine transferable
alignment; only the last necessarily helps validation. Synthetic tests demonstrate
that scaling already-correct logits lowers CE with unchanged ranks. High capacity
versus1800 rows and low training loss are consistent with overfitting, but absent
matched train/validation retrieval and per-query predictions they do not prove it.
Five epochs/145 updates could also leave transfer underfit, with LR nearly exhausted.
These explanations can coexist. Do not infer either from one best-score summary.
The observed Patch peak at3 then no higher score is not proof of degradation without
the full curves and uncertainty. Per-step gradient health is not recoverable from
saved histories; optimizer moments/weights may inform, not reconstruct, it.

## 5. Prior experiments: documented, not directly comparable

Source: [PHASE1_AUDIT.md](PHASE1_AUDIT.md), historical base bd1046d2; no CSVs or new
final-test artifacts opened for this task. The earlier audit documents:

| Historical setting | Documented Top1/Top5 | Unit / coverage | Why not a pilot validation comparator |
|---|---|---|---|
| 06b within route_c cosine | 78.8333 /97.4167% | 10 subjects x3 training seeds, 200-way outer image retrieval | Full training pool; image-level validation across1654 seen concepts;60 epochs,batch1024,OneCycleLR; old split/session normalization |
| 06b LOSO plain |26.25 /57.40%|10 target subjects,1 seed,200-way outer retrieval| Nine-source training, all-channel LOSO code, target-selection and normalization caveats; not within-subject |
| LOSO CSLS |37.15 /70.65%|same gallery with query-set-dependent scoring| Explicit label-free transductive scoring, not inductive |
| Recovery/SCORE n_land64 |44.70 /79.15%|historical recovery setting| Transductive recovery; source-only hyperparameter provenance incomplete |
| Capacity n_land128/all |46.45% Top1|different recorded budget| Not rounding of44.7 and not the same experiment |
| 09/09b labeled calibration |No independently recomputed table in prior audit|k-dependent labeled adaptation, often CSLS as well| Calibration CSVs absent at audited base; CV and target-normalization caveats |

06b source c2:27 holds out the last image of every concept (approximately1654
validation candidates); c5:55 sets60epochs/batch1024 and c5:66 OneCycleLR. The
corrected pilot holds out entire concepts and fits moments solely on fit rows.
Its Baseline preserves architecture, not the original training pipeline. Historical
outer metrics and these validation metrics cannot be subtracted as a regression.
Reusable ideas: multi-layer routing already exists in this Baseline; larger-negative
batches, schedule, broader training coverage and temporal/spatial EEG encoders are
candidates for controlled tests, not proven gains here. Exact-ID multi-positive
source alignment matters for future source-subject pretraining/LOSO, not duplicate-free
sub-01 batches. SCORE and labeled calibration must remain separate regimes. ConvEEG,
augmentation, source pretraining and text auxiliaries are not active in this pilot.

## 6. Minimal next plan and predeclared ablations (not launched)

1. Run the new read-only CPU metadata/statistics audit. Resolve any training row-map,
   duplicate, nonfinite or token-mean discrepancy before touching architecture.
2. If those checks pass, optionally re-score only training validation on CPU using
   frozen best states; report exact-image plus same-concept errors and paired
   predictions. CPU FP32 may not reproduce BF16 ties exactly; keep historical scores.
3. Reconcile upstream EEG row identities via a training-only authoritative export
   and, if approved later, raw-training repetition-average spot checks. Existing
   EEG dictionaries lack an independently checked stimulus-ID vector; do not claim
   it is certified merely from shapes. No raw/private array upload is needed.
4. Freeze a diagnostic conclusion before requesting a small controlled training
   experiment. Never use final-test feedback to choose architecture or settings.

| ID | Single proposed factor | Fixed controls | Purpose / gate |
|---|---|---|---|
| D0 | No training: CPU train/validation integrity + selected-state diagnostics | Existing arrays/checkpoints/metric | Mandatory first; any mismatch -> stop |
| A0 | Matched rerun only if a verified data/code defect requires correction | Save separate artifacts, same seed/split/heads | No speculative fix currently authorized |
| A1 | LR schedule/horizon only, predeclared before running | Data,batch,objective,arms fixed; report updates | Distinguish schedule limitation; more epochs not promised to help |
| A2 | Negative-set batch size only, matched update budget | Split,architecture,normalization fixed | Test objective/generalization effect; account for unequal data exposure |
| A3 | Broader fit-concept coverage using training split only | Fixed held-out concepts, candidate identities and updates | Test coverage; must remain disjoint from existing validation |
| A4 | EEG encoder only: existing ConvEEG option | Identical data/visual head/objective | Test inductive bias, not generic capacity stacking |
| A5 | Patch branch identity initialization OR normalization only (separate runs) | Same Mean control, primary metric, split | Only if variance/branch diagnostics motivate it |
| A6 | Exact-ID multi-positive with source-subject pretraining | Source-only tuning and strict LOSO boundary | Later cross-subject task; not expected to alter unique-pair sub-01 loss |
| A7 | Semantic text auxiliary only | Preserve image objective; ambiguity masks only for auxiliary | Planned EEG-image-text work, not implemented pilot evidence |

No candidate-pool resizing, concept relabeling or cherry-picked epoch reporting to
raise accuracy. If diagnostic readout confirms correct alignment and experiment
choices are frozen, final evaluation can be requested for an honest benchmark—even
if validation remains9%. It is NO-GO now because unresolved identity/semantic error
questions are still being used to guide method development.

## Safe diagnostic scope and local validation

`validation_root_audit.py` reads only the training prefix of catalog JSON, the
training selection prefix of dataset JSON, and the initial config prefix of
provenance JSON. It stops before test identity fields. It never calls read_catalog,
TokenCache, final_audit, training runners or any optimizer. It opens only train EEG,
train means, train token array/completion mask, and the three latest checkpoints.
There are no writes to Drive, score exports, model downloads or CUDA calls.
Default metadata/statistics mode does not execute neural inference. Optional
--validation-inference is frozen CPU-only validation scoring; expect it to be slow
for Patch. Prefix parsing intentionally fails closed if artifact ordering changes.
Saved full provenance split/image-ID dictionaries are not loaded because they mix
in final-test identities; reconstructed config split is labeled accordingly.

Synthetic tests cover prefix read boundaries, malformed/reordered metadata refusal,
exact-image versus concept counterexample, actual split/batch counts, duplicate
positive semantics, train-only normalization independence, affine/mean commutation,
temperature/rank counterexample and Wilson calculations. Selected existing pure CPU
architecture/gradient/identity tests are run separately; no optimizer-training test,
real training, feature extraction, GPU, or final-test code is run for this audit.
