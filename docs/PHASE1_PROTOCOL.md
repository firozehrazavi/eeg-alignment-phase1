# Running the new components

Read PHASE1_AUDIT.md first. These additions implement the highest-priority foundation, not a completed ten-subject research result. Historical notebooks, data and result files are immutable. The branch is local until explicitly pushed.

## Verified commands

From the repository root, with Python 3.12, NumPy and PyTorch:

```sh
python -m unittest discover -s tests -v
python -m phase1.smoke --output validation/my_new_smoke
python -m phase1.reconcile --output validation/my_new_reconciliation.json
```

Smoke uses generated tensors, a deliberately tiny model and a 200-candidate synthetic gallery. It compares diagonal and multi-positive training with identical seeds, initialization, data, batches, dropout, optimizer, schedule, validation and epoch budget. Each arm selects its own best checkpoint using the same source-validation rule. It repeats both runs and requires exactly equal histories. Synthetic accuracies are not THINGS-EEG2 results. Output creation is exclusive and refuses reuse. `validation/final_synthetic_smoke/smoke.json` records the final executed CPU check and source hashes; the earlier smoke artifact is retained separately.

`phase1.models` copies the three 06b architecture classes unchanged; an AST regression test verifies them. `phase1.experiment.paired_train` uses those full-size models by default, with no text and no change to the baseline loss. Its intentionally small smoke model is passed explicitly. Training uses float32, without notebook AMP; **both comparison arms must use the same precision**. Historical notebook numbers are not the control arm for this corrected protocol. Train a fresh diagonal control with the corrected split/normalization. GPU determinism may require `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python; record CUDA/cuDNN/device versions.

## Data contract and strict outer-fold execution

Required artifacts are not in the repository: pre-z-score EEG, authoritative image/concept metadata, content hashes, InternViT feature caches, and (for text) frozen embeddings with model revision and prompt provenance. Old `_z16.npy` caches are disallowed for inductive claims. Record upstream preprocessing, repetition counts, channel order and time window. Trial averaging is allowed within a stimulus; report its repetition budget and keep all repetitions of an image in one split. Do not silently replace 63 channels with a fallback.

1. Build one unique image catalog: `image_id`, `concept_id`, `content_sha256`, relative path. Validate with `validate_catalog`; deduplicate identical content before splitting and ensure raw EEG/feature row mappings are joined by IDs. Fail on any train/test concept, image or content overlap. Save the catalog and hash. Do not derive concept IDs from `row // 10`.
2. For each outer target subject, give the inner search only the other nine subjects. For each inner held-out subject, train on the remaining eight, with concept-disjoint fit/validation subsets. Select layers/routing, loss settings, epoch budget and adaptation parameters using source-only pseudo-target validation, never the outer 200 test answers. Fit normalizers inside each inner fold. Store inner train-subject lists, held-out source, split hashes, candidate config, score, data hashes and model revision. `select_source_config` requires complete source validation and rejects target rows; it does not itself train inner models or validate the truth of externally supplied provenance.
3. Lock one selection JSON per outer fold. Use the same locked visual recipe for both loss arms. `paired_train` accepts **source EEG only**, shaped `[source, image, channel, time]`, a unique aligned image-ID list, concept IDs, and a list of frozen visual feature matrices `[image, feature]`. It fits pooled source channel normalization and per-layer visual normalization on fit concepts only, validates every source, and returns models, frozen normalizers and partition indices. Save its state dicts, normalizer tensors and split as new run artifacts. For within-subject use one subject's training EEG and select channels explicitly before calling. Pooling source statistics is a deliberate corrected baseline; per-subject source scaling is a separate ablation.
4. Only after fitting, load the outer evaluation EEG. `evaluate` applies frozen training statistics and an explicit unique 200-image gallery. It permits repeated query identities but never duplicate candidates. Store ID-based predictions/scores to permit metric recomputation. Test-gallery row permutation must not change accuracy; score ties break lexicographically by stable string identity. Candidate identity type must be consistent. Report query count and candidate count with Top-1/Top-5 percentages.
5. For each seed, the training arms share initial weights and the complete training recipe. Seed 0, 1 and 2 independently rerun training; changing calibration draw seeds is not model replication. The JSON manifest lists all ten subjects, three model seeds and staged ablations. No GPU-scale runs have been performed here.

## Evaluation regimes (never pool these)

| Setting | Target labels for fitting | Target query statistics | Scoring |
|---|---|---|---|
| within_subject | training concepts only | none | cosine with frozen within-training stats |
| loso_inductive | none | none | cosine with source-fit stats |
| loso_transductive | none | permitted, declared | CSLS/recovery, source-selected settings |
| labeled_calibration | k disjoint calibration images | none | cosine after calibration-only fit |
| labeled_calibration_transductive | k disjoint calibration images | permitted, declared | calibration + CSLS/recovery |

`AccessPolicy` guards declarations; `csls` refuses an inductive policy. It cannot inspect preprocessing performed elsewhere. Test-gallery features are known retrieval candidates, but fitting any EEG transform using the outer query set is transductive. Full target training-session statistics exceed a k-calibration budget and are rejected by the defined policies. Historical within-CSLS can be reported in a separate explicitly transductive table, not as strict within-subject.

Calibration implementation remains a next-stage experiment, not a silent port of the flawed notebooks. Use the same nested ID permutation per subject/draw for all methods and budgets, disjoint from final test concepts/content. Fit preprocessing on source + permitted k examples, never all target training rows. If calibration labels choose alpha/rho, declare that target-label access and use concept folds inside calibration with normalization refitted per fold. Global hyperparameters must still be locked on source pseudo-target subjects. For k<2, retain the identity map. Count unique images, raw repetitions, and acquisition time separately. Store hybrid labeled/transductive results separately. No never-worse guarantee is made.

## Tri-modal hypothesis and staged ablations

Keep InternViT frozen and the visual retrieval head as the primary objective. Cache text from a pinned frozen text encoder, using curated concept names and image-grounded descriptions. Keep ambiguous captions marked; do not use test EEG, retrieved test labels or outcome-guided caption editing. Template selection, semantic-neighbor thresholds and loss weights use source validation only. Check content duplicates and caption provenance before fitting.

`SemanticHead` is an experimentally testable auxiliary component, not yet wired into the full training runner. It has lightweight EEG, text and visual projection heads. Frozen text/visual inputs are detached. A prototype blends projected text with its paired image; the adaptive EEG-text weight uses detached nonnegative text/image agreement. A source image/text alignment term trains those heads. The visual loss remains exact-image multi-positive loss. Semantically related images do not become correct retrieval answers: only their auxiliary semantic denominator entries can be masked. Shared concept captions should be either multi-positive in that auxiliary task or excluded as negatives; unrelated captions remain negatives. Supply boolean positive/ambiguous masks explicitly and report retained-negative counts and gate distributions. Gate detachment prevents a direct gradient shortcut but does not guarantee absence of gradual gate collapse; monitor it.

Test the manifest's visual-only, fixed text, grounded fixed text and grounded adaptive text variants. Include weight zero and grounded-only (blend=1) controls; use one source-chosen weight for outer test. Compare fixed and adaptive loss with matched compute, source selection budget and initialization. Report degradation and negative transfer as well as improvement. Unit tests verify frozen inputs, masking and gradients; they do not establish useful semantics or improved retrieval. Train-time integration and real text ablations remain pending the data/provenance stage.

## Run records and reporting

One exclusive output directory per protocol/outer-subject/model-seed/ablation, with git commit + dirty-file hashes, Python/library/device versions, deterministic flags, full config, feature/model revisions, data/catalog hashes, selected source subjects and split IDs, epoch history, best checkpoint, access policy, calibration draw and IDs, query/gallery IDs, scores, Top-1 and Top-5. Use `open(..., 'x')` for new metadata and refuse existing run directories. Avoid filename-only cache reuse. New files never replace historical CSVs.

Required table key: `(protocol_version, setting, subject, model_seed, ablation, calibration_k, draw, scoring)`. Keep plain/CSLS/recovery separate. Check missing/duplicate rows before aggregating. Average draws within model seed, then model seeds within subject; report mean and SD across ten subjects and paired subject-level differences. Bootstrap subjects (not 30 seed rows as independent people), report uncertainty and worst-subject performance, and correct the predeclared family of comparisons (e.g. Holm). Do not choose a winner on outer test outcomes and then report an unadjusted p-value. Calibration thresholds and confidence coverage belong to separate analyses.

The historical reconciliation JSON is evidence about committed CSVs, not certification of their protocol. It explains the 44.7/46.45 difference and records input hashes. Missing 09/09b CSVs prevent full calibration-table verification.
