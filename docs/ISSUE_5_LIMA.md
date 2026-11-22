# Issue 5: evidence-gated LIMA candidate

Starting source: `research/live-caption-grad-bd-20261121` at
`6cbb5eb7bffb446d4b318b4c7159faf853cc1dda`.
Scope: sub-01, seed 0; 14,886 optimizer rows and a 1,654-caption
training-partition development gallery. No official-test evaluation is authorized.

**Decision: NO-GO for GPU training until the real-data feasibility gate passes,
its provenance limitations are reviewed, and GPU execution is explicitly
authorized.** The implementation supplies a CPU gate and a dormant candidate,
not a training runner. Synthetic regression success is not real-data feasibility.

## Executed checks

The complete available CPU suite passed: **175 tests in 54.814 seconds**, using
Python 3.12.5, PyTorch 2.9.1+cpu and NumPy 2.3.5. Exact commands and the complete
log are in `validation/issue5/`. These are synthetic regression tests, including
existing checkpoint/resume fixtures, not repeated baseline training on research
data. The local input existence check returned **NOT_MEASURED / NO_GO** with
seven required files missing. No GPU or official-test execution occurred.

## Evidence status

The following are reported evidence from [Issue #5](https://github.com/firozehrazavi/eeg-alignment-phase1/issues/5),
not measurements repeated in this implementation session:

| Frozen 06b development comparison | Top-1 |
|---|---:|
| EEG to image | 27.81137% (460/1,654) |
| Image to caption, image-fitted ridge | 44.80048% (741/1,654) |
| EEG to caption, image-fitted ridge | 0.90689% (15/1,654) |
| EEG to caption, EEG-fitted ridge | 1.20919% (20/1,654) |
| Shuffled EEG-to-caption ridge | 0.06046% (1/1,654) |

The reported paired 95% interval for EEG-fitted minus image-fitted ridge is
[-0.2418, 0.8464] percentage points; it does not establish improvement.
B-live's reported 1.27% uses a different checkpoint and is not a paired baseline.
Historical approximately 78.8% and external RPA 95.4% are different-protocol
200-way results; neither is a result of this implementation.

**NOT MEASURED here:** LIMA real-data teacher retrieval, paired confidence
interval, student retrieval, GPU time, peak GPU memory, or monetary cost.
The full EEG cache, complete visual cache, historical checkpoint and associated
caption artifacts are required together on mounted Drive. No GPU cost estimate
can be inferred from unrelated B/D logs or synthetic CPU tests. Record measured
wall time and memory from an authorized bounded pilot before estimating a longer
run; monetary estimates additionally need the actual billing rate.

## Targeted code audit

References below describe the canonical starting source; the historical modules
and existing experiment signatures remain unchanged.

| Observation | Source | Consequence |
|---|---|---|
| File hashes and channel/shape declarations contain no independent stimulus IDs. Concept blocks and row order are positional assumptions. | `phase1/historical_06b.py:98`, `:244`, `:252` | Hash verification protects artifact identity, not semantic EEG/image/caption alignment. Independent row verification remains outstanding. |
| Visual means and standard deviations include development rows. | `phase1/historical_06b.py:245` | Preserve the historical checkpoint's input transform and disclose this limitation; do not describe development as untouched validation. |
| Caption preflight validates partition declarations, dimensions and identical-caption embedding consistency. | `phase1/trimodal_06b.py:58`, `:81` | It cannot independently verify the generating image or caption provenance. New text fits must use optimizer rows only. |
| Earlier bridge diagnostics call `_load_training` without the full historical source-hash preflight. | `diagnostics/image_caption_bridge_gate.py:147`, `diagnostics/eeg_bridge_transfer_gate.py:187` | The new gate must verify source hashes and the baseline's manifest binding before loading its inputs. |
| Exact-caption positives are aggregated; retrieval has stable tie ordering. | `phase1/joint_model.py:26`, `diagnostics/image_caption_bridge_gate.py:70` | Preserve duplicate-aware scoring; LIMA caption loss sums positive probability mass. |
| Existing direct caption scores consume EEG and text only. Existing distillation detaches teacher and student EEG. | `phase1/joint_model.py:97`, `:132` | Keep gallery-free student inference; test the new marginal semantic gradient separately from KL distillation. |

The new CPU gate uses only optimizer image embeddings as mediators, with fixed
temperatures and an optimizer-only image-to-caption fit. Development caption
embeddings are retrieval candidates, never fit labels. Real and cross-concept
shuffled caption conditions share the EEG posterior, split, gallery and scoring.
The gate must not substitute development image embeddings for mediators.

## Mechanism and implementation boundary

`phase1/lima.py` implements
`pbar(caption|EEG) = sum_image pi(image|EEG) * p(caption|image)`.
Its `marginal_log_probs` uses exact matrix marginalization with global
normalization and a blocked log-space fallback for extreme inputs, rather than
independently normalized minibatches or prototypes.
`caption_mass_nll` handles duplicate positives. `duplicate_caption_kl` aggregates
caption identities and detaches the teacher target. The marginal loss retains
an EEG gradient; the KL student receives detached EEG. `LimaAligner` preserves
the historical `Aligner` and compatible checkpoint loading, and exposes direct
`text_scores(EEG_embedding, captions)` with no image bank at inference.

The teacher is a train-gallery retrieval proxy; a useful teacher alone does not
establish a useful gallery-free student or a novel scientific contribution.
The issue's primary gate uses all optimizer image mediators. The training teacher
excludes same-concept mediators to prevent self-image shortcuts. A separately
labelled sensitivity check repeats the marginal calculation with that exclusion,
using the same fitted maps and temperatures; it does not fit another model.
Passing the primary gate alone does not establish feasibility for the excluded-
concept training pathway. Continuation also needs evidence that the compatible
pathway carries signal. The candidate is dormant while that evidence is absent.
No existing B/D schedule,
checkpoint, signature, or Drive result is overwritten.

## One CPU execution and fixed decisions

Use the existing `notebooks/06c_checkpointed_06b_and_trimodal.ipynb`. Its first
bootstrap cell mounts Drive, verifies the reviewed sources and executes the CPU
gate; no additional diagnostic notebook is needed. The equivalent command from
the verified checkout is:

```bash
python diagnostics/lima_feasibility_gate.py \
  --root /content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec \
  --cache /content/drive/MyDrive/eeg_results/cache \
  --output /content/drive/MyDrive/eeg_results/lima_gate_issue5_NEW.json
```

Choose a new output path; never replace an earlier report. Retain the complete
report with source/input/checkpoint hashes, partition rows, fixed configuration,
per-query paired hits, CPU timing and decision. Missing artifacts mean
**NOT MEASURED / NO-GO**, not a zero score or a simulated measurement.

Predeclared feasibility: teacher duplicate-aware Top-1 **at least 5%**, with the
paired bootstrap 95% lower bound for real minus shuffled **strictly above zero**,
in the identical 1,654-caption gallery using only optimizer image mediators.
Passing is permission to review a bounded pilot, not evidence of generalization
and not authorization to run one. A failure stops this candidate; do not tune
temperatures on development or switch architectures.

If later authorized, compare real/shuffled subject-01 seed-0 arms with identical
warm start, initialization, batches, schedule, budgets and selection rules.
Predeclare an epoch ceiling and wall-time budget; finish epochs and save atomic
resumable checkpoints. Require direct caption Top-1 **above 3%**, real-minus-
shuffled paired CI lower bound **above zero**, and image Top-1 no more than
**one percentage point below frozen 06b** on the same development rows. Assess
both metrics from the same selected checkpoint. Report failed criteria as NO-GO.

## Eventual official evaluation checklist

- Freeze the complete 200-way gallery and all ten subjects before opening the
  official test; obtain explicit authorization separately.
- Match EEG preprocessing, trial/repetition averaging, time window, channel
  names/order, image feature encoder/layers/pooling, and caption encoder/cache.
- Independently reconcile stimulus IDs across EEG, visual features and captions;
  preserve original artifacts, hashes, generator/encoder revisions and row maps.
- Freeze seeds, schedules, arm budgets and checkpoint selection using training
  and nested development only; never select epochs or temperatures on test.
- Use exact image positives and duplicate-aware caption positives; publish gallery
  identity, tie semantics, per-subject denominators and paired uncertainty.
- Verify direct EEG-to-caption inference with all image-gallery access disabled.
- Record selected epoch, checkpoint and source provenance, runtime environment,
  GPU model, elapsed time and peak memory for each matched arm.
- Compare historical 06b and external methods only after reconciling every
  protocol difference. A single-subject development result is not a ten-subject
  official benchmark or a SOTA/novelty claim.
