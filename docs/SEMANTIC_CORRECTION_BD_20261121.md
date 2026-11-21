# B-prime / D-prime: calibrated triadic-only gradient experiment

This is a new, opt-in scientific experiment, NOT a continuation of the completed
A/B/C/D results. It reuses the completed historical 06b checkpoint and existing
official-training-only captions, with no new caption generation. The old branch,
checkpoints, and outputs must remain unchanged.

## Prior registered evidence (sub-01, seed 0, 1,654 training-partition validation queries)

At epoch 15, image Top-1: A 26.2394%, B 26.2999%, C 26.2394%, D 26.3603%.
Caption Top-1: A 1.3301%, B 1.2092%, C 1.0883%, D 1.0883%.
There was no robust B advantage over the shuffled control.

Read-only D1 on existing epoch-15 weights measured:
- B (real): H/B row-std 0.050752, semantic-backbone gradient share 0.015857,
  teacher NCE delta -0.021731, top-1 flips 0%.
- D (cross-concept shuffled): H/B 0.046064, gradient share 0.016405,
  NCE delta +0.001359, top-1 flips 0%.
- Training-mode main gradient norm before clipping: 2.545763 (clip threshold 1).

These are one-batch, train-row diagnostic values and do NOT establish
population-level improvement or prove the complete gradient is semantically inert.

## One linked correction, one control

For corrected teacher logits, the image score B is a detached **context**:
`teacher = B.detach() + alpha * H`. The separate base image loss
`NCE(B)` remains differentiable and unchanged. The H interaction connects EEG,
image, and residual-caption embeddings through their triadic projections.
Thus the joint loss can update the EEG backbone *through H*, but not through
its redundant copy of B. The caption readout still consumes EEG without an
image gallery at inference. Existing readout/distillation detach behavior is
unchanged for this controlled test.

`alpha=20.0` is fixed from the pre-existing train-fit D1 B ratio
(1 / 0.050752 = 19.70 approximately, rounded to 20).
It is applied identically to B-prime and D-prime, not adjusted by validation
or changed during training. **This approximates logit-std matching at D1,
not guaranteed matching for fresh initialization or later training.**
The effective gradient share MUST be measured separately; no benefit is
assumed.

Only two arms are allowed:
- **B-prime** (stored as `arm=B`): real residualized-caption training features.
- **D-prime** (stored as `arm=D`): cross-concept deranged residual captions.

Both arms use the same 06b checkpoint, optimizer, OneCycleLR 60-epoch schedule,
row split, random seed, image loss, caption readout, and distillation weighting.
The correction is enabled by `semantic_correction=True` and enters the
experiment signature alongside source hashes, alpha and input provenance.

## Data and evaluation caveats

The residualizer is a rank-64 random projection/ridge model, and its output
does not establish unique textual information beyond vision. The legacy
visual-feature normalization includes the 1,654 validation rows, so this is
development validation, not untouched inductive validation. No official test
or 200-way ten-subject run is authorized by this experiment.

## Execution gate, output and budget

New directory ONLY:
`/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec/trimodal_semantic_correction_bd_20261121`.

Existing 06c notebook is edited **in this new branch only**, with
`RUN_BASELINE=False`, `RUN_TRIMODAL=False`, `ARMS=('B','D')`,
`PROFILE=True`, `TIME_BUDGET_SECONDS=900.0`. Training remains off until
explicit GPU approval. The soft 900-second budget can overrun while completing
the current epoch, validation, and atomic checkpoint. The five-epoch ceiling is
absolute and resumable. No code edit is permitted during resume.

Before GPU: run focused CPU tests; confirm notebook source hashes and branch,
completed historical baseline, train-only captions, available GPU/storage,
and account quota. None of the prior A/B/C/D files may be rewritten.

After the five-epoch prefix: verify that **both** arms reach epoch 5, with
finite losses, distinct verified latest checkpoints, image/text validation
metrics, wall-clock times, per-epoch memory peaks, and valid resume metadata.
Assess semantic gradient share again on the new B-prime checkpoint:
a meaningful increase (e.g. above 0.05 of base image gradient) and changes
in teacher rankings are mechanistic evidence only, not proof of improved
retrieval. If any gate fails, NO-GO. Do not select B-prime by validation alone.

Continuation to epoch 60 would require a recorded GPU time/compute-unit
projection, adequate budget, matched B-prime/D-prime and a distinct
scientific rationale. If B-prime remains indistinguishable from D-prime under
a fully matched, pre-specified evaluation, no claim of semantic benefit.
Official test remains unopened.

## Developer checks (no data access or CUDA)

```bash
python -m unittest discover -s tests -p "test_trimodal_execution.py" -v
python -m unittest discover -s tests -p "test_trimodal_06b.py" -v
python -m unittest discover -s tests -p "test_06c_notebook_audit.py" -v
```

Additional tests verify the detached joint image-base gradient, fresh
B-prime/D-prime output, source-pin bootstrap, absolute epoch ceiling,
bit-exact resume and prevention of old output reuse.
