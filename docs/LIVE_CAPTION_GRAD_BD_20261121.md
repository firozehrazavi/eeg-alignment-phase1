# Direct EEG-to-caption gradient: controlled B-live / D-live intervention

## Why one more limited experiment, not another architecture
The historical 06b image-aligned EEG checkpoint remains the backbone. Matched corrected
B-prime and D-prime (real semantic residual vs cross-concept deranged residual) have already
completed epoch 15. On sub-01 seed-0 **1,654-way training-partition validation**:
- B-prime: EEG->Image Top-1 26.3603%, EEG->Caption Top-1 1.2696%.
- D-prime: EEG->Image Top-1 26.2394%, EEG->Caption Top-1 1.2092%.
- D1-prime train-row gradient ratio: B-prime 12.1027%, D-prime 12.3117%;
  both teacher argmax flip rates 0.391% on the sampled rows.

The increase in gradient magnitude did NOT prove a meaningful semantic retrieval advantage.
Inspection of the corrected runner showed the **direct caption retrieval loss** used
`model.text_scores(ze.detach(), text_batch)`. It trained the caption readout layers
but could not directly train the EEG encoder. The distillation loss in
`joint_model.py` also deliberately detaches the student EEG embedding, and is
unchanged in this new intervention. Thus the previous correction does not test
whether supervised direct caption learning can improve the EEG representation.

## Registered intervention and scientific comparison
One changed computation:
```python
model.text_scores(ze if live_caption_gradient else ze.detach(), text_batch)
```

The model, rank, alpha=20, joint teacher, image contrastive loss, caption readout loss
weight=0.1, distillation weight, images, text targets, 06b warm-start, schedule (60 epochs),
fit/validation split, optimizer, seed, and duplicate-caption-aware scorer stay unchanged.
Only the gradient from direct-caption retrieval is allowed to reach EEG.

- **B-live** (arm `B`): direct supervised Caption loss reaches EEG; real semantic residual.
- **D-live** (arm `D`): identical direct caption supervision; shuffled semantic residual.

B-live vs D-live tests the *additional value of matched semantic residuals*, conditional
on direct caption supervision being live.
B-live vs archived B-prime and D-live vs archived D-prime examine the effect of releasing
the direct caption gradient, subject to the same initialization/implementation/split.
These are exploratory single-subject results, not full statistical confirmation.

## Safety and result identity
Opt-in `semantic_correction=True, live_caption_gradient=True`, only arms B/D.
New output MUST start with `trimodal_live_caption_bd_`. The exact output used
by the notebook is
`/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec/trimodal_live_caption_bd_20261121`.
The old B-prime/D-prime outputs and checkpoints remain untouched. The new
`live_caption_gradient` option is in the execution signature, together with
input hashes, code hashes, device settings, and training schedule. Full resume
checks input provenance and preprocessing artifact hashes; changed source code
or a different experiment configuration must not resume the same output.

The existing 06c notebook in this branch is updated, pinned to the changed
`phase1/trimodal_06b.py` SHA256, and `RUN_TRIMODAL=False` by default.
It runs only B/D with a **five-epoch absolute ceiling** and soft 900-second
invocation wall-clock budget when deliberately authorized.

## Before new GPU (CPU only)
- Execute focused unittest suites `test_trimodal_execution.py`,
  `test_trimodal_06b.py`, and `test_06c_notebook_audit.py`.
- Test nonzero EEG-embedding gradient from the live direct-caption loss
  versus None from the old detached readout.
- Test independent output directory, matching B/D signatures, resume
  idempotence, and no changes to old checkpoints.
- Confirm fresh branch/hash pin, Drive baseline, caption cache, T4 quota
  and available space before explicit authorization.
- DO NOT start automatically on importing or opening the notebook.

## Stop/go criteria after the 5-epoch pilot
- Stop if loss nonfinite, any checkpoint or preprocessing hash does not match,
  or corrected resume fails.
- Mechanism check: `grad(loss_caption, EEG_embedding)` is nonzero on
  a real train batch and old detached control gives no such gradient.
- Compare B-live and D-live Image/Caption Top-1 and Top-5 at matched epochs.
  Do NOT claim an improvement on the basis of one or two 1,654-way successes.
- If image retrieval collapses sharply against archived baselines or Caption
  fails to improve beyond controls, NO-GO for full 60 epochs.
- Otherwise consider controlled extension to epoch 15 with estimated
  GPU minutes/compute units, then paired per-query tests. No automatic extension.

## Mandatory caveats
This is **training-partition** development validation. Historical visual
normalization used all 16,540 rows including validation rows. The EEG-image
row identity and caption encoder provenance must be independently verified
before official protocol claims. No official 200-way, 10-subject test, no
SOTA number and no paper-significance claim are produced by this change.
