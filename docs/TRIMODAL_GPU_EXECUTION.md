# Budgeted tri-modal GPU execution

This execution note continues the published `120db89aa036b465ac46fc8ec99e67094f81d235`
implementation. It does not restart the architecture or change the scientific
split. No GPU training or official-test evaluation was performed while preparing
this change.

## Starting evidence and interpretation

The user reports a completed sub-01, seed-0 historical baseline, selected at
epoch 25 with **27.811% validation Top-1 on the 1,654-image training validation
gallery**. The user also reports verified `[16540,512]` caption embeddings,
15,676 unique exact-caption identities, and two corrected embeddings. These
observations were supplied by the user, not independently remeasured here. They
are not 200-way official-test results or evidence of recovered historical 78.8%.

Before a new run, the baseline record/checkpoint and corrected caption files must
pass the existing hash and provenance checks. Freeze the corrected caption
artifacts before beginning: their new hashes define this experiment. Do not
resume an old output whose signature describes the pre-correction inputs.

## What the comparisons establish

| Arm | Image-backbone objective | Caption-readout objective |
|---|---|---|
| A | Historical image contrastive loss | Shared caption contrastive loss on detached EEG embeddings |
| B | Image loss plus joint loss using residualized captions | Shared readout loss plus teacher distillation |
| C | Same loss structure as B, with variance-scaled random residual features | Same loss structure as B |
| D | Same loss structure as B, with fit-concept-deranged residual captions | Same loss structure as B |

All arms start from the identical baseline weights and new-head initialization,
use the same row split, batch sequence, optimizer schedule and clipping settings,
and receive caption supervision for their readout. A is the image-only
**backbone** control; its complete system still contains supervised text heads.

B is a valid first feasibility test for the existing tri-modal implementation.
B versus A tests the combined added joint/distillation training package. It does
not isolate the semantic content of residual captions. B versus C and D is needed
to examine that content under matched added losses. A gain in this one subject
and seed would remain development evidence; it would not establish independent
linguistic ground truth, population generalization, or SOTA. Captions originate
from images, and the residualizer removes only its fitted linear visual component.

The preserved split uses 14,886 optimizer images and the last image per concept
for 1,654 validation images. Historical visual and EEG preprocessing includes
validation images in its fitted statistics; the added text normalizer and
residualizer fit optimizer rows only. The same validation gallery selects epochs
and supplies reported development metrics. No new holdout is introduced.

Text inference remains image-free: raw EEG passes through `backbone.eeg`, and
`text_scores` consumes its embedding and normalized caption candidates. Candidate
image features, the joint teacher and the residualizer are not inputs to direct
caption scoring. The preserved text normalization and readout weights are needed
to reproduce it.

## Execution controls and output identity

Use a **new** output directory:
`/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec/trimodal_budgeted_20261121`.
Use that same directory for the measured prefix and its subsequent continuation.

The runner accepts `arms`, `max_seconds`, and `max_epochs_per_arm` as execution
controls. They do not change the fixed 60-epoch schedule, 900 OneCycleLR updates
per arm, validation every five epochs, or scientific signature.

* `arms=('A','B','C','D')` selects work for this invocation. Selected subsets can
  be resumed in the same directory; only completed, verified arms are skipped.
* `max_epochs_per_arm=5` is an **absolute epoch ceiling**, not five additional
  epochs. Repeating that request advances unfinished selected arms toward epoch
  five and never advances an arm from five to ten. Use `None` only for the later
  full continuation.
* `max_seconds` is a soft wall-clock budget beginning before preflight/setup.
  The runner completes an already-started epoch, scheduled validation, and the
  atomic latest checkpoint before stopping. It can exceed the requested time by
  that work and does not terminate a paid Colab runtime automatically.
* Paused runs retain per-arm progress and latest checkpoints. They must not be
  represented as completed 60-epoch arms in `results.csv`. A complete selected
  subset is not a completed A/B/C/D experiment.

Keep the historical baseline training gate disabled. This task reuses its
completed checkpoint; it does not rerun that baseline.

## Short measured GPU profile

The next GPU action, when explicitly authorized, is a bounded prefix of the
actual experiment. First run all four arms through epoch five in the same new
output. This measures each distinct arm and includes one genuine scheduled
validation. Preserve the full 60-epoch scheduler during this prefix.

Example after the notebook has verified code and defined its input paths:

```python
from phase1.trimodal_06b import run as run_trimodal

budgeted_output = RUN_ROOT / 'trimodal_budgeted_20261121'
# Set from the user's available runtime/cost budget, not an estimated speed.
profile_wall_budget_seconds = APPROVED_PROFILE_SECONDS
profile_result = run_trimodal(
    MANIFEST, RUN_ROOT / 'baseline', TEXT_FEATURES, CAPTION_IDS,
    CAPTION_PROVENANCE, budgeted_output, device_name='cuda', resume=True,
    arms=('A', 'B', 'C', 'D'),
    max_seconds=profile_wall_budget_seconds,
    max_epochs_per_arm=5,
)
```

`APPROVED_PROFILE_SECONDS` is an explicit budget to supply, not a default cost
claim. If the budget expires before all four arms reach five, inspect progress
and rerun with the same ceiling and output only within the authorized budget.
Do not declare the all-arm cost profile complete from a partial subset.

Record the actual Colab GPU model, precision, software/runtime signature, account
billing rate or compute-unit rate, invocation wall time, and memory peaks. The
epoch log separates `train_seconds`, `validation_seconds`, and
`checkpoint_seconds`, and records CUDA allocated/reserved peaks. Exclude each
arm's first epoch from steady-state training estimates; include its observed
cost in the measured prefix. CUDA work must be synchronized for elapsed timing.

For each arm `a`, let `e_a` be its completed epoch, `t_a` its observed steady-state
training seconds per epoch, `v_a` its measured scheduled-validation seconds, and
`c_a` its measured atomic-checkpoint seconds per epoch. A transparent remaining
runtime projection is:

```text
remaining_seconds_a = (60 - e_a) * (t_a + c_a)
                    + (12 - floor(e_a / 5)) * v_a
remaining_seconds = sum(remaining_seconds_a for A, B, C, D)
                  + expected further invocation setup/final-save overhead
```

Use observations from the actual GPU, input sizes and all four arms. Report the
estimator used, observation count and timing spread; one validation observation
per arm gives a weak estimate of its variability. Include repeated setup,
Drive latency, final artifacts and billable idle time in the practical budget.
The profile-summary helper's projections are planning estimates, not guarantees.

```text
projected_cost = projected_billable_hours * actual_account_rate_per_hour
projected_compute_units = projected_billable_hours * observed_units_per_hour
```

If the account does not expose a currency rate, report compute units or GPU hours;
do not invent a dollar conversion. No numerical GPU duration or cost estimate is
available before these measurements. The old CPU batch-32 profile is unsuitable
for this calculation.

## GO / NO-GO decision

**GO for the bounded prefix** only when the completed baseline and corrected
training-only caption artifacts verify, source/runtime checks pass, a suitable
GPU and sufficient storage are available, and the profile's time/cost budget is
explicitly authorized. Otherwise retain **NO-GO** and report the failing gate.

**GO for continuation** only after all four arms reach epoch five with finite
losses, valid checkpoints, scheduled image/text metrics, usable CUDA timing and
memory observations, and a measured remaining-time/cost projection within the
authorized budget. Review the recorded caption gallery/duplicate counts and
confirm that direct text scores are computable without image candidates.
Performance gains at epoch five are not a necessary engineering gate and must
not be used to select only favorable arms.

After GO, resume the same output and all four arms with
`max_epochs_per_arm=None` and an explicit continuation wall budget. If projected
cost exceeds the budget, timing evidence is incomplete, an arm fails, or artifacts
cannot be verified, stop at **NO-GO** pending a concrete budget or defect
resolution. Do not shorten `EPOCHS`, reset OneCycleLR, silently restart heads,
drop unfavorable controls, or label a truncated run a matched full experiment.

## Metric report

For each arm report completed epochs, latest epoch, image-selected best epoch,
image Top-1/Top-5 with **1,654 image candidates**, and duplicate-aware text
Top-1/Top-5 with the **observed unique validation caption count**. The full-cache
15,676 unique captions is not the validation gallery size. Preserve query
weighting, exact-caption identities, score files and hashes. Partial prefix
metrics must be labeled epoch-five training-validation development metrics;
best-so-far prefix scores are not the final 60-epoch selected results.

Report B−A, B−C and B−D descriptively with their different interpretations above,
using the same validation queries and selection rule. Image-based checkpoint
selection is fixed; do not select a second checkpoint by text performance after
seeing results. Official-test evaluation remains unstarted and outside this
execution plan.
