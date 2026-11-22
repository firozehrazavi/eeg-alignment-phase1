# Fixed shared-CLIP teacher preflight (2026-11-22)

## Why this experiment, and what it cannot prove

**Research goal** remains genuine joint EEG–Image–Text training with direct gallery-free EEG→Caption inference while preserving historical 06b InternViT 200-way retrieval performance. However, recent **real** 1,654-candidate development diagnostics on subject 01 seed 0 were negative for several frozen text-bridge approaches:
- EEG→image: 27.81137% (different task/protocol from historical 200-way 78.8%).
- EEG→image-trained text Ridge: 0.90689%; EEG-trained BLIP-caption Ridge: 1.20919%.
- EEG→canonical concept-text Ridge: 0.18138%, shuffled concept mapping: 0.06046%; paired concept-minus-BLIP 95% CI entirely negative.
- LIMA teacher (existing PR #6): ~1.995%, below its 5% feasibility gate; no GPU authorization.

Image features come from frozen InternViT, caption embeddings from OpenCLIP text. That representation mismatch **may** be a contributing bottleneck, but has NOT been causally established by the above evidence. This experiment isolates whether a **single already-pretrained vision-language model** provides a reliable paired image/text target on actual *training* images before risking extraction of a complete CLIP visual feature cache or any EEG training.

## One preregistered test, no tuning
- Use exact previous OpenCLIP \`ViT-B-32 pretrained=openai\` weights, verifying the actual full state-dict SHA \`a3ce3c4a2245ed2a572d2eb864a62f0a4ca62b3123b83f8a52a523c3b1dd32a4\`.
- Select **256 distinct training concepts**, deterministically from 1,654 with NumPy seed \`20261122\`. Take the **first exemplar** (\`10*concept\`), which is always an optimizer-fit row rather than the historic development row \`10*concept+9\`.
- Source original JPEG paths using the user-owned \`THINGS-EEG2/images_set/image_metadata.npy\` and pinned \`eeg_results/cache/meta_v6.pt\`. Verify all 16,540 path/name alignments and source hashes against caption provenance; never open \`test_images\`, official test EEG, or dev image rows.
- Read only the \`train\` key of the original \`captions_blip_large.json\`; require exact equality to the 16,540 \`caption_ids_train_only.json\` strings and metadata source SHA matches.
- Encode those 256 original images, 256 original BLIP captions, and 256 corresponding canonical concept prompts (\`a photo of a {concept}\`) in **the same frozen OpenCLIP model**, no gradients. Score all queries in a **256-candidate** gallery for each text target, duplicate-caption-aware.
- Negative controls: shuffle caption *vectors* (leave identities in place) by one fixed 256-way cyclic shift; likewise for concept prompt vectors. Paired bootstrap CI for Top-1 differences.
- **Fixed feasibility criterion:** paired-real-vs-shuffled CI lower bounds > 0, image→BLIP-caption Top-1 ≥25%, AND image→concept-prompt Top-1 ≥35%. This is a pragmatic teacher quality gate, not an EEG performance target.
- If the gate **fails**: do not replace the old representation or train EEG on CLIP; return to 06b, identify a different evidence-based bottleneck. If it **passes**: the shared model is only a *candidate teacher*. Require an independently audited alignment/cheap frozen EEG compatibility probe before any controlled GPU change; do not conflate teacher quality with EEG→Caption feasibility or a SOTA result.

## Implementation and tests

- \`diagnostics/shared_clip_teacher_gate.py\`: fixed 256-way original-image and text preflight, read-only inputs, exclusive-new JSON output. CPU only. No new notebooks.
- \`tests/test_shared_clip_teacher_gate.py\`: synthetic CPU tests for train-only choice, source ordering, refusal of test/traversal, positive vs shuffle behavior, duplicate-caption scoring and no-go thresholds.
- **Test status**: tests are added to the branch but not yet confirmed run in the cloud GitHub runner. Run the test command before any real-data gate. Do not call this a scientific PASS unless real-data metrics are reported.

## Single CPU Colab execution in existing 06c notebook

Only one code cell, no T4/GPU, no re-training. Use existing mounted Google Drive. The script will encode 256 images using CPU; it may fetch OpenCLIP weights if not already cached, without requiring API credentials.

~~~python
import subprocess, sys, importlib.util
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

drive.mount('/content/drive')
repo = Path('/content/eeg-alignment-phase1-06c-live-caption')
ref = 'research/shared-clip-teacher-preflight-20261122'

def call(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)

if not (repo/'.git').is_dir():
    call(['git', 'clone', '--depth', '1', '--branch', ref,
          'https://github.com/firozehrazavi/eeg-alignment-phase1.git', str(repo)])
else:
    call(['git', 'fetch', 'origin', ref], repo)
    call(['git', 'switch', '--detach', 'FETCH_HEAD'], repo)

if importlib.util.find_spec('open_clip') is None:
    call([sys.executable, '-m', 'pip', 'install', '-q', 'open_clip_torch'], repo)

call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
      '-p', 'test_shared_clip_teacher_gate.py', '-v'], repo)

folder = Path('/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/shared_clip_teacher_reports')
folder.mkdir(parents=True, exist_ok=True)
output = folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_sharedclip.json')

call([sys.executable, '-u', 'diagnostics/shared_clip_teacher_gate.py',
      '--output', str(output)], repo)
print('REPORT SAVED:', output)
~~~

The JSON is the only real-data result; no 200-way official test is opened. Archive the timestamp, source hashes, exact weights SHA, selected image SHA and full metric/CI values. No old checkpoint/report overwritten.

## Limitations
- This is **image→text** teacher feasibility, **not EEG→Caption**. It neither trains EEG nor guarantees any downstream improvement.
- Caption quality issues and train image semantic ambiguity remain; canonical concept prompts test a different retrieval task from image-specific captions.
- Original EEG source stimulus IDs and InternViT extraction row identity remain incompletely independently verified. The provenance audit only verifies the image metadata/BLIP source against legacy catalog/strings.
- The historical 06b image normalizer included dev rows; no official score or new generalization claim is made.
