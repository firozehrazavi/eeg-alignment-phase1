# Frozen EEG → shared-CLIP visual teacher: single CPU compatibility gate

**Branch date:** 22 November 2026. **Pre-run status:** UNMEASURED. Run this once; do not tune prompts or the split.

## Motivation and scope
- 06b historical InternViT EEG→Image route remains untouched.
- Real teacher preflight on 256 TRAIN-fit images: **OpenCLIP Image→BLIP-Caption 90.625%**, Image→Concept 84.76562% Top-1 (256-way), both versus 0% shuffled. **This was an image-only/text-only test, NOT EEG performance.**
- Frozen EEG→canonical concept-text Ridge previously **failed**, 0.18138% vs 1.20919% frozen EEG→BLIP-caption Ridge (1,654-way development; different target tasks).
- This preflight asks whether **paired, frozen EEG embeddings** can transfer into the *visual* side of the shared OpenCLIP model and retrieve corresponding image-specific BLIP captions **without an image gallery at inference**.

## Exact design
1. Use the **same fixed 256 distinct concepts** as the previous successful shared-CLIP teacher preflight (NumPy generator seed 20261122), with **fit EEG/image at row 10*c** (first exemplar) and **query EEG/caption at row 10*c+9** (last exemplar). This is the historical 9/1 within-training-partition split; 256 query/candidate images and captions. Existing EEG checkpoint is frozen and was trained on the full permitted historic fit split, so concepts are known, and this is *not* unseen-concept generalization.
2. Read the original first-exemplar JPEGs *only* from the official TRAIN directory. Encode those 256 fit images, 256 fit BLIP strings and 256 last-exemplar dev BLIP strings using **one** frozen, weight-hash-pinned OpenCLIP ViT-B/32 OpenAI image-and-text checkpoint. **Do not encode or open the query/development JPEGs**. They are used only as identities for caption text. Do not use any official test file.
3. Recover frozen EEG embeddings for the 256 fit and 256 development rows from signed, historical \`sub-01\` seed 0 checkpoint after verifying manifest/baseline and the official train-EEG cache hash. The \`06b\` checkpoint, EEG encoder and image projection head remain unchanged.
4. Exactly three matched train-only fixed-ridge readouts:
   - **A (hypothesis):** EEG-06b → OpenCLIP **visual** embedding (fit on 256 first images) → direct OpenCLIP text gallery of the *last* exemplar's BLIP captions
   - **B:** EEG-06b → OpenCLIP **text** embeddings of the same 256 fit image-specific captions → same dev caption gallery
   - **C:** EEG-06b → original \`captions_train_only.npy\` caption embeddings, on the same 256 fit identities → its own encoded 256 dev captions (same caption strings, possible legacy revision difference)
   - **Control D:** shuffle the first-exemplar **image feature target rows** cyclically across different concepts before fitting A. Keep actual dev captions/identities fixed
5. All use identical 256 EEG fit rows and 256 EEG dev queries; same image-specific target strings and duplicate-caption-aware positive criterion. Top-1, Top-5, paired bootstrap CI and counts are reported. No temperature/lambda/prompt sweep.
6. **Fixed positive gate:** A Top-1 ≥5% on 256 options **and** paired Top-1 CI lower bounds A−D, A−B, A−C all >0. A pass means *candidate compatible visual teacher*, **NOT** GPU authorization or tri-modal novelty. Any fail means NO_GO/INCONCLUSIVE and no architecture pivot. This gate is deliberately stringent; it tests image-mediated readout superiority.

## Verification, evaluation limitations
- Repository unit tests include split/order safeguards, 256-way paired retrieval, label duplicates, shuffled controls, shape/finiteness guards. **Test execution must occur in Colab before real-data run.**
- Baseline *1.20919%* referred to **a separate 1,654-way, 14,886-fit** experiment and cannot be directly compared numerically with this 256-way, 256-fit diagnostic. Only A/B/C/D scores **inside the same JSON report** are paired here.
- EEG stimulus IDs and InternViT feature extraction IDs are not independently established; matching metadata & hashes is insufficient proof. Historical InternViT normalizer included dev images. This pilot does not establish an unbiased official THINGS-EEG2 benchmark, a 10-subject score, novel concept generalization, or joint EEG–Image–Text learning.
- A successful shared teacher score does **not** imply EEG information is sufficient. This gate tests precisely that missing compatibility.
- No new full-training feature cache, no checkpoint mutation, no GPU or test set.

## One CPU cell in the **existing 06c Colab**

~~~python
import sys, subprocess, importlib.util
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

drive.mount("/content/drive")
repo = Path("/content/eeg-alignment-phase1-06c-live-caption")
branch = "research/frozen-eeg-sharedclip-compat-20261122"

def call(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)

if not (repo / ".git").is_dir():
    call(["git", "clone", "--depth", "1", "--branch", branch,
          "https://github.com/firozehrazavi/eeg-alignment-phase1.git", str(repo)])
else:
    call(["git", "fetch", "origin", branch], repo)
    call(["git", "switch", "--detach", "FETCH_HEAD"], repo)

if importlib.util.find_spec("open_clip") is None:
    call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], repo)

call([sys.executable, "-m", "unittest", "discover", "-s", "tests",
      "-p", "test_frozen_eeg_sharedclip_compat_gate.py", "-v"], repo)

report_dir = Path("/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/frozen_eeg_sharedclip_compat_reports")
report_dir.mkdir(parents=True, exist_ok=True)
output = report_dir/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_compat.json")
call([sys.executable, "-u", "diagnostics/frozen_eeg_sharedclip_compat_gate.py",
      "--output", str(output)], repo)
print("REPORT SAVED:", output)
~~~

**Report first and only:** Share \`results\` from the resulting JSON file. If a safe-guard fails, stop, report the exact error, and correct the code rather than disable provenance verification or run GPU.
