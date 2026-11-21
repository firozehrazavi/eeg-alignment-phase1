# CPU-only Image -> Caption feasibility gate (NO model training)

Purpose: decide if a visually anchored caption mapping is justified BEFORE
adding a new EEG architecture component. Reuse the historical 06b image
projection (no new EEG training), train-only BLIP captions encoded with CLIP
ViT-B/32 and the same train-fit text standardization as 06c.

Script: `diagnostics/image_caption_bridge_gate.py`

## Fixed scientific protocol

- Subject 01, seed 0, **official training partition only**.
- Original 06b image projection from verified baseline checkpoint.
- Fit rows 0-8 of each of 1,654 concepts = 14,886 images.
- Development queries: image 9 of each concept = 1,654, using **the same**
  1,654 caption candidates and duplicate-aware caption ID matching.
- Fixed ridge regularization ratio 0.05 of image-fit feature Gram trace / width.
  No tuning on 1,654-way validation.
- Fit image -> caption ridge on real paired rows; repeat with a fixed
  cross-concept shuffled caption pairing from training rows only.
- Report Top-1, Top-5, matched control deltas, query-paired bootstrap 95% CI.
- Additional stricter check: withhold 165 entire concepts from ridge fitting,
  then query 165 held-out concepts against the SAME 1,654-candidate caption
  gallery. Train and validation rows do not overlap in either check.
- Optional low-cost sanity: caption self-retrieval (not a model capability)
  is omitted because it cannot establish image-caption correspondence.

## Predeclared interpretation

- Strong positive result / proceed to ONE image-anchored EEG-caption bridge:
  Seen-concept real Top-1 >= 10%, B-real minus shuffled >= 5 percentage
  points with paired 95% CI lower > 0, AND concept-heldout real minus
  shuffled 95% CI lower > 0. These are screening thresholds, **not proof**
  of publication-worthiness or expected EEG->Caption improvements.
- If seen-concept passes but heldout fails: image/text mapping may memorize
  concepts, so do NOT claim generalization; examine evidence before a new GPU.
- If paired real-vs-shuffled improvement is negligible: NO-GO to adding the
  bridge. Re-audit caption targets, false negatives and EEG/image/text row
  identity first.

## Critical reproducibility limitations

- Historical image normalization was fitted on the entire **training partition**
  including development queries; thus neither evaluation is a clean inductive
  external test.
- Caption cache provenance declares row order based on meta_v6 train_files
  and train_img_files but acknowledges that independent EEG/image row
  identity is **not established**. Numerical success does not prove identity.
- The concept-heldout check still shares the underlying image-normalization
  statistics; it is only an additional diagnostic, not official unseen-concept
  performance.
- Do not open official test files, score 200-way official test, adjust
  regularization based on heldout results, overwrite old Drive outputs,
  rerun historical 06b training, or claim SOTA from this gate.

## Execution from the existing CPU Colab session

```python
import subprocess, sys
subprocess.run(
    ['git','-C',str(REPO),'pull','--ff-only','origin',
     'research/live-caption-grad-bd-20261121'], check=True)
subprocess.run(
    [sys.executable,'-m','unittest','discover','-s','tests',
     '-p','test_image_caption_bridge_gate.py','-v'],
    cwd=str(REPO), check=True)
subprocess.run(
    [sys.executable,str(REPO/'diagnostics/image_caption_bridge_gate.py')],
    cwd=str(REPO), check=True)
```

Use CPU runtime; the script prints JSON only, with provenance and elapsed
CPU seconds. It never writes model/checkpoint/Drive files.
