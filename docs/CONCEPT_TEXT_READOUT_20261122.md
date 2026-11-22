# Caption-target quality diagnostic — frozen 06b (22 November 2026)

**Status:** source provenance audited on real user-owned Drive artifacts; frozen EEG→concept-text retrieval **NOT MEASURED** until the single CPU Colab run. This is not a trainable new architecture and does not approve GPU.

## Research question
Is the ~1.21% frozen EEG→BLIP-caption retrieval failure partly attributable to noisy/overly specific image-generated captions rather than an absence of decodable object concepts in the preserved EEG→image backbone?

This is a **diagnostic**, not a comparable benchmark of EEG→Caption. Known-concept label retrieval is easier than recovering an image-specific caption; a concept advantage alone does not demonstrate direct caption recovery or tri-modal novelty.

## Verified provenance facts (user-owned Drive, local CPU audit)
- Official \`THINGS-EEG2/images_set/DESCRIPTION.md\`: 1,654 sorted training concepts × 10 images; official preprocessed EEG first dimension is documented in that same order.
- \`THINGS-EEG2/images_set/image_metadata.npy\` has 16,540 training files and concepts.
- \`eeg_results/cache/meta_v6.pt\` has **exactly equal** \`train_files\` in the same 16,540-row order. The source file SHA-256 matches the pinned caption-provenance record: \`d5d53527851d0ffbdfbc2d3f27946090cd3a3206bb10e5837c979f8b62206e9f\`.
- \`eeg_results/cache/captions_blip_large.json['train']\` is **exactly equal row-for-row** to \`caption_ids_train_only.json\` (all 16,540 rows); pinned source SHA-256 \`67db4197c2aa0f75a21261fa97fd97d9cde35648b4d191a4e3ff723cffd94014\`.
- Across the 16,540 caption strings, **8,583 (51.89%)** have a literal spelling match for the canonical concept and **467/1,654 concepts** have no literal name match in any of ten captions. This is *not* a caption-error rate (valid synonyms and paraphrases are missed). Some obvious misidentifications merit review: aardvark images described as an armadillo or baby elephant.
- Original legacy caption encoder: \`open_clip ViT-B-32 pretrained=openai\` with exact original revision **not pinned**. New text encoder hash/version are recorded, **not** claimed to match original unverified weights byte-for-byte.
- **Not independently verified:** stimulus ID for the derived 06b EEG cache and independent stimulus-row identity of the InternViT feature extraction. Matching lengths, filenames, and provenance records are not a substitute.

## One pre-registered CPU experiment
- Input: frozen historical 06b checkpoint; subject 01 seed 0; exact 14,886-fit and 1,654-development rows from the training partition.
- New fixed target: \`a photo of a {concept}\`, for all 1,654 concepts, embedded once using OpenCLIP ViT-B-32 / \`pretrained=openai\`, CPU. Labels use the **training-image metadata**, not test categories.
- The same fixed-regularization affine Ridge implementation is fit on optimizer-train rows, with frozen EEG embeddings, to (a) canonical concept embeddings, (b) the legacy BLIP caption embeddings. The caption standardization is optimizer-fit-only, preserving the previous readout convention.
- Negative control: cross-concept shuffle of the concept-text training pairs using the existing fixed derangement.
- Evaluation: paired development queries, each in a 1,654-candidate gallery, Top-1/Top-5, counts, and paired query bootstrap CIs. Distinguish exact-caption duplicate positives from unique canonical concept candidates.
- A positive paired CI for concept versus both the shuffled arm and BLIP indicates **retrievable known-concept information**, not comparable image-specific caption quality. It licenses a carefully controlled further *hypothesis*, **not GPU training**, not 200-way official-test access, and not a SOTA claim.
- If no robust gain, stop this target-quality hypothesis; do not tune prompt formats or revisit architecture on the same development partition.

## CPU regression evidence
9 synthetic-data CPU unit tests passed on 2026-11-22 (no full model checkpoint available in local execution environment). They cover fixed 1,654×10 ordering, failures on swapped/duplicate rows, source hash tampering, caption-string tampering, explicit unknown EEG stimulus IDs, lexical caveats, and CPU concept encoding with a fake OpenCLIP model.

\`\`\`bash
python -m unittest discover -s tests -p 'test_concept_text_readout_gate.py' -v
\`\`\`

## Run in the *existing* Colab CPU session (no new notebook)
From the already mounted Drive, switch to branch \`research/concept-text-audit-20261122\` and run once:

\`\`\`bash
cd /content/eeg-alignment-phase1-06c-live-caption
git fetch origin research/concept-text-audit-20261122
git switch --detach origin/research/concept-text-audit-20261122
python -m pip install -q open_clip_torch
python diagnostics/concept_text_readout_gate.py --output /content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/concept_text_gate_reports/20261122_sub01_s0_fixed.json
\`\`\`

The report is **exclusive-create**: if that path already exists, choose a fresh timestamped name rather than overwrite. No GPU, no test set, no checkpoint mutations. The first OpenCLIP weight fetch may require network access; if unavailable, record \`NOT_MEASURED\`, do not substitute an unrelated encoder or silently claim equivalence.

## Unchanged safeguards
- Historical 06b normalization, layers, checkpoints and train/validation split remain untouched.
- PR #6 LIMA remains NO-GO.
- No alteration of main, no tuning on dev, no new notebook, no official 200-way evaluation.
- Model source audits are technical evidence only; actual direct-caption performance must be measured separately on a strict image-gallery-free path.
