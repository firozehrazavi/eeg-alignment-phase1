# dev branch — work in progress (not for citation)

Snapshot 2026-11-04. THINGS-EEG2, 200-way zero-shot retrieval, Top-1 (chance 0.5%).

| notebook | target | within-subject (10 subj) | LOSO direct (10 folds) | LOSO + label-free recovery |
|---|---|---|---|---|
| 06 | DINOv2-giant, layer 20 / route 20–36 | 70.5 (seed 0) | 21.6 | 35.3 |
| 06b | InternViT-6B, route_c layers 20–36 | 78.8 (3 seeds) | 26.2 (seed 0) | 44.7 (seed 0) |
| 06b | InternViT-6B, layer 28 | 77.9 (3 seeds) | 24.6 | 39.3 |

- Within-subject: 17 posterior channels. LOSO: 63 channels, train on 9 subjects.
- Recovery = SCORE-style one-step coordinate recovery (moment matching, CSLS, 64 MNN landmarks, regularised Procrustes); label-free but transductive (uses the unlabelled test EEG of the new subject). CSLS columns are transductive as well.
- 07: landmark-selection rules for recovery (running).
- Results CSVs: Drive `eeg_results/v6`, `eeg_results/v6b` → copy to `results/` here.
