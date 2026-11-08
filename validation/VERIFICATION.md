# Verification record

Executed on 2026-11-08 using Python 3.12.5, PyTorch 2.9.1+cpu, NumPy 2.3.5. CUDA unavailable.

- `python -m unittest discover -s tests -v`: **17 tests passed**.
- `python -m phase1.smoke --output validation/final_synthetic_smoke`: **passed**, both objective arms trained for three synthetic epochs and their independent repeats produced identical histories.
- `python -m phase1.reconcile --output validation/historical_reconciliation.json`: **passed**; 44.70 vs 46.45 traces to different landmark budgets, with ten distinct subjects in each group.
- Before staging additions, `git diff --exit-code` and `git diff --check`: **passed**. No original tracked notebook, README or CSV changed.

This verifies code behavior, not scientific improvement. No THINGS-EEG2 training/evaluation, calibration replication, frozen-text extraction or real tri-modal training was executed. Data/cache provenance and source-only inner-loop hyperparameter selection must be completed before strict experimental claims. The new selector checks records but does not replace nested model fitting. The full-size baseline architecture is regression-tested against notebook ASTs; the executed optimizer smoke uses an explicitly smaller model. Full-run and tri-modal integration requirements are in `docs/PHASE1_PROTOCOL.md`.
