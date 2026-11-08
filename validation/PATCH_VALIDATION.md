# Patch experiment validation — 2026-11-08

- All 25 unittest tests passed on Python 3.12.5 / torch 2.9.1+cpu / NumPy 2.3.5.
- Eight new tests include two independent complete three-arm synthetic LOSO
  runs with equal results, explicit resume, 200-way galleries, deletion of
  held-out training EEG, adapter gradients and checkpoint-chunk equivalence.
- A synthetic preparation test checks authoritative row selection, repetition
  averaging, posterior-channel selection and preservation of pooled caches.
- All Colab code cells compiled as Python; cells have no stored executions or
  outputs. GPU dependency installation and extraction remain untested here.
- Full configs cover 10 subjects × 3 seeds, with separate diagonal and
  multi-positive objectives and a strict-preprocessing guard.
- The historical notebooks, recorded CSV results, original baseline module
  and prior Phase 1 tests are unchanged.
- No real EEG, pooled features or native patch caches exist in this checkout.
  No real-data training or new retrieval accuracy was measured. The synthetic
  test scores are not research results.
- Exact required Drive paths and resource estimates are in
  `docs/PATCH_EXPERIMENT.md`; the new Colab notebook performs preflight checks.

Commands:

```text
python -m unittest discover -s tests -v
python -m phase1.patch_prepare --help
python -m phase1.patch_cache --help
python -m phase1.patch_experiment --help
```

The pilot uses a subset and corrected evaluation protocol. It does not replay
the historical 78.8% or establish a SOTA baseline. Real Colab runs are the next
validation gate, followed by full within-subject/LOSO experiments.
