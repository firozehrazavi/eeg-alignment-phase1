# Section B review

CPU preflight opens only manifest/catalog/state JSON and memory-mapped NPY headers.
It checks the confirmed sub-01 17x250 EEG layout, five 3200-wide pooled layers,
200-way catalog, completion records, configuration and concept-disjoint split.
It does not scan payloads or verify large-file hashes; preparation already does so.
Patch tokens are intentionally absent at this stage. Training requires completed
L28 token arrays and compares sampled token means with the old pooled cache.

Fixed stale GPU code checkout and stale local staging reuse: each now gets a fresh
runtime directory. Drive caches/checkpoints remain the persistent source. On a
reconnect, completed tokens are staged again, never silently reused from an old
local staging directory. Preflight runs before extraction as well as on CPU.

Memory: a 6B model's FP16/BF16 weights alone are approximately 12 GB decimal before
layer truncation; extraction truncates to layer28 before CUDA transfer. Loading
still needs CPU RAM and model download disk space. Activations, attention buffers
and allocator overhead make 16 GiB GPU fit uncertain; 24+ GiB is a more comfortable
planning target, not a guarantee. Batch1 extraction and OOM retries are enabled.
A 64x1024x3200 FP32 patch batch alone is 0.78 GiB, before gradients/adapter states.
Mixed precision and adapter chunks mitigate activations, not all persistent memory.
Training fit must be measured; no fixed GPU or throughput is assumed.

Smoke procedure: first run the new CPU preflight cell only (skip Section A).
After GPU approval, run runtime/setup and prepared-input checks, enter actual CU
balance/rate. Enable only APPROVE_EXTRACTION_PROBE for 32 images, batch1. Stop
before remaining extraction. Model download/load can dominate this short probe.
Estimate remaining extraction from throughput.json, with the notebook 50% allowance.
No training smoke is possible until tokens are complete. After separately approving
remaining extraction, the one-epoch-per-arm profile is the next training gate;
it is not a single-batch smoke test. Leave five-epoch approval false.

Checkpoints live in Drive pilot_sub01_200concepts_run1. Latest epoch checkpoints
use temporary-file replacement and include optimizer, scheduler, scaler, RNG,
best validation state and adaptive chunk. Resume verifies code/config/input hashes;
changes require a new output directory. Interrupted partial epochs rerun from the
last completed epoch. Model-loading OOM cannot be solved by batch retries.
CU caps terminate subprocesses and may lose the current epoch, not saved epochs.
Runtime estimates require the actual probe and epoch timings. Local tests cannot
validate Colab remote model loading, GPU precision/VRAM, Drive throughput or billing.
