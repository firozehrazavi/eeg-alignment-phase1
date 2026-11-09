# Section A CPU review

Trace: CPU assertion, Drive mount, fresh runtime checkout, dependency installation,
`pilot_inputs.prepare_inputs` -> inventory -> byte/hash verification -> inspect_eeg
-> latest report -> readiness assertion -> `patch_prepare.prepare`.
Preparation hashes all source files, checks resume ownership/configuration, loads
image metadata and cached row order, selects 200 training concepts, hashes selected
images, validates disjoint catalogs and all 200 test stimuli, writes five pooled
layers per split, checks acquisition hashes, loads each EEG split separately,
averages repetitions in float32, selects posterior17, writes atomic derived arrays,
and publishes dataset.json last. No extraction/training function is called.

The actual supplied EEG axes (16540,4,63,250) and (200,80,63,250) are compatible;
metadata offset 50 is interpreted only during verification, never recropped.
Paths use THINGS-EEG2 as parent, matching the successful private 10b verification.

Fixed stale-checkout reuse by creating a fresh runtime directory on every Section A
execution and printing its SHA. Fixed metadata container handling in preparation.
Released pooled tensors before EEG loading; added a 2.2x file-size free-RAM guard.
Reduced averaging chunks from 64 to 8 images (test float64 input copy about 77 MiB
rather than 615 MiB). Whole pickled files still require RAM; select CPU High-RAM.
The earlier verification RAM guard requires roughly 17 GiB free for training.

Tests cover CPU gates, fresh checkout, metadata containers, exact row averages,
source preservation, interrupted resume, low-RAM failure and recovery, modified
configuration rejection, and catalog identity checks. Synthetic data are smaller
than real files; Colab memory pressure, filesystem throughput, permissions, and
full-scale serialization remain untested locally. No completion-time guarantee.

Run only Section A with accelerator None, High-RAM, existing-only flag True.
Keep RAW and CACHE at verified paths; preserve PREP and let ownership/hash checks
resume it. Stop at the CPU completion message before Section B. This deliberately
rechecks source hashes instead of trusting stale reports; allow time for Drive I/O.
Original EEG/images/caches remain read-only; new derived files and reports go to
owned preparation/report directories. Run one preparation process at a time.
