# Observed sub-01 input convention

The user ran the private-Drive diagnostic in Colab and reported float64 dictionary EEG:
training (16540,4,63,250), test (200,80,63,250), 63 channels, and 300 time values
from -0.2 to 0.996 seconds at 0.004-second intervals, with zero at index 50.
They also observed 16,740 images under THINGS-EEG2/images_set, image metadata,
and existing sampled cached paths relative to THINGS-EEG2. These are user-observed
facts, not a local inspection of private data or proof that every cached path exists.

## Implementation evidence

The [dataset preprocessing implementation](https://github.com/ncclab-sustech/EEG_Image_decode/blob/main/EEG-preprocessing/preprocessing_utils.py)
was inspected on 2026-11-09. In `epoching`, lines 70–79 create the baseline-containing
epoch and retain its resampled time axis. Line 101 removes the first 50 samples from
EEG, while line 105 returns the unchanged time axis. `save_prepr` retains that axis
in both dictionaries (lines 225–242 and 270–283). Therefore the effective EEG axis
is metadata indices 50:300, 0 through 0.996 seconds. No second EEG crop is warranted.
This explains the observed metadata, but does not establish the complete processing
history of the distributed binaries; the benchmark_preprocessed regime remains.

## Validation and inventory

Validation accepts either sample-aligned 250 Hz metadata or exactly the observed
300-point -0.2..0.996 axis with 250 EEG samples. Other offsets, lengths, rates,
nonfinite values, and nonuniform intervals fail. Reports retain both metadata and
effective-axis ranges and the 50-sample metadata offset. Existing EEG shape, float
dtype, chunked finiteness, channel order, byte-size and hash checks remain active.

The previous inventory did not use bounded scans. A null root could mean missing
metadata, cache row-order disagreement, or any missing referenced image; its old
summary did not distinguish them. The supplied observations do not establish which
condition caused the earlier failure. The new inventory reports counts and specific
reasons, accepts either the dataset parent or images_set directory, and checks all
16,540 training and 200 test paths without scan limits. Only the displayed missing
path examples are capped. Metadata ordering, duplicate paths and pooled feature
shapes remain checked. Bounded diagnostic scan counts never determine readiness.

## Safe resume and evidence boundaries

Use `notebooks/10b_cpu_verify_observed_inputs.ipynb` with accelerator None and CPU
High-RAM. It clones code into a new temporary runtime directory and runs only
`--verify-existing-sub01`; no dataset download or preparation/training runs. Existing
EEG, images and caches are read-only. New verification reports and the existing
latest-report pointer are written by the verifier. Missing files fail without download.

Local regression tests use synthetic float64 data with reduced image counts and
real repetition/channel/time axes, both pickle container types, full-size metadata
lists with mocked image existence, and small-storage expanded feature tensors.
They verify exact-path checks including the last training image, both root layouts,
row-order rejection, and time-axis rejection cases. No private EEG/image arrays
were downloaded, uploaded or evaluated locally. The corrected verifier still needs
to run against all real files in Colab; no training or accuracy claim is made.
