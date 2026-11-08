# CPU-only sub-01 acquisition and preparation

Start section **A** of `notebooks/10_rpa_patch_pilot.ipynb` with Colab's
hardware accelerator set to **None**. Use CPU High RAM where available.
For resuming after the dictionary-loader error, keep
`VERIFY_EXISTING_EEG_ONLY = True` (the updated default). It revalidates existing
files without downloading anything. Missing files or the saved download plan
produce a clear stop. For an initial download rather than verification, the
original `--download-sub01` mode is still available explicitly.
Do not allocate a GPU for this stage. It downloads only:

```text
/content/drive/MyDrive/THINGS-EEG2/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_training.npy
/content/drive/MyDrive/THINGS-EEG2/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_test.npy
```

The [source folder](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main/Preprocessed_data_250Hz/sub-01)
lists approximately 8.34 GB training plus 2.02 GB test. Exact byte counts and
available LFS SHA-256 hashes are resolved from Hub metadata at runtime, then
pinned to an immutable dataset commit in `THINGS-EEG2/pilot_sub01_download_plan.json`.
The Hub download implementation resumes incomplete transfers in the persistent
Drive staging directory; it never requests a full repository snapshot.

## Read-only inventory first

Before any Hub metadata/data request, the script checks:

- `MyDrive/eeg_results/cache/meta_v6.pt` and `feat6_internvit6b.pt`.
- The five pooled route_c feature layers, expected train/test row counts and
  3200-dimensional vectors, loaded on CPU with mmap where supported.
- Image metadata under `MyDrive/THINGS-EEG2/images_set`,
  `MyDrive/eeg_results/images_set`, and `/content/EEG_Data/images_set`.
- Original image paths referenced by `meta_v6.pt` and exact agreement between
  cached image order and the authoritative image metadata.
- Whether each requested EEG file already exists.

Edit `IMAGE_ROOTS` if the existing image tree lives elsewhere. No image archive,
pooled feature cache, model weights or other subject is downloaded automatically.
Existing files are not moved. Preparation can use a separate `--image-root`.
If using `/content/EEG_Data`, remember that runtime-local images may disappear
when switching to a GPU runtime; use a persistent Drive image tree for the
subsequent pilot. The report lists unresolved image/cache paths and prevents
preparation from being called ready when those inputs are missing or incompatible.

## Verification and provenance

Existing destination EEG files are verified rather than overwritten. A size
or available remote SHA-256 mismatch stops execution and preserves the file.
New transfers go under `THINGS-EEG2/.pilot_downloads/<dataset-commit>/` and are
renamed into place only after byte verification. Hashes are computed locally
even if the Hub does not supply a comparable remote SHA-256.

Each file is then loaded sequentially on CPU, checked for finite floating-point
values and these expected shapes from the
[dataset card](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/blob/main/README.md):

| File | Expected shape |
|---|---|
| Training | `(16540, 4, 63, 250)` |
| Test | `(200, 80, 63, 250)` |

The EEG files may load directly as dictionaries or as scalar NumPy object
arrays containing dictionaries; both verification and subsequent preparation
accept these two containers. All validation checks still apply to the contents.
The files contain pickle-based dictionaries from the selected trusted
dataset mirror. Verify bytes before loading; do not substitute an untrusted
pickle. Channel names and 250 Hz time spacing are checked when embedded metadata
is present; absent metadata is recorded as unverified, not invented. A conflicting
channel order/time axis stops preparation. The loader checks free RAM against
2.2 times the file size (roughly 18 GiB for the larger file); insufficient RAM
leaves the verified download intact for a later High-RAM CPU session.

Timestamped reports under `THINGS-EEG2/pilot_input_reports/` retain byte sizes,
local/remote hash status, actual shape/dtype, finite-value checks, available
channel/time metadata, dataset revision and source-card URL. A latest-report
pointer supports the notebook; older reports are preserved.

Origin and file integrity are distinct from a strict evaluation guarantee.
The mirror identifies THINGS-EEG2-derived, downsampled data, but does not establish
that whitening or other upstream fitted statistics are target-independent.
The report therefore retains `benchmark_preprocessed` and
`upstream_target_independent: false`. Downloading these files does **not** make
strict inductive LOSO valid. No additional z-score is applied by acquisition.

## Resume and prepare the pilot

Rerun section A after a disconnect. Its fixed allowlist contains only the two
sub-01 paths; other subjects cannot be added through a modified download plan.
Completed EEG files are rechecked and skipped. A partial transfer is resumed
by the pinned Hub downloader. A process lock prevents concurrent invocations
of this acquisition step in the same runtime.

After all required inputs pass, preparation creates a new pilot dataset under
`MyDrive/eeg_results/patch_phase1/pilot_sub01_200concepts_v1`. A preparation
ownership marker records input hashes/configuration and hashes of each completed
output. `--resume` skips completed outputs only when those hashes match; an
interrupted EEG-average array restarts from its temporary file, while completed
arrays are retained. A changed input/configuration or unowned existing directory
is rejected. The finished manifest links the acquisition provenance and hashes.
No historical `_z16.npy`, `.pt`, notebook or result is overwritten.

The CPU stage stops here. Section B has a new default-false
`APPROVE_GPU_SESSION` gate, in addition to the existing extraction/training
approvals. Verify the report and obtain approval before selecting a GPU runtime.
No ten-subject download or full training command is included.

## Validation performed locally

43 CPU tests pass, including direct-dictionary/scalar-object-array regression
fixtures, unchanged content validation, no-download verification, interrupted transfers, checksum/size
failures, non-overwrite behavior, subject allowlisting, shape/channel/time checks,
RAM refusal, read-only inventory and notebook ordering. The existing preparation
test now also interrupts processing and verifies exact resumed outputs.
All nine notebook code cells compile. These are synthetic/offline tests: the
10.4 GB real files have not been downloaded or verified in this local workspace,
and the user's Drive has not been accessed from here.
