# Missing preparation outputs investigation

At commit 33d8a2bb Section A uses an absolute PREP path ending in
patch_phase1/pilot_sub01_200concepts_v1. patch_prepare has one early return:
it requires a completed dataset.json whose hash matches the ownership state,
and checks every recorded completed file. A missing manifest cannot trigger that
return. A fresh run creates the directory before loading/converting arrays.
The CLI calls prepare normally. No no-op success was reproduced locally.

The notebook did trust subprocess exit status without checking resulting files.
That is fixed: both preparation return paths and the notebook independently check
the manifest, catalog, every referenced pooled/subject file and completion hashes.
The reason for the reported absent directory is not established: private runtime
stdout, actual checkout SHA and mounted filesystem were not inspected here.
Do not infer that valid preparation exists from the old success message alone.

Run 10c_cpu_preparation_recovery.ipynb in CPU High-RAM. It loads the existing
successful report and runs preparation only, omitting the previous full EEG
inspection/inventory pass. Source hashes are still recomputed to detect changes.
Owned partial artifacts are reused via --resume; an absent directory is created.
Unknown or corrupt outputs fail closed. No original data are changed. If preparation
never happened, conversion remains necessary; this is not an instant recovery.
The explicit output verification prints the absolute directory and checked count.
Local synthetic tests exercise normal completion, missing manifest/referenced
output failure and resume guards. Private full-scale execution remains necessary.
