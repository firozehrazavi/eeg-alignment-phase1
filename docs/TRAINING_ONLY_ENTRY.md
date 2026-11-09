# Training-only entry

11_training_only_profile.ipynb pins code da4ffe964869a0c368a9b2136b575eca221dc41e.
It never invokes extraction or downloads model weights. Run its first cell on CPU,
then rerun that cell after GPU reconnect before runtime setup/staging/profile.
Prepared inputs stay read-only. Staging copies only manifest-listed arrays and token
completion metadata to a new local directory. About 13.4 GiB token space plus other
arrays and 10% reserve is required. Failed staging can be discarded by reconnecting;
Drive inputs remain the source. No old local cache is reused silently.

Preflight checks headers, all completion flags, token/catalog identity, exact pilot
shapes, disk space and existing checkpoint provenance/record structure. It does not
scan all token payloads; the runner checks sampled pooled/token consistency and
hashes recorded inputs. Full GPU precision compatibility is checked by the runner.
New output is training_only_sub01_profile_v1; existing older-code experiments are
not overwritten or migrated. Interrupted profile resumes at epoch one only for
unfinished arms. Completed arms no longer advance another epoch on profile resume.

Resource caps and persistent child logs are retained; enter actual CU balance/rate.
Latest epoch checkpoints include optimizer/scheduler/scaler/RNG and best validation
state. An interrupted partial epoch reruns. Profile bypasses target EEG evaluation
and emits no test results. No full-run command is supplied. Training-only setup
uses installed Colab torch/numpy, records pip freeze and runtime precision; GPU
compatibility and wall-clock throughput remain private-runtime checks.

Local suite: synthetic profile rerun leaves checkpoint hashes unchanged and emits
no test results; completion-cache guards from existing tests, new disk preflight
checks and notebook gate checks pass. No real-data training or accuracy is claimed.
