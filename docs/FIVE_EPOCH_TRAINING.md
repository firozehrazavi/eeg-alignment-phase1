# Controlled five-epoch training

Notebook 12 pins code 1958251c2af98d8d065db9b84a7801b433effa40. It adds a
train-only switch and per-epoch telemetry. This changes the runner code hash relative
to the profile's pinned da4ffe96 code. Strict resume therefore rejects the old
profile; notebook checks explicitly, preserves it, and uses controlled_sub01_five_epochs_v1.
Other compatibility failures are not swallowed. No optimizer migration is claimed.
New-run interruptions resume latest epoch + 1, including all optimizer/scheduler/RNG
state. Regression verifies completed training reruns do not change checkpoint hashes.

Configuration remains five epochs, seed0, batch64, baseline/mean/patch, same concept
split and schedule. No early test evaluation: train-only skips the entire test branch
for every arm. TRAINING_COMPLETE is distinct from COMPLETE. There is no evaluation
cell; separate approval is required. Latest checkpoints are written atomically each
epoch in Drive, with loss, validation Top-1, seconds and peak allocated GPU bytes in
history and epoch_timing.jsonl, also streamed to persistent logs.

Observed profile times sum to 111.91 seconds/epoch across arms. Five epochs are about
9.3 minutes compute if throughput holds; a 50% allowance gives about 14 minutes,
excluding staging, hashing, initialization and Drive I/O. This is not a guaranteed
runtime. CU estimates use entered actual rate, cumulative caps and session time
checks. GPU free memory is checked and monitored; OOM chunk retries remain enabled.
CPU preflight validates manifests, completed token flags, staging space and checkpoint
compatibility. Original EEG/images/features and prior profile remain unchanged.

Local synthetic regression validates train-only has no test outputs, repeated resume
has no extra epochs, and separately requested evaluation reproduces the unchanged
reference result. No real five-epoch training or final accuracy was run locally.
