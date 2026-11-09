# CPU import gate

The standalone cpu-environment-smoke cell runs before GPU approval, installs pinned
requirements, and invokes the real import/synthetic-image smoke in a fresh process.
Both installer and smoke output are streamed and saved under environment_smoke/logs.
Only an exit-zero smoke with the expected JSON status publishes evidence. Prior
success evidence is archived before retrying so a failed retry cannot leave a stale
success pointer. Prepared EEG and feature data are not touched.

CPU and GPU checkouts explicitly detach at a9e03c549a5af55025920de6d70cd50a9d5b73b3,
which contains the extraction/backend smoke implementation. The newer notebook is
an orchestrator for that frozen code, not a claim of local Python3.13 validation.
GPU setup requires revision-matched evidence, an unchanged log, and matching
requirements digest. It repeats the import smoke in the GPU runtime because its
installed torch/torchvision may differ. No guarantee of GPU fit follows from CPU
import success. Evidence remains on private Drive, never uploaded.

Run only the CPU environment smoke cell with accelerator None. Do not run Section
A again. After it succeeds, switch GPU and run both GPU setup cells; they reuse the
saved evidence and identical code revision. Enable only the 32-image probe after
setup passes. No full extraction or training is authorized by this gate.
