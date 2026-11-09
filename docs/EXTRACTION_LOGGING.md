# Extraction subprocess diagnostics

Previously Popen inherited stdout/stderr: output went to the notebook kernel's OS
streams, not a file. Colab may not render those streams as cell output. No log file
was ever created. The wrapper now merges stderr/stdout into a pipe, streams lines
through Python print, and flushes an exclusive Drive log under patch_phase1/logs
before launching any child. Python children are unbuffered. Unique log names retain
previous failures. Launch errors and nonzero exits are displayed and persisted.
Periodic budget checks remain active even when the child is silent.

TOKEN_DIR is created only after imports, catalog/revision validation, CUDA checks,
runtime profiling and the initial budget check. Its absence suggests one of these
early stages or mkdir failed; it does not identify the exception. Transformers,
Pillow/timm/torchvision/torch compatibility, missing imports, catalog validation,
invalid revision, CUDA visibility, budget exhaustion and Drive permissions are
possible failure points. The next log is required to distinguish them.

Model revision resolution runs in the parent notebook via model_info before the
child; failure there appears as a notebook exception. Existing revision files are
reused. Processor/model download occurs after output-directory creation. Access,
network, disk and remote-code dependency errors at that stage should now appear
in the persistent child log. The optional HF_TOKEN warning alone does not prove an
authentication failure. Do not change credentials or model provenance speculatively.

Restart neither preparation nor full extraction. Run both Section B setup cells
from the updated notebook, then only the 32-image probe. Copy the actual traceback
if it fails, redacting any credentials or private path components. Prepared inputs
are unchanged. No actual Hugging Face model load or GPU test was performed locally.
