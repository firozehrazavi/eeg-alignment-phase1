# PyTorch-only extraction environment

The reported traceback enters JAX during CLIP processor import. Transformers 4.45.2
reads USE_FLAX, USE_TF and USE_TORCH at module initialization:
https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/utils/import_utils.py
Fresh logged child processes now force USE_FLAX=0, USE_TF=0, USE_TORCH=1 and
USE_TORCH_XLA=0. Direct extractor invocation also configures before Transformers
imports. No JAX uninstallation or NumPy attribute monkeypatch is used.

NumPy 1.26 is retained for Python <3.13; Python >=3.13 uses NumPy 2.2.6.
NumPy 2.2 supports Python 3.10-3.13:
https://numpy.org/doc/2.2/release/2.2.0-notes.html
This does not certify the whole Colab environment or future Python versions.

Setup runs a fresh-process offline import/processor smoke before model resolution
or weight download. It imports torch, torchvision, timm, einops, accelerate,
AutoModel and CLIPImageProcessor, checks synthetic RGB processing and confirms
JAX/Flax were not imported. Exact versions and Python are printed to persistent logs.
The existing Colab torch/torchvision pair is preserved; ABI/operator mismatches fail
at this inexpensive gate rather than after a model download. Parent notebook
imports may remain cached after pip changes; extraction/smoke use fresh processes.

Remaining path audit: catalog and CUDA checks precede output creation; processor
and model use the same recorded Hub revision; flash attention is optional/disabled;
remote model code loads with trust_remote_code; layer28 is selected before CUDA;
FP16/BF16, hook shape checks and batch1 remain. The offline smoke cannot establish
remote-code API compatibility, access/network/disk availability or actual GPU fit.
The approved 32-image probe remains necessary for those. No full extraction or
training was run. No prepared EEG/feature files were modified.

Local regression suite checks backend configuration, ordering and notebook gates.
The real import smoke was attempted locally but stopped because torchvision is
not installed (Python 3.12.5 CPU environment); Python 3.13/L4 execution is not
claimed. Rerun both updated Section B setup cells; stop if smoke fails and share
the logged traceback. Only after smoke passes rerun the 32-image probe. Skip A.
