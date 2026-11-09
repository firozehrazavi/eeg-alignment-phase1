"""PyTorch-only backend selection and offline import/processor smoke test."""
import os


def configure():
    # Transformers reads these at import time. Always use a fresh child process.
    os.environ.update(USE_TORCH='1',USE_FLAX='0',USE_TF='0',USE_TORCH_XLA='0')


def smoke():
    configure()
    import sys
    import json
    import importlib.metadata as md
    import numpy as np
    import torch
    import torchvision
    import timm
    import einops
    import accelerate
    from PIL import Image
    from transformers import AutoModel, CLIPImageProcessor
    from transformers.utils import is_flax_available, is_tf_available
    if is_flax_available() or is_tf_available():
        raise RuntimeError('Optional backend already initialized; use a fresh subprocess')
    processor=CLIPImageProcessor(size={'shortest_edge':448},crop_size={'height':448,'width':448})
    pixels=processor(images=Image.new('RGB',(448,448)),return_tensors='pt')['pixel_values']
    if pixels.shape!=(1,3,448,448) or not torch.isfinite(pixels).all():
        raise RuntimeError('Image processor smoke failed')
    if any(k=='jax' or k.startswith('jax.') or k=='flax' or k.startswith('flax.') for k in sys.modules):
        raise RuntimeError('Unexpected JAX/Flax import')
    result={'status':'offline_import_smoke_passed','python':sys.version,
        'versions':{p:md.version(p) for p in ['numpy','torch','torchvision','transformers','tokenizers','timm','accelerate','huggingface-hub']},
        'pixels_shape':list(pixels.shape)}
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':smoke()
