"""Resumable one-layer InternViT extraction, plus validated mmap cache access.

CLI: python -m phase1.patch_cache --catalog catalog.json --output NEW_DIR
     --revision FULL_HF_COMMIT_SHA [--batch 1 --layer 28 --limit 2200]
Catalog: {train:[{id,concept,path,sha256}],test:[...]} in EEG row order.
Paths are absolute or relative to catalog. Hashes refer to original image bytes.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .core import validate_disjoint_catalogs
from .patch_resources import BudgetMeter, runtime_profile, retry_oom


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_catalog(path):
    catalog = json.loads(Path(path).read_text())
    validate_disjoint_catalogs(*[{
        'image_ids': [r['id'] for r in catalog[s]],
        'concept_ids': [r['concept'] for r in catalog[s]],
        'content_hashes': [r['sha256'] for r in catalog[s]]} for s in ('train', 'test')])
    if len(catalog['test']) != 200:
        raise ValueError('Exactly 200 unique test stimuli required')
    return catalog


class TokenCache:
    def __init__(self, root, catalog):
        self.root = Path(root)
        self.meta = json.loads((self.root/'manifest.json').read_text())
        if self.meta['catalog_sha256'] != digest(catalog):
            raise ValueError('Patch cache belongs to another catalog')
        records = read_catalog(catalog)
        self.arrays = {}
        for split in ('train', 'test'):
            self.arrays[split] = np.load(self.root/f'{split}.npy', mmap_mode='r')
            done = np.load(self.root/f'{split}_done.npy')
            if done.dtype != np.bool_ or done.shape != (len(records[split]),):
                raise ValueError('Invalid completion mask')
            if not done.all():
                raise ValueError(f'Incomplete {split} cache: {int((~done).sum())} rows missing')
            if list(self.arrays[split].shape) != self.meta['shapes'][split]:
                raise ValueError('Patch cache shape mismatch')
            expected=(len(records[split]),int(np.prod(self.meta['grid'])),self.meta['width'])
            if self.arrays[split].shape != expected or self.arrays[split].dtype != np.float16:
                raise ValueError('Expected FP16 native patch array aligned with catalog')

    def get(self, split, rows):
        x = torch.from_numpy(np.array(self.arrays[split][rows], copy=True)).float()
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite cached patch tokens')
        return x


def extract(args):
    from PIL import Image
    from transformers import AutoModel, CLIPImageProcessor
    import transformers.dynamic_module_utils as dmu
    catalog = read_catalog(args.catalog)
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Pin an immutable 40-character Hugging Face model revision')
    if not torch.cuda.is_available():
        raise RuntimeError('InternViT extraction requires CUDA; no features were fabricated')
    profile=runtime_profile();print(json.dumps(profile),flush=True)
    meter=BudgetMeter();meter.check('extraction_start',force=True)
    inference_dtype=torch.bfloat16 if profile['precision']=='bf16' else torch.float16
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(model='OpenGVLab/InternViT-6B-448px-V2_5', revision=args.revision,
                    layer=args.layer, layer_convention='hidden_states[L] = encoder.layers[L-1] output',
                    catalog_sha256=digest(args.catalog), dtype='float16', cls_removed=True,
                    inference_dtype=profile['precision'],
                    preprocessing='Pinned CLIPImageProcessor, RGB, no augmentation')
    mp = root/'manifest.json'
    if not mp.exists() and any(root.glob('*.npy')):
        raise ValueError('Orphaned arrays without provenance; choose a new cache directory')
    if mp.exists() and any(json.loads(mp.read_text()).get(k) != v for k, v in identity.items()):
        raise ValueError('Existing cache has different provenance; choose a new directory')
    processor = CLIPImageProcessor.from_pretrained(identity['model'], revision=args.revision)
    # InternViT catches optional flash_attn at runtime; transformers 4.45's
    # static import checker otherwise rejects it even with use_flash_attn=False.
    original_imports = dmu.get_imports
    try:
        dmu.get_imports = lambda path: [m for m in original_imports(path) if m != 'flash_attn']
        model = AutoModel.from_pretrained(identity['model'], revision=args.revision,
            trust_remote_code=True, torch_dtype=inference_dtype,
            low_cpu_mem_usage=True, use_flash_attn=False).eval()
    finally:
        dmu.get_imports = original_imports
    if not 1 <= args.layer <= len(model.encoder.layers):
        raise ValueError('Layer out of range')
    model.encoder.layers = torch.nn.ModuleList(list(model.encoder.layers)[:args.layer])
    try:
        model.requires_grad_(False).cuda()
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError('Backbone does not fit assigned GPU. No cache changed; request a higher-VRAM runtime.') from None
    captured = []
    def capture(module, inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    hook = model.encoder.layers[-1].register_forward_hook(capture)
    processed = 0
    inference_start=time.time()
    microbatch=args.batch
    try:
        for split in ('test', 'train'):  # Always complete the 200-image gallery first.
            records = catalog[split]
            donepath, arrpath = root/f'{split}_done.npy', root/f'{split}.npy'
            done = np.load(donepath) if donepath.exists() else np.zeros(len(records), dtype=bool)
            array = np.load(arrpath, mmap_mode='r+') if arrpath.exists() else None
            todo = np.flatnonzero(~done)
            for start in range(0,len(todo),args.batch):
                meter.check('extracting_tokens')
                rows = todo[start:start+args.batch]
                if args.limit and processed + len(rows) > args.limit:
                    return
                images = []
                for i in rows:
                    path = Path(args.catalog).parent / records[i]['path']
                    if digest(path) != records[i]['sha256']:
                        raise ValueError(f'Image hash mismatch: {path}')
                    with Image.open(path) as im:
                        images.append(im.convert('RGB').copy())
                def forward(use_batch):
                    pieces=[]
                    for offset in range(0,len(images),use_batch):
                        captured.clear()
                        with torch.inference_mode():
                            px=processor(images=images[offset:offset+use_batch],return_tensors='pt')['pixel_values']
                            model(pixel_values=px.cuda().to(inference_dtype), output_hidden_states=False)
                        if len(captured)!=1:raise RuntimeError('Expected one encoder hook output')
                        pieces.append(captured[0][:,1:].contiguous().float().cpu().numpy().astype('float16'))
                        captured.clear()
                    return np.concatenate(pieces)
                tokens,microbatch=retry_oom(forward,microbatch)
                grid = int(round(tokens.shape[1]**.5))
                if grid*grid != tokens.shape[1] or not np.isfinite(tokens).all():
                    raise ValueError('Invalid native patch grid or values')
                if array is None:
                    array = np.lib.format.open_memmap(arrpath, mode='w+', dtype='float16', shape=(len(records),*tokens.shape[1:]))
                if tuple(array.shape[1:]) != tuple(tokens.shape[1:]):
                    raise ValueError('Existing array shape differs')
                array[rows] = tokens
                array.flush()
                done[rows] = True
                with open(str(donepath)+'.tmp','wb') as f:
                    np.save(f,done)
                os.replace(str(donepath)+'.tmp',donepath)
                meta = {**identity, 'processor':processor.to_dict(), 'grid':[grid,grid], 'width':tokens.shape[2],
                        'shapes':{s:[len(catalog[s]),*tokens.shape[1:]] for s in ('train','test')}}
                tmp = root/'manifest.tmp'
                tmp.write_text(json.dumps(meta,indent=2))
                os.replace(tmp,mp)
                processed += len(rows)
                elapsed=time.time()-inference_start
                (root/'throughput.json').write_text(json.dumps(dict(images=processed,seconds=elapsed,
                    seconds_per_image=elapsed/processed,runtime=profile,microbatch=microbatch),indent=2))
                print(f'{split}: {done.sum()}/{len(done)}', flush=True)
    finally:
        hook.remove()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',required=True); p.add_argument('--output',required=True)
    p.add_argument('--revision',required=True); p.add_argument('--layer',type=int,default=28)
    p.add_argument('--batch',type=int,default=1); p.add_argument('--limit',type=int,default=0)
    extract(p.parse_args())
