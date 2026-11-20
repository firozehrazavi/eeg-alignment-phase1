"""Create a strict train-partition-only *development* manifest from VERIFIED cached subset.

No official test files are read. This 200-concept, 2000-image set was selected for
previous exploratory work and is NOT a pristine/official EEG benchmark.
"""
import argparse, hashlib, json, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch

LAYERS = (20, 24, 28, 32, 36)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def prepare(source, output):
    src=Path(source).resolve();dst=Path(output).resolve()
    if dst.exists(): raise FileExistsError(str(dst))
    data=json.loads((src/'dataset.json').read_text());state=json.loads((src/'preparation_state.json').read_text())
    cat=json.loads((src/'catalog.json').read_text())
    caps=json.loads((src/'captions_blip_large.json').read_text())
    idx=data['selection']['train']
    n=len(idx)
    if n!=2000 or len(set(idx))!=n: raise ValueError('Unexpected development selection')
    if len(cat['train'])!=n or len(caps['train'])!=16540: raise ValueError('Training image/caption size mismatch')
    if {x['concept'] for x in cat['train']} & {x['concept'] for x in cat['test']}:
        raise ValueError('Train/test concept overlap in source catalog')
    ccounts=Counter(x['concept'] for x in cat['train'])
    if len(ccounts)!=200 or set(ccounts.values())!={10}:raise ValueError('Expected 200 concepts by 10 images')
    dups=Counter(s.strip().casefold() for s in (caps['train'][i] for i in idx))
    if len(dups)<1800: raise ValueError('Unexpected caption collision distribution')
    # Verify the original model-preparation outputs against its recorded SHA256.
    inputs=['sub-01_train_prez.npy']+[f'train_L{l}_mean.npy' for l in LAYERS]
    expected=state['completed']
    for name in inputs:
        p=src/(name+'.bin' if (src/(name+'.bin')).exists() else name)
        if not p.exists():raise FileNotFoundError(str(p))
        if sha(p)!=expected[name]:raise ValueError(f'Historical preparation SHA mismatch for {name}')
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        exp=(n,17,250) if name.startswith('sub-') else (n,3200)
        if a.shape!=exp or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError(f'Array shape/dtype/nonfinite: {name}, actual={a.shape}')
    feat=torch.load(src/'feats7_blip_caps_clip_b32.pt',map_location='cpu',weights_only=True)
    if feat['train'].shape!=(16540,512) or feat['test'].shape!=(200,512):
        raise ValueError('Caption cache expected whole official training partition')
    feats=feat['train'][idx].to(dtype=torch.float32).numpy()
    from collections import defaultdict
    groups=defaultdict(list)
    norm=[caps['train'][i].strip().casefold() for i in idx]
    for j,t in enumerate(norm):groups[t].append(j)
    for positions in groups.values():
        if len(positions)>1 and not np.array_equal(feats[positions],np.repeat(feats[positions[0]][None],len(positions),axis=0)):
            raise ValueError('Same caption mapped to inconsistent text feature')
    if not np.isfinite(feats).all(): raise ValueError('Nonfinite captions')
    dst.mkdir(parents=True)
    manifest={'partition':'official_training',
              'benchmark_use':'historically examined 200-concept development subset, NOT official evaluation',
              'eeg_normalization':'not_split_zscored',
              'preprocessing_provenance':json.dumps(data['preprocessing'],sort_keys=True),
              'text_provenance':'BLIP-large captions generated from training images; open_clip ViT-B-32 OpenAI frozen text; selected via original 16540-row image index mapping; text is NOT independent of image',
              'text_unit':'image_caption', 'visual_layers':list(LAYERS),
              'catalog':{'image_ids':[x['id'] for x in cat['train']],
                         'concept_ids':[x['concept'] for x in cat['train']],
                         'content_hashes':[x['sha256'] for x in cat['train']],
                         'caption_ids':['cap_'+hashlib.sha256(t.encode()).hexdigest() for t in norm]},
              'visuals':[], 'subjects':{},'data_origin':{
               'selected_original_training_indices':idx,
               'original_dataset_json_sha256':sha(src/'dataset.json'),
               'original_preparation_state_sha256':sha(src/'preparation_state.json'),
               'caption_full_cache_sha256':sha(src/'feats7_blip_caps_clip_b32.pt'),
               'caption_strings_sha256':sha(src/'captions_blip_large.json')}}
    def add(name,original=None):
        p=dst/name
        if original is not None:
            shutil.copyfile(original,p)
        return {'path':name,'partition':'official_training','sha256':sha(p)}
    name='sub-01_train_prez.npy';source_path=src/(name+'.bin' if (src/(name+'.bin')).exists() else name)
    manifest['subjects']['sub-01']=add(name,source_path)
    for layer in LAYERS:
        name=f'train_L{layer}_mean.npy';source_path=src/(name+'.bin' if (src/(name+'.bin')).exists() else name)
        manifest['visuals'].append(add(name,source_path))
    name='captions_train_selected.npy';np.save(dst/name,feats);manifest['text']=add(name)
    (dst/'train.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    meta={'source':'selected 200 concepts, no official test data accessed',
          'train_images':n,'concepts':len(ccounts), 'unique_captions':len(groups),
          'duplicate_caption_rows':sum(k for k in dups.values() if k>1),
          'duplicate_caption_groups':sum(1 for k in dups.values() if k>1),
          'duplicate_counts':dict(Counter(dups.values())),
          'manifest_sha256':sha(dst/'train.json')}
    (dst/'preflight_report.json').write_text(json.dumps(meta,indent=2))
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output),indent=2))
