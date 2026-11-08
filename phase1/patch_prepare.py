"""Prepare NEW row-aligned pilot/full arrays; never reads legacy _z16 EEG.

The distributed preprocessed EEG may contain session-dependent whitening.
This converter therefore declares benchmark_preprocessed, NOT strict induction.
Only use pickle inputs downloaded from a trusted THINGS-EEG2 source.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .patch_cache import digest, read_catalog

POST17 = [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]
LAYERS = [20,24,28,32,36]


def prepare(root, cache, output, subjects, concepts=200, seed=0):
    root, cache, output = Path(root), Path(cache), Path(output)
    required = [root/'images_set/image_metadata.npy',cache/'meta_v6.pt',cache/'feat6_internvit6b.pt']
    required += [root/f'Preprocessed_data_250Hz/{s}/preprocessed_eeg_{split}.npy'
                 for s in subjects for split in ('training','test')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing real-data inputs:\n'+'\n'.join(missing))
    if output.exists():
        raise FileExistsError(f'Choose a NEW prepared directory: {output}')
    meta = np.load(required[0],allow_pickle=True).item()
    old = torch.load(required[1],map_location='cpu',weights_only=False)
    features = torch.load(required[2],map_location='cpu',weights_only=False)
    for s in ('train','test'):
        if list(meta[f'{s}_img_files']) != list(old[f'{s}_files']):
            raise ValueError('Feature metadata and raw image row order disagree')
    unique = sorted(set(map(str,meta['train_img_concepts'])))
    rng = np.random.default_rng(seed)
    chosen = set(rng.permutation(unique)[:concepts]) if concepts else set(unique)
    selection = {'train':[i for i,c in enumerate(meta['train_img_concepts']) if str(c) in chosen],
                 'test':list(range(len(meta['test_img_files'])))}
    catalog = {}
    for s in ('train','test'):
        catalog[s] = []
        for i in selection[s]:
            path = root/old[f'{s}_rel'][i]
            catalog[s].append(dict(id=str(meta[f'{s}_img_files'][i]),
                concept=str(meta[f'{s}_img_concepts'][i]),path=str(path.resolve()),sha256=digest(path)))
    output.mkdir(parents=True)
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2))
    read_catalog(output/'catalog.json')
    manifest = dict(catalog='catalog.json',tokens='tokens_L28',subjects={},pooled={},
        preprocessing={'regime':'benchmark_preprocessed','upstream_target_independent':False,
            'description':'Distributed EEG, repetitions averaged, posterior17, no additional z-score. Upstream whitening not audited.',
            'channels':POST17,'time_slice':'all supplied samples'},
        selection=selection,source_hashes={str(p):digest(p) for p in required[:3]})
    for s in ('train','test'):
        manifest['pooled'][s] = []
        for layer in LAYERS:
            name = f'{s}_L{layer}_mean.npy'
            x = features['tr' if s=='train' else 'te'][(layer,'mean')]
            np.save(output/name,np.asarray(x[selection[s]].float(),dtype='float32'))
            manifest['pooled'][s].append(name)
    for subject in subjects:
        manifest['subjects'][subject] = {}
        for s,rawsplit in [('train','training'),('test','test')]:
            source = root/f'Preprocessed_data_250Hz/{subject}/preprocessed_eeg_{rawsplit}.npy'
            obj = np.load(source,allow_pickle=True).item()
            x = obj['preprocessed_eeg_data']
            if x.ndim != 4 or x.shape[0] != len(meta[f'{s}_img_files']) or x.shape[2] != 63:
                raise ValueError(f'Expected image x repetitions x 63 x time: {source}, got {x.shape}')
            name = f'{subject}_{s}_prez.npy'
            # Bound temporary RAM: average image chunks instead of all repetitions.
            arr = np.lib.format.open_memmap(output/name,mode='w+',dtype='float32',
                shape=(len(selection[s]),17,x.shape[-1]))
            for start in range(0,len(arr),64):
                rows = selection[s][start:start+64]
                arr[start:start+len(rows)] = x[rows].mean(1,dtype=np.float32)[:,POST17]
            arr.flush()
            manifest['subjects'][subject][s] = name
            manifest['source_hashes'][str(source)] = digest(source)
            del x,obj,arr
    (output/'dataset.json').write_text(json.dumps(manifest,indent=2))
    print(output/'dataset.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/content/EEG_Data')
    p.add_argument('--cache',default='/content/drive/MyDrive/eeg_results/cache')
    p.add_argument('--output',required=True)
    p.add_argument('--subjects',nargs='+',default=['sub-01'])
    p.add_argument('--concepts',type=int,default=200,help='0 = all training concepts')
    p.add_argument('--seed',type=int,default=0)
    a = p.parse_args(); prepare(a.root,a.cache,a.output,a.subjects,a.concepts,a.seed)
