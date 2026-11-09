"""Fast CPU header/config checks; no array scan, GPU work or writes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .patch_cache import read_catalog, digest
from .patch_experiment import enforce_provenance, schedule
from .core import concept_split


def preflight(dataset, config):
    root=Path(dataset).resolve().parent
    m=json.loads(Path(dataset).read_text());c=json.loads(Path(config).read_text())
    state=json.loads((root/'preparation_state.json').read_text())
    enforce_provenance(m,c)
    if c['setting']!='within_subject' or sorted(m['subjects'])!=['sub-01']:
        raise ValueError('Pilot requires within-subject sub-01 only')
    if c['seeds']!=[0] or c['arms']!=['baseline','mean','patch']:
        raise ValueError('Unexpected pilot schedule')
    if c['epochs']!=5 or c['batch']!=64 or c['patch_layer']!=28:
        raise ValueError('Unexpected pilot training configuration')
    if not 0<c['val_fraction']<1:raise ValueError('Invalid validation fraction')
    cat=read_catalog(root/m['catalog'])
    for name in ['dataset.json',m['catalog']]:
        if state['completed'].get(name)!=digest(root/name):raise ValueError('Manifest/catalog hash mismatch')
    shapes={}
    for split in ('train','test'):
        names=m['pooled'][split]
        if len(names)!=5:raise ValueError('Expected five pooled layers')
        pairs=[(name,(len(cat[split]),3200)) for name in names]
        pairs.append((m['subjects']['sub-01'][split],(len(cat[split]),17,250)))
        for name,expected in pairs:
            path=root/name
            if not path.resolve().is_relative_to(root):raise ValueError('Unsafe manifest path')
            if name not in state['completed']:raise ValueError('Missing completion record')
            x=np.load(path,mmap_mode='r',allow_pickle=False)
            if x.shape!=expected or x.dtype!=np.float32:raise ValueError(f'Wrong shape/dtype: {name}')
            shapes[name]=list(x.shape);del x
    concepts=[r['concept'] for r in cat['train']]
    train,val=concept_split(concepts,seed=c['split_seed'],fraction=c['val_fraction'])
    if len(train)<2 or len(val)<2:raise ValueError('Insufficient training/validation rows')
    return dict(status='header_preflight_passed',shapes=shapes,schedule=schedule(c,['sub-01']),
                train_rows=len(train),validation_rows=len(val),
                limitations='Array contents/hashes not rescanned; patch tokens and GPU fit require separate checks')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();print(json.dumps(preflight(a.dataset,a.config),indent=2))
