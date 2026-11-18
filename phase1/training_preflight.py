"""CPU-only completed-token/profile compatibility checks. No model downloads."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .pilot_preflight import preflight
from .patch_cache import TokenCache, digest


def check(dataset,config,output,staging_parent,*,include_test=False):
    """Training-only by default; final-evaluation staging is explicitly opt-in."""
    result=preflight(dataset,config,train_only=not include_test)
    root=Path(dataset).resolve().parent;m=json.loads(Path(dataset).read_text())
    cfg=json.loads(Path(config).read_text());out=Path(output)
    splits=('train','test') if include_test else ('train',)
    tokens=TokenCache(root/m['tokens'],root/m['catalog'],splits=splits)
    if tokens.meta['layer']!=28 or tokens.meta['width']!=3200:raise ValueError('Wrong token layer/width')
    expected={'train':(2000,1024,3200),'test':(200,1024,3200)}
    if any(tokens.arrays[s].shape!=expected[s] for s in splits):
        raise ValueError('Expected completed 2000-image training/200-image test L28 pilot')
    files=['dataset.json','preparation_state.json',m['catalog']]
    files += [p for s in splits for p in m['pooled'][s]]
    files += [subject[s] for subject in m['subjects'].values() for s in splits]
    token_files=['manifest.json']+[p for s in splits for p in (f'{s}.npy',f'{s}_done.npy')]
    files += [str(Path(m['tokens'])/p) for p in token_files]
    needed=sum((root/p).stat().st_size for p in files)
    if shutil.disk_usage(staging_parent).free<needed*1.1:raise OSError('Insufficient staging space (need 10% reserve)')
    if out.exists():
        prev=json.loads((out/'provenance.json').read_text())
        if not include_test and (prev.get('data_access_splits')!=['train']
                                 or not set(prev['input_hashes'])<=set(files)):
            raise ValueError('Training-only preflight refuses prior test or unscoped input access')
        if (prev['config']!=cfg or prev['dataset_sha256']!=digest(dataset)
            or prev['catalog_sha256']!=digest(root/m['catalog']) or prev['token_manifest']!=tokens.meta):
            raise ValueError('Checkpoint provenance mismatch; use a new output directory')
        code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        if prev['code_hashes']!=code:raise ValueError('Checkpoint code mismatch')
        for name,h in prev['input_hashes'].items():
            if digest(root/name)!=h:raise ValueError('Checkpoint input changed')
        for path in out.glob('*_latest.pt'):
            ck=torch.load(path,map_location='cpu',weights_only=False)
            for key in ['epoch','model','optimizer','scheduler','scaler','numpy_rng','torch_rng','best_state']:
                if key not in ck:raise ValueError(f'Incomplete checkpoint: {path}')
    return dict(**result,staging_bytes=needed,files=files,checkpoint_status='compatible' if out.exists() else 'new')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--staging-parent',default='/content')
    p.add_argument('--include-test',action='store_true',help='Explicitly check and stage official-test artifacts')
    a=p.parse_args();print(json.dumps(check(a.dataset,a.config,a.output,a.staging_parent,include_test=a.include_test),indent=2))
