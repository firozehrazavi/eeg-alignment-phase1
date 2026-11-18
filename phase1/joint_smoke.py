"""Synthetic engineering fixture; never biological evidence or SOTA results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .joint_data import digest
from .joint_experiment import run


def fixture(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(82)
    n = 40; latent = rng.normal(size=(n, 4))
    def save(name, value):
        p = root / name; np.save(p, value.astype('float32'))
        return dict(path=name, sha256=digest(p), partition='official_training')
    manifest = dict(partition='official_training', eeg_normalization='not_split_zscored',
        preprocessing_provenance='SYNTHETIC FIXTURE: random arrays; not THINGS-EEG2',
        text_provenance='SYNTHETIC latent linear projection', text_unit='image_caption',
        visual_layers=[20,24,28,32,36],
        catalog=dict(image_ids=[f'i{i}' for i in range(n)], concept_ids=[f'c{i//2}' for i in range(n)],
            caption_ids=[f't{i}' for i in range(n)], content_hashes=[hashlib.sha256(f'synthetic{i}'.encode()).hexdigest() for i in range(n)]))
    manifest['visuals'] = [save(f'layer{l}.npy', latent @ rng.normal(size=(4, 6))) for l in manifest['visual_layers']]
    manifest['text'] = save('captions.npy', latent @ rng.normal(size=(4, 7)))
    manifest['subjects'] = {s: save(f'{s}.npy', (latent @ rng.normal(size=(4, 8))).reshape(n,2,4)) for s in ['sub-01','sub-02']}
    # Deliberately nonexistent protected artifact: loader must never touch this.
    manifest['official_test'] = {'path': 'PROTECTED_OFFICIAL_TEST_DO_NOT_OPEN.npy'}
    (root/'train.json').write_text(json.dumps(manifest, indent=2))
    config=dict(setting='within_subject', subjects=['sub-01'], seeds=[0], arms=['A','B','C','D'],
        epochs=2, batch=16, lr=.001, weight_decay=.0001, device='cpu', max_seconds=120,
        split_seed=20261117, rank=4, ridge=1., joint_weight=.1, distill_weight=.1, readout_weight=.1)
    (root/'config.json').write_text(json.dumps(config, indent=2))
    return root/'train.json', root/'config.json'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    manifest, config=fixture(root/'synthetic_inputs')
    rows=run(manifest,config,root/'synthetic_runs')
    (root/'NOTICE.md').write_text('Synthetic CPU integration run only. These scores are not EEG research findings.\n')
    print(json.dumps({'status':'synthetic_only','completed_arms': [r['arm'] for r in rows]}))
