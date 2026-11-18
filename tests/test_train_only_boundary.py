"""Executable data-access regression tests; all arrays are synthetic."""
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from phase1.patch_cache import TokenCache, digest
from phase1.patch_experiment import run


def fixture(root):
    cache = root / 'tokens'
    cache.mkdir()
    catalog = {
        split: [dict(id=f'{split}-{i}', concept=f'{split}-c{i // 2}',
                     sha256=f'{split}-hash-{i}', path='unused') for i in range(n)]
        for split, n in [('train', 12), ('test', 200)]
    }
    # Test concepts must be unique across splits, not necessarily within split.
    cp = root / 'catalog.json'
    cp.write_text(json.dumps(catalog))
    rng = np.random.default_rng(8)
    tokens = rng.normal(size=(12, 4, 16)).astype('float16')
    np.save(cache / 'train.npy', tokens)
    np.save(cache / 'train_done.npy', np.ones(12, dtype=bool))
    (cache / 'manifest.json').write_text(json.dumps(dict(
        catalog_sha256=digest(cp), layer=28, grid=[2, 2], width=16,
        shapes={'train': [12, 4, 16], 'test': [200, 4, 16]})))
    manifest = dict(
        catalog='catalog.json', tokens='tokens',
        preprocessing={'regime': 'benchmark_preprocessed'},
        pooled={split: [f'{split}_L{layer}.npy' for layer in [20,24,28,32,36]]
                for split in ['train', 'test']},
        subjects={s: {split: f'{s}_{split}.npy' for split in ['train', 'test']}
                  for s in ['sub-01', 'sub-02']})
    for p in manifest['pooled']['train']:
        np.save(root / p, tokens.astype('float32').mean(1))
    np.save(root / 'sub-01_train.npy', rng.normal(size=(12, 3, 4)).astype('float32'))
    # Official-test and outer-held-subject arrays deliberately do not exist.
    (root / 'dataset.json').write_text(json.dumps(manifest))
    config = dict(setting='loso', targets=['sub-02'], seeds=[0],
                  regime='benchmark_preprocessed', objective='multi_positive',
                  arms=['baseline', 'patch'], epochs=1, batch=6, lr=1e-4,
                  weight_decay=1e-4, patch_layer=28, split_seed=0,
                  val_fraction=.33, device='cpu', bottleneck=16, groups=8,
                  adapter_chunk=2)
    (root / 'config.json').write_text(json.dumps(config))
    forbidden = {root / p for p in manifest['pooled']['test']}
    forbidden |= {root / manifest['subjects'][s]['test'] for s in manifest['subjects']}
    forbidden |= {root / 'sub-02_train.npy', cache / 'test.npy', cache / 'test_done.npy'}
    return {p.resolve() for p in forbidden}


@contextlib.contextmanager
def deny_files(forbidden):
    """Reject actual opens and np.load attempts, including hashing/mmap reads."""
    seen = []

    def guard(fn):
        def wrapped(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                p = Path(os.fsdecode(path)).resolve()
                seen.append(p)
                if p in forbidden:
                    raise AssertionError(f'Protected artifact access: {p.name}')
            return fn(path, *args, **kwargs)
        return wrapped

    with mock.patch('builtins.open', side_effect=guard(builtins.open)), \
         mock.patch('io.open', side_effect=guard(io.open)), \
         mock.patch('numpy.load', side_effect=guard(np.load)):
        yield seen


class TrainOnlyBoundaryTests(unittest.TestCase):
    def test_training_and_profile_never_open_or_hash_test_artifacts(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forbidden = fixture(root)
            with deny_files(forbidden) as seen:
                for mode in ['train_only', 'profile_one_epoch']:
                    with self.subTest(mode=mode):
                        output = root / mode
                        self.assertEqual(run(root/'dataset.json', root/'config.json',
                                             output, **{mode: True}), [])
                        self.assertEqual(run(root/'dataset.json', root/'config.json',
                                             output, resume=True, **{mode: True}), [])
                        provenance = json.loads((output/'provenance.json').read_text())
                        self.assertEqual(provenance['data_access_splits'], ['train'])
                        self.assertFalse({(root/p).resolve() for p in provenance['input_hashes']} & forbidden)
                        self.assertEqual(len(list(output.glob('*_latest.pt'))), 2)
                        self.assertFalse((output/'results.csv').exists())
            self.assertIn((root/'tokens/train.npy').resolve(), seen)
            self.assertIn((root/'sub-01_train.npy').resolve(), seen)
            self.assertFalse(set(seen) & forbidden)

    def test_scoped_token_reader_refuses_unloaded_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forbidden = fixture(root)
            with deny_files(forbidden):
                cache = TokenCache(root/'tokens', root/'catalog.json', splits=('train',))
                self.assertEqual(cache.get('train', [0]).shape, (1,4,16))
                with self.assertRaisesRegex(ValueError, 'not authorized'):
                    cache.get('test', [0])
                del cache  # Release the read-only mmap before Windows cleanup.

    def test_resume_rejects_test_hash_before_opening_it(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forbidden = fixture(root)
            output = root/'run'
            with deny_files(forbidden):
                run(root/'dataset.json', root/'config.json', output, train_only=True)
                p = output/'provenance.json'
                previous = json.loads(p.read_text())
                previous['input_hashes']['test_L20.npy'] = 'must-not-be-read'
                p.write_text(json.dumps(previous))
                with self.assertRaisesRegex(ValueError, 'refuses prior test'):
                    run(root/'dataset.json', root/'config.json', output,
                        resume=True, train_only=True)


if __name__ == '__main__':
    unittest.main()
