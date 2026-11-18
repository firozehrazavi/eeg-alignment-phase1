import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase1.joint_inference import run
from phase1.joint_model import JointAligner, Residualizer


class JointInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dims = dict(n_in=6, dims=[3]*5, text_dim=4, rank=3)
        self.model = JointAligner(**self.dims).eval()
        self.saved = dict(model=self.model.state_dict(), model_dimensions=self.dims,
                          eeg_mean=torch.ones(1, 2, 1), eeg_std=torch.ones(1, 2, 1)*2,
                          text_mean=torch.ones(1, 4), text_std=torch.ones(1, 4)*3,
                          visual_mean=[torch.ones(1, 3)]*5,
                          visual_std=[torch.ones(1, 3)*4]*5)
        self.eeg = np.random.default_rng(0).normal(size=(3, 2, 3)).astype('float32')
        self.text = np.random.default_rng(1).normal(size=(5, 4)).astype('float32')
        np.save(self.root / 'queries.npy', self.eeg)
        np.save(self.root / 'captions.npy', self.text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_inference_never_accesses_image_inputs_or_residualizer(self):
        # The text-only checkpoint intentionally has NO image normalizers.
        self.saved.pop('visual_mean'); self.saved.pop('visual_std')
        torch.save(self.saved, self.root / 'model.pt')
        original_load = np.load
        touched = []
        def guard(path, *args, **kwargs):
            touched.append(Path(path).name)
            self.assertIn(Path(path).name, ['queries.npy', 'captions.npy'])
            self.assertFalse(kwargs.get('allow_pickle', True))
            return original_load(path, *args, **kwargs)
        with patch('phase1.joint_inference.np.load', side_effect=guard), \
             patch.object(Residualizer, 'transform', side_effect=AssertionError('Image residualizer used')), \
             patch.object(JointAligner, 'image_scores', side_effect=AssertionError('Image score used')):
            info = run(self.root / 'model.pt', self.root / 'queries.npy',
                       self.root / 'captions.npy', 'text', self.root / 'scores.npz', batch=2)
        self.assertEqual(touched, ['queries.npy', 'captions.npy'])
        self.assertEqual(info['gallery_count'], 5)
        x = (torch.from_numpy(self.eeg)-1)/2
        t = (torch.from_numpy(self.text)-1)/3
        with torch.no_grad():
            expected = self.model.text_scores(self.model.backbone.eeg(x), t).numpy()
        with np.load(self.root / 'scores.npz', allow_pickle=False) as result:
            np.testing.assert_allclose(result['scores'], expected, rtol=1e-5, atol=1e-5)
            self.assertEqual(result['query_count'].item(), 3)
            self.assertEqual(result['modality'].item(), 'text')

    def test_image_inference_and_refusal_to_overwrite(self):
        torch.save(self.saved, self.root / 'model.pt')
        visual = {f'layer{i}': np.random.default_rng(i).normal(size=(5, 3)).astype('float32')
                  for i in [20, 24, 28, 32, 36]}
        np.savez(self.root / 'images.npz', **visual)
        args = (self.root / 'model.pt', self.root / 'queries.npy',
                self.root / 'images.npz', 'image', self.root / 'scores.npz')
        info = run(*args, batch=2)
        self.assertEqual(info['gallery_count'], 5)
        with torch.no_grad():
            ze = self.model.backbone.eeg((torch.from_numpy(self.eeg)-1)/2)
            zi = self.model.backbone.img([(torch.from_numpy(v)-1)/4 for v in visual.values()])
            expected = self.model.image_scores(ze, zi).numpy()
        with np.load(self.root / 'scores.npz', allow_pickle=False) as result:
            np.testing.assert_allclose(result['scores'], expected, rtol=1e-5, atol=1e-5)
        with self.assertRaises(FileExistsError):
            run(*args)

    def test_invalid_gallery_rejected_before_output(self):
        torch.save(self.saved, self.root / 'model.pt')
        np.savez(self.root / 'images.npz', layer20=np.zeros((5, 3), dtype='float32'))
        with self.assertRaises(ValueError):
            run(self.root / 'model.pt', self.root / 'queries.npy', self.root / 'images.npz',
                'image', self.root / 'scores.npz')
        self.assertFalse((self.root / 'scores.npz').exists())


if __name__ == '__main__':
    unittest.main()
