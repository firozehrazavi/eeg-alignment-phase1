"""Focused synthetic CPU tests of frozen 06b EEG->Caption scoring gate."""
import unittest
import torch
from diagnostics.eeg_bridge_transfer_gate import (
    bridge_scores_from_embeddings, check, compare, direct_eeg_ridge_diagnostic)
from diagnostics.image_caption_bridge_gate import bridge_scores


class EEGBridgeGateTests(unittest.TestCase):
    def test_bridge_scoring_equals_validated_image_bridge_scoring(self):
        torch.manual_seed(14)
        images = torch.randn(9, 1024)
        texts = torch.randn(9, 512)
        fit = (torch.randn(1024, 512) * 0.01,
               torch.randn(1, 1024) * 0.1,
               torch.randn(1, 512) * 0.1)
        a = bridge_scores_from_embeddings(images, texts, fit)
        b = bridge_scores(images, texts, fit, torch.arange(9), torch.arange(9))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_caption_duplicate_identity_scores(self):
        text = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        q = text.clone()
        fit = (torch.eye(2), torch.zeros(1,2), torch.zeros(1,2))
        result, hits, _ = check(q, text, fit,
                                ['cat','cat','dog'], [10,11,12])
        self.assertEqual(result['hits_top1'], 3)
        self.assertEqual(hits.tolist(), [1.,1.,1.])

    def test_direct_readout_rejects_partial_or_invalid_shape(self):
        eeg = torch.randn(10, 1024)
        cap = torch.randn(10, 512)
        with self.assertRaisesRegex(ValueError, '16,540'):
            direct_eeg_ridge_diagnostic(
                eeg, cap, ['x']*10,
                torch.arange(9), torch.tensor([9]), (eeg[-1:], None))

    def test_shapes_and_invalid_oracle_fail_closed(self):
        z = torch.randn(11, 1024)
        txt = torch.randn(11, 512)
        fit = (torch.zeros(1024,512), torch.zeros(1,1024), torch.zeros(1,512))
        with self.assertRaisesRegex(ValueError, 'both'):
            compare(z[:3], z[:2], txt, [], [], fit, fit)
        with self.assertRaisesRegex(ValueError, '1,654'):
            compare(z, z, txt, [], [], fit, fit)


if __name__=="__main__":
    unittest.main(verbosity=2)
