"""Focused synthetic CPU tests for the read-only 06b image-caption gate."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics/image_caption_bridge_gate.py"
spec = importlib.util.spec_from_file_location("image_caption_bridge_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ImageCaptionGateTests(unittest.TestCase):
    def test_real_pairing_beats_shuffled_on_unseen_images(self):
        torch.manual_seed(78)
        x = torch.randn(180, 16)
        y = x @ torch.randn(16, 24) + 0.02 * torch.randn(180, 24)
        fit = torch.arange(150)
        query = torch.arange(150, 180)
        real = gate.fit_bridge(x, y, fit)
        null = gate.fit_bridge(
            x, y, fit, torch.as_tensor(np.random.default_rng(7).permutation(len(fit)))
        )
        ids = [str(i) for i in query.tolist()]
        a, _ = gate.caption_hits(
            gate.bridge_scores(x, y, real, query, query), ids, ids, query.tolist()
        )
        b, _ = gate.caption_hits(
            gate.bridge_scores(x, y, null, query, query), ids, ids, query.tolist()
        )
        self.assertGreater(float(a.mean()), 0.8)
        self.assertGreater(float((a-b).mean()), 0.5)

    def test_duplicate_caption_is_a_valid_positive(self):
        scores = torch.tensor([[0.1, 0.99, 0.0], [0.8, 0.0, 0.1]])
        top1, top5 = gate.caption_hits(
            scores, ["cat", "dog"], ["cat", "cat", "dog"], [1, 2, 3]
        )
        self.assertEqual(top1.tolist(), [1.0, 0.0])
        self.assertTrue(torch.all(top5 == 1))

    def test_fit_cannot_duplicate_rows_or_inject_validation(self):
        x, y = torch.randn(9, 4), torch.randn(9, 5)
        with self.assertRaisesRegex(ValueError, "unique"):
            gate.fit_bridge(x, y, torch.tensor([1, 1, 2]))
        with self.assertRaisesRegex(ValueError, "control|Shuffled"):
            gate.fit_bridge(x, y, torch.arange(4), torch.tensor([0, 1, 2, 2]))

    def test_missing_positives_and_duplicate_gallery_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            gate.caption_hits(torch.ones(1, 2), ["missing"], ["a", "b"], [0, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gate.caption_hits(torch.ones(1, 2), ["a"], ["a", "a"], [0, 0])

    def test_bootstrap_reproducibility(self):
        a = torch.tensor([1., 1., 1., 0., 1.])
        b = torch.tensor([0., 0., 0., 0., 1.])
        self.assertEqual(gate.bootstrap_ci(a, b, seed=3),
                         gate.bootstrap_ci(a, b, seed=3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
