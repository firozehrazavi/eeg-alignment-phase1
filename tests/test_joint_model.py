"""CPU falsification checks for the triadic mechanism and leakage boundary."""
import unittest

import torch

from phase1.core import multi_positive_loss
from phase1.joint_model import JointAligner, Residualizer, symmetric_identity_nce
from phase1.models import Aligner


class JointModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = JointAligner(6, [7, 5], 8, rank=4).eval()
        self.eeg = torch.randn(4, 2, 3)
        self.visuals = [torch.randn(4, 7), torch.randn(4, 5)]
        self.text = torch.randn(4, 8, requires_grad=True)
        self.ids = ['a', 'b', 'c', 'd']

    def encode(self):
        return self.model.encode(self.eeg, self.visuals)

    def test_backbone_initialization_and_image_score_preserve_06b(self):
        torch.manual_seed(7)
        historical = Aligner(6, [7, 5]).eval()
        for key, value in historical.state_dict().items():
            torch.testing.assert_close(value, self.model.backbone.state_dict()[key], rtol=0, atol=0)
        e, v = self.encode()
        expected = historical.scale() * torch.nn.functional.normalize(e, dim=-1) @ torch.nn.functional.normalize(v, dim=-1).T
        torch.testing.assert_close(self.model.image_scores(e, v), expected)

    def test_plain_06b_checkpoint_load_preserves_backbone_outputs(self):
        torch.manual_seed(19)
        historical = Aligner(6, [7, 5]).eval()
        transferred = JointAligner(6, [7, 5], 8, rank=4).eval()
        transferred.load_06b_backbone_state(historical.state_dict())
        expected_eeg = historical.eeg(self.eeg)
        expected_image = historical.img(self.visuals)
        actual_eeg, actual_image = transferred.backbone.eeg(self.eeg), transferred.backbone.img(self.visuals)
        torch.testing.assert_close(actual_eeg, expected_eeg, rtol=0, atol=0)
        torch.testing.assert_close(actual_image, expected_image, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'plain, compatible'):
            transferred.load_06b_backbone_state({'backbone.' + k: v for k, v in historical.state_dict().items()})

    def test_joint_objective_reaches_all_three_projections_and_backbone(self):
        e, v = self.encode()
        symmetric_identity_nce(self.model.joint_scores(e, v, self.text), self.ids).backward()
        for module in [self.model.backbone.eeg, self.model.backbone.img,
                       self.model.joint_eeg, self.model.joint_image, self.model.joint_text]:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.norm().item(), 0, name)
        self.assertIsNone(self.text.grad)

    def test_joint_has_nonzero_three_way_mixed_difference(self):
        # Any sum of independent pairwise energies has zero mixed third finite
        # difference. This check isolates the triadic score from the base score.
        e, v = [x.detach() for x in self.encode()]
        total = torch.zeros(4, 4)
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    total += (-1)**(i+j+k) * self.model.joint_scores(e*i, v*j, self.text*k)
        self.assertGreater(total.abs().max().item(), 1e-5)

    def test_shuffled_text_changes_joint_but_not_image_inference(self):
        e, v = self.encode()
        a = self.model.joint_scores(e, v, self.text)
        b = self.model.joint_scores(e, v, self.text[[1, 2, 3, 0]])
        self.assertFalse(torch.allclose(a, b))
        image = self.model.image_scores(e, v)
        self.assertEqual(image.shape, (4, 4))
        torch.testing.assert_close(image, self.model.image_scores(e, v))

    def test_zero_joint_weight_backbone_gradient_equals_baseline_even_with_student(self):
        e, v = self.encode()
        baseline = multi_positive_loss(e, v, self.model.backbone.scale(), self.ids, self.ids)
        joint = self.model.loss(e, v, self.text, self.text, self.ids,
                                joint_weight=0, distill_weight=.5)
        parameters = list(self.model.backbone.parameters())
        expected = torch.autograd.grad(baseline, parameters, retain_graph=True)
        actual = torch.autograd.grad(joint, parameters)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_distillation_only_trains_readout_and_no_raw_text(self):
        e, v = self.encode()
        self.model.loss_terms(e, v, self.text, self.text, self.ids)['distill'].backward()
        for parameter in self.model.backbone.parameters():
            self.assertIsNone(parameter.grad)
        for module in [self.model.joint_eeg, self.model.joint_image, self.model.joint_text]:
            for parameter in module.parameters():
                self.assertIsNone(parameter.grad)
        for module in [self.model.readout_eeg, self.model.readout_text]:
            for parameter in module.parameters():
                self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertIsNone(self.text.grad)

    def test_direct_text_inference_accepts_eeg_and_text_only(self):
        e = self.model.backbone.eeg(self.eeg)
        scores = self.model.text_scores(e, torch.randn(9, 8))
        self.assertEqual(scores.shape, (4, 9))
        self.assertTrue(torch.isfinite(scores).all())

    def test_identity_loss_handles_duplicate_and_rectangular_matches(self):
        logits = torch.tensor([[20., 0.], [20., 0.], [0., 20.]], requires_grad=True)
        loss = symmetric_identity_nce(logits, ['a', 'a', 'b'], ['a', 'b'])
        self.assertLess(loss.item(), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        all_positive = symmetric_identity_nce(torch.randn(4, 3), ['a']*4, ['a']*3)
        self.assertEqual(all_positive.item(), 0)
        with self.assertRaises(ValueError):
            symmetric_identity_nce(torch.eye(2), ['a', 'b'], ['a', 'c'])

    def test_invalid_shapes_and_nonfinite_values_fail(self):
        e, v = self.encode()
        for bad in [torch.empty(0, 2, 3), torch.ones(4, 3, 3), self.eeg * float('nan')]:
            with self.assertRaises(ValueError):
                self.model.encode(bad, self.visuals)
        with self.assertRaises(ValueError):
            self.model.encode(self.eeg, self.visuals[:1])
        with self.assertRaises(ValueError):
            self.model.joint_scores(e, v, self.text[:3])
        with self.assertRaises(ValueError):
            self.model.text_scores(e, self.text * float('inf'))
        with self.assertRaises(ValueError):
            self.model.loss(e, v, self.text, self.text, self.ids, joint_weight=-1)
        with self.assertRaisesRegex(ValueError, 'rows are not aligned'):
            self.model.encode(self.eeg[:3], self.visuals)


class ResidualizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.visuals = [torch.randn(12, 5), torch.randn(12, 3)]
        self.text = torch.randn(12, 6)
        self.fit_rows = list(range(8))

    def fit(self, visuals=None, text=None):
        return Residualizer.fit(self.visuals if visuals is None else visuals,
                               self.text if text is None else text,
                               self.fit_rows, rank=4, ridge=.1, seed=14)

    def test_fitted_state_is_independent_of_heldout_values(self):
        a = self.fit()
        visuals = [x.clone() for x in self.visuals]
        text = self.text.clone()
        for x in visuals:
            x[8:] = float('nan')
        text[8:] = float('nan')
        b = self.fit(visuals, text)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_transform_is_pointwise_and_fitted_state_roundtrips(self):
        a = self.fit()
        result = a.transform(self.visuals, self.text)
        one = a.transform([x[8:9] for x in self.visuals], self.text[8:9])
        torch.testing.assert_close(one, result[8:9])
        b = Residualizer.from_state_dict(a.state_dict())
        torch.testing.assert_close(b.transform(self.visuals, self.text), result, rtol=0, atol=0)
        self.assertEqual(len(list(a.parameters())), 0)

    def test_fitting_indices_and_invalid_inputs_fail(self):
        for rows in [[], [0], [0, 0], [-1, 0], [0, 12], [0., 1.]]:
            with self.assertRaises(ValueError):
                Residualizer.fit(self.visuals, self.text, rows)
        with self.assertRaises(ValueError):
            Residualizer.fit(self.visuals, self.text, self.fit_rows, ridge=0)
        with self.assertRaises(ValueError):
            self.fit().transform(self.visuals[:1], self.text)
        with self.assertRaises(ValueError):
            self.fit().transform(self.visuals, self.text[:8])

    def test_linear_visual_information_is_removed_on_simple_problem(self):
        x = torch.randn(80, 3)
        text = x @ torch.randn(3, 5) + 2
        fitted = Residualizer.fit([x], text, list(range(60)), rank=3, ridge=1e-7)
        residual = fitted.transform([x], text)
        self.assertLess(residual[60:].square().mean().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
