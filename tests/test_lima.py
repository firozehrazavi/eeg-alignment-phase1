"""Synthetic CPU correctness evidence, never a real-data feasibility claim."""
import inspect
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from phase1.lima import (LimaAligner, marginal_log_probs, caption_mass_nll,
                         duplicate_caption_kl)
from phase1.models import Aligner
from phase1.joint_model import symmetric_identity_nce


class MarginalMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_exact_normalized_marginal_and_block_size_invariance(self):
        image = torch.randn(3, 7, requires_grad=True)
        conditional = torch.randn(7, 5, requires_grad=True)
        actual = marginal_log_probs(image, conditional, mediator_block_size=2)
        expected = image.softmax(1) @ conditional.detach().softmax(1)
        torch.testing.assert_close(actual.exp(), expected)
        torch.testing.assert_close(actual.exp().sum(1), torch.ones(3))
        torch.testing.assert_close(actual, marginal_log_probs(image, conditional, mediator_block_size=99))
        grad = torch.autograd.grad(actual[:, 0].sum(), (image, conditional), allow_unused=True)
        self.assertGreater(grad[0].norm().item(), 0)
        self.assertIsNone(grad[1])

    def test_extreme_scores_remain_finite_without_underflow_clamp(self):
        image = torch.tensor([[10000., -10000.]], requires_grad=True)
        conditional = torch.tensor([[10000., -10000.], [-10000., 10000.]])
        actual = marginal_log_probs(image, conditional, mediator_block_size=1)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(actual[0, 1].item(), -1000)
        (-actual[0, 1]).backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_marginal_gradients_match_dense_log_formula_in_both_paths(self):
        for extreme in (False, True):
            image = torch.randn(2, 4, requires_grad=True)
            conditional = torch.randn(4, 3)
            if extreme:
                conditional[:, 0] -= 10000  # Forces log-space fallback.
            mask = torch.tensor([[False, False, True, True], [True, False, True, False]])
            actual = marginal_log_probs(image, conditional, allowed_mediators=mask, mediator_block_size=1)
            log_pi = image.masked_fill(~mask, -torch.inf).log_softmax(1)
            reference = (log_pi[:, :, None] + conditional.log_softmax(1)[None]).logsumexp(1)
            torch.testing.assert_close(actual, reference)
            a = torch.autograd.grad(actual[:, 0].sum(), image, retain_graph=True)[0]
            b = torch.autograd.grad(reference[:, 0].sum(), image)[0]
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.isfinite(a).all())

    def test_excluded_blocks_have_zero_mass_and_finite_gradients(self):
        image = torch.randn(2, 4, requires_grad=True)
        conditional = torch.randn(4, 3)
        mask = torch.tensor([[False, False, True, True], [True, False, True, False]])
        actual = marginal_log_probs(image, conditional, allowed_mediators=mask, mediator_block_size=1)
        expected = image.masked_fill(~mask, -torch.inf).softmax(1) @ conditional.softmax(1)
        torch.testing.assert_close(actual.exp(), expected)
        actual[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertTrue((image.grad[~mask] == 0).all())

    def test_caption_loss_accepts_all_duplicate_positives(self):
        probabilities = torch.tensor([[.2, .3, .5], [.1, .2, .7]])
        loss = caption_mass_nll(probabilities.log(), ['same', 'other'], ['same', 'same', 'other'])
        expected = -(torch.tensor(.5).log() + torch.tensor(.7).log()) / 2
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(caption_mass_nll(probabilities.log(), ['same', 'same'], ['same']*3), torch.tensor(0.))
        with self.assertRaisesRegex(ValueError, 'needs a positive'):
            caption_mass_nll(probabilities.log(), ['missing', 'other'], ['same', 'same', 'other'])

    def test_duplicate_kl_does_not_demand_distinguishing_identical_captions(self):
        # Equal identity mass, unequal row mass: the caption student need not
        # reproduce the teacher's arbitrary split between identical strings.
        student = torch.tensor([[.25, .25, .5]]).log().requires_grad_()
        target = torch.tensor([[.1, .4, .5]]).log().requires_grad_()
        loss = duplicate_caption_kl(student, target, ['same', 'same', 'other'])
        self.assertLess(abs(loss.item()), 1e-7)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_invalid_mediator_support_or_shapes_fail(self):
        for mask in (torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                marginal_log_probs(torch.ones(2, 3), torch.ones(3, 4), allowed_mediators=mask)
        with self.assertRaises(ValueError):
            marginal_log_probs(torch.ones(2, 3), torch.ones(2, 4))
        with self.assertRaises(ValueError):
            marginal_log_probs(torch.ones(2, 3), torch.ones(3, 4), mediator_block_size=0)
        with self.assertRaises(ValueError):
            marginal_log_probs(torch.full((2, 3), float('nan')), torch.ones(3, 4))


class LimaModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = LimaAligner(6, [5, 7], 4, rank=3).eval()
        self.eeg = torch.randn(3, 2, 3)
        self.visuals = [torch.randn(3, 5), torch.randn(3, 7)]
        self.text = torch.randn(4, 4, requires_grad=True)
        self.bank = torch.randn(6, 1024, requires_grad=True)
        self.conditional = torch.randn(6, 4, requires_grad=True)
        self.kwargs = dict(image_ids=['i1', 'i2', 'i3'],
                           query_caption_ids=['cat', 'dog', 'bird'],
                           gallery_caption_ids=['cat', 'cat', 'dog', 'bird'],
                           query_concept_ids=['a', 'b', 'c'],
                           mediator_concept_ids=['a', 'a', 'b', 'b', 'c', 'd'])

    def terms(self, **kwargs):
        ze, zi = self.model.encode(self.eeg, self.visuals)
        return self.model.loss_terms(ze, zi, self.text, self.bank, self.conditional,
                                     **self.kwargs, **kwargs)

    def test_historical_initialization_and_loaded_logits_are_bit_identical(self):
        torch.manual_seed(19)
        historical = Aligner(6, [5, 7]).eval()
        for name, value in historical.state_dict().items():
            torch.testing.assert_close(value, self.model.backbone.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(29)
        historical = Aligner(6, [5, 7]).eval()
        state = {k: v.clone() for k, v in historical.state_dict().items()}
        self.model.load_06b_backbone_state(state)
        ze, zi = self.model.encode(self.eeg, self.visuals)
        expected = historical.scale() * F.normalize(historical.eeg(self.eeg), dim=1) @ F.normalize(historical.img(self.visuals), dim=1).T
        torch.testing.assert_close(self.model.image_scores(ze, zi), expected, rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(value, historical.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.model.load_06b_backbone_state({'backbone.'+k: v for k, v in state.items()})

    def test_zero_new_loss_coefficients_preserve_base_value_and_gradient(self):
        ze, zi = self.model.encode(self.eeg, self.visuals)
        base = symmetric_identity_nce(self.model.image_scores(ze, zi), self.kwargs['image_ids'])
        terms = self.model.loss_terms(ze, zi, self.text, self.bank, self.conditional,
                                      **self.kwargs, marginal_weight=0, distill_weight=0)
        torch.testing.assert_close(terms['total'], base, rtol=0, atol=0)
        params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        expected = torch.autograd.grad(base, params, retain_graph=True)
        actual = torch.autograd.grad(terms['total'], params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_marginal_is_live_to_eeg_and_frozen_to_bank_and_conditional(self):
        self.terms()['marginal'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in self.model.backbone.eeg.parameters()))
        for module in (self.model.backbone.img, self.model.readout_eeg, self.model.readout_text):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNone(self.bank.grad)
        self.assertIsNone(self.conditional.grad)
        self.assertIsNone(self.text.grad)

    def test_distillation_is_detached_from_backbone_teacher_and_raw_caption(self):
        self.terms()['distill'].backward()
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.readout_eeg.parameters()))
        self.assertIsNone(self.bank.grad)
        self.assertIsNone(self.conditional.grad)
        self.assertIsNone(self.text.grad)

    def test_same_concept_training_mediator_exclusion_and_shuffled_effect(self):
        ze, zi = self.model.encode(self.eeg, self.visuals)
        qids, mids = self.kwargs['query_concept_ids'], self.kwargs['mediator_concept_ids']
        actual = self.model.caption_teacher(ze, self.bank, self.conditional, qids, mids)
        changed = self.conditional.detach().clone()
        changed[:2] += torch.tensor([100., -100., -100., -100.])
        changed_result = self.model.caption_teacher(ze, self.bank, changed, qids, mids)
        torch.testing.assert_close(actual[0], changed_result[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(actual[1:], changed_result[1:]))
        shuffled = self.model.caption_teacher(ze, self.bank, self.conditional.flip(0), qids, mids)
        self.assertFalse(torch.allclose(actual, shuffled))
        with self.assertRaises(ValueError):
            self.model.caption_teacher(ze, self.bank, self.conditional, ['a']*3, ['a']*6)

    def test_direct_caption_inference_never_calls_image_encoder(self):
        self.assertEqual(list(inspect.signature(self.model.text_scores).parameters), ['ze', 'text'])
        with patch.object(self.model.backbone.img, 'forward', side_effect=AssertionError('Image gallery used')):
            ze = self.model.backbone.eeg(self.eeg)
            scores = self.model.text_scores(ze, self.text)
        self.assertEqual(tuple(scores.shape), (3, 4))
        self.assertTrue(torch.isfinite(scores).all())

    def test_loss_is_finite_and_checkpoint_round_trip_preserves_direct_readout(self):
        terms = self.terms()
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        terms['total'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        restored = LimaAligner(6, [5, 7], 4, rank=3).eval()
        restored.load_state_dict(state, strict=True)
        actual = self.model.text_scores(self.model.backbone.eeg(self.eeg), self.text)
        repeated = restored.text_scores(restored.backbone.eeg(self.eeg), self.text)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
