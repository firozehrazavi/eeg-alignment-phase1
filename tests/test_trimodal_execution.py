"""New execution controls only; tiny CPU fixtures, no official data or CUDA."""
import json
import unittest
from unittest.mock import patch

import torch
import test_trimodal_06b as fixtures
from phase1 import trimodal_06b as runner


class ExecutionTests(unittest.TestCase):
    setUp = fixtures.Historical06BTrimodalTests.setUp
    _baseline = fixtures.Historical06BTrimodalTests._baseline

    def invoke(self, directory, **kwargs):
        return runner.run(self.manifest, self.root / 'baseline', self.text_path,
                          self.ids_path, self.provenance_path, self.root / directory,
                          device_name='cpu', **kwargs)

    def metadata(self, directory):
        return json.loads((self.root / directory / 'run.json').read_text())

    def assert_states_equal(self, left, right):
        for arm in 'ABCD':
            filename = f'sub-01_s0_{arm}_best.pt'
            expected = torch.load(self.root / left / filename, weights_only=True)
            actual = torch.load(self.root / right / filename, weights_only=True)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_selected_arm_then_other_arms_match_all_at_once(self):
        self._baseline()
        self.invoke('reference')
        selected = self.invoke('selected', arms=('B',), resume=True)
        self.assertEqual([r['arm'] for r in selected], ['B'])
        self.assertEqual(self.metadata('selected')['status'], 'partial_training_only')
        self.assertFalse((self.root / 'selected/sub-01_s0_A_latest.pt').exists())
        completed = self.invoke('selected', arms=('D', 'A', 'C'), resume=True)
        self.assertEqual({r['arm'] for r in completed}, set('ABCD'))
        self.assertEqual(self.metadata('selected')['status'], 'complete_training_only')
        self.assert_states_equal('reference', 'selected')

    def test_epoch_ceiling_is_idempotent_and_resume_keeps_full_schedule(self):
        self._baseline()
        self.invoke('reference')
        self.assertEqual(self.invoke('profile', max_epochs_per_arm=1), [])
        self.assertEqual(self.metadata('profile')['status'], 'paused_epoch_limit')
        for arm in 'ABCD':
            checkpoint = torch.load(self.root / f'profile/sub-01_s0_{arm}_latest.pt',
                                    weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['scheduler']['total_steps'], 2)
            self.assertEqual(checkpoint['signature']['epochs'], 2)
        with patch.object(torch.optim.AdamW, 'step',
                          side_effect=AssertionError('Ceiling rerun updated weights')):
            self.invoke('profile', max_epochs_per_arm=1, resume=True)
        self.invoke('profile', resume=True)
        self.assert_states_equal('reference', 'profile')

    def test_zero_budget_stops_without_updates_and_can_resume(self):
        self._baseline()
        with patch.object(torch.optim.AdamW, 'step',
                          side_effect=AssertionError('Zero budget updated weights')):
            self.assertEqual(self.invoke('zero', max_seconds=0), [])
        self.assertEqual(self.metadata('zero')['status'], 'paused_budget')
        self.assertFalse(list((self.root / 'zero').glob('*_latest.pt')))
        self.invoke('zero', arms=('A',), resume=True)
        self.assertEqual(self.metadata('zero')['status'], 'partial_training_only')

    def test_budget_expiry_finishes_epoch_and_resumes_bit_exactly(self):
        self._baseline()
        with patch.object(runner, 'BATCH', 2):
            self.invoke('reference')
            updates = []
            original_step = torch.optim.AdamW.step

            def step(optimizer, *args, **kwargs):
                result = original_step(optimizer, *args, **kwargs)
                updates.append(1)
                return result

            with patch.object(torch.optim.AdamW, 'step', step), \
                 patch.object(runner, '_budget_expired',
                              side_effect=lambda *a: bool(updates)):
                self.invoke('budget', max_seconds=100)
            # Two fit batches comprise one epoch; expiry after batch one must
            # not cut it in half or start epoch two.
            self.assertEqual(len(updates), 2)
            latest = torch.load(self.root / 'budget/sub-01_s0_A_latest.pt', weights_only=True)
            self.assertEqual(latest['epoch'], 1)
            self.assertEqual(self.metadata('budget')['status'], 'paused_budget')
            self.assertFalse((self.root / 'budget/results.csv').exists())
            self.invoke('budget', resume=True)
            self.assert_states_equal('reference', 'budget')

    def test_semantic_correction_requires_fresh_bd_only_output(self):
        # Guardrails must fire before reading real data or touching old outputs.
        with patch.object(runner, 'baseline_preflight',
                          side_effect=AssertionError('Invalid experiment accessed data')):
            for arms in (('A',), ('C',), ('B','C'), ('A','B','C','D')):
                with self.subTest(arms=arms), self.assertRaisesRegex(ValueError, 'only permits arms B and D'):
                    self.invoke('trimodal_semantic_correction_bd_fixture',
                                arms=arms, semantic_correction=True)
            with self.assertRaisesRegex(ValueError, 'separate'):
                self.invoke('reference', arms=('B','D'), semantic_correction=True)

    def test_semantic_correction_bd_resume_and_artifact_identity(self):
        self._baseline()
        dirname = 'trimodal_semantic_correction_bd_fixture'
        records = self.invoke(dirname, arms=('B','D'), resume=True,
                              semantic_correction=True, max_epochs_per_arm=1)
        self.assertEqual(records, [])
        state = self.metadata(dirname)
        self.assertEqual(state['status'], 'paused_epoch_limit')
        self.assertTrue(state['signature']['semantic_correction'])
        self.assertEqual(state['signature']['joint_alpha'], 20.0)
        self.assertFalse(state['official_test_opened'])
        self.assertFalse((self.root / dirname / 'sub-01_s0_A_latest.pt').exists())
        prep = torch.load(self.root / dirname / 'preprocessing.pt', weights_only=True)
        dims = prep['model_dimensions']
        self.assertTrue(dims['semantic_correction'])
        self.assertEqual(dims['alpha'], 20.)
        with patch.object(torch.optim.AdamW, 'step',
                          side_effect=AssertionError('Repeated corrected epoch')):
            self.invoke(dirname, arms=('B','D'), resume=True,
                        semantic_correction=True, max_epochs_per_arm=1)
        finished = self.invoke(dirname, arms=('B','D'), resume=True,
                               semantic_correction=True)
        self.assertEqual({r['arm'] for r in finished}, {'B','D'})
        self.assertEqual(self.metadata(dirname)['status'], 'complete_training_only')
        # A completed run returns numeric Python values; reloading results.csv
        # via csv.DictReader returns string values. Test persisted identities,
        # artifact integrity and zero optimizer updates instead of mismatched
        # Python types from two equivalent representations.
        import csv
        records_path = self.root / dirname / 'results.csv'
        original_csv = records_path.read_bytes()
        checkpoint_paths = [
            self.root / dirname / f'sub-01_s0_{arm}_latest.pt'
            for arm in 'BD'
        ] + [
            self.root / dirname / f'sub-01_s0_{arm}_best.pt'
            for arm in 'BD'
        ]
        original_checkpoints = {p: p.read_bytes() for p in checkpoint_paths}
        with patch.object(torch.optim.AdamW, 'step',
                          side_effect=AssertionError('Corrected experiment restarted')):
            same = self.invoke(dirname, arms=('B','D'), resume=True,
                               semantic_correction=True)
        with records_path.open(newline='') as handle:
            persisted = list(csv.DictReader(handle))
        self.assertEqual(same, persisted)
        self.assertEqual({row['arm'] for row in same}, {'B', 'D'})
        self.assertEqual(records_path.read_bytes(), original_csv)
        self.assertEqual({p: p.read_bytes() for p in checkpoint_paths},
                         original_checkpoints)
        # Even the already-complete fast path must reject deleted/tampered
        # preprocessing without re-hashing and legitimizing corrupted bytes.
        prep_path = self.root / dirname / 'preprocessing.pt'
        original_prep = prep_path.read_bytes()
        record_path = self.root / dirname / 'run.json'
        original_record = record_path.read_bytes()
        for changed in (False, True):
            with self.subTest(changed=changed):
                if changed:
                    prep_path.write_bytes(original_prep + b'corruption')
                else:
                    prep_path.unlink()
                try:
                    with patch.object(torch.optim.AdamW, 'step',
                                      side_effect=AssertionError('Corrupt resume trained')), \
                         self.assertRaisesRegex(ValueError, 'preprocessing artifact'):
                        self.invoke(dirname, arms=('B','D'), resume=True,
                                    semantic_correction=True)
                    self.assertEqual(record_path.read_bytes(), original_record)
                finally:
                    prep_path.write_bytes(original_prep)
        with self.assertRaisesRegex(ValueError, 'signature|Resume'):
            self.invoke(dirname, arms=('B','D'), resume=True, semantic_correction=False)

    def test_tri_residual_teacher_gradient_has_no_image_base_path(self):
        from phase1.joint_model import JointAligner
        model = JointAligner(3, [4], 5, rank=2, alpha=20.,
                             semantic_correction=True).eval()
        torch.manual_seed(1)
        e = torch.randn(3, 1024, requires_grad=True)
        v = torch.randn(3, 1024, requires_grad=True)
        r = torch.randn(3, 5)
        scores = model.joint_scores(e, v, r)
        base = model.image_scores(e, v)
        interaction = scores - base.detach()
        grad = torch.autograd.grad(scores.sum(), e, retain_graph=True)[0]
        pure_grad = torch.autograd.grad(interaction.sum(), e, retain_graph=True)[0]
        torch.testing.assert_close(grad, pure_grad, atol=1e-7, rtol=1e-6)
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertFalse(torch.allclose(
            grad, torch.autograd.grad(base.sum(), e)[0]))

    def test_live_caption_gradient_passes_from_text_to_eeg(self):
        from phase1.joint_model import JointAligner
        model = JointAligner(3, [4], 5, rank=2, alpha=20.,
                             semantic_correction=True)
        ze = torch.randn(3, 1024, requires_grad=True)
        text = torch.randn(3, 5)
        identity = ['a', 'b', 'c']
        loss_live = runner.symmetric_identity_nce(model.text_scores(ze, text),
                                                   identity, identity)
        g_live = torch.autograd.grad(loss_live, ze)[0]
        self.assertTrue(torch.isfinite(g_live).all())
        self.assertGreater(float(g_live.norm()), 0.)
        # Legacy readout intentionally trained only its own heads, never EEG.
        loss_detached = runner.symmetric_identity_nce(
            model.text_scores(ze.detach(), text), identity, identity)
        g_detached = torch.autograd.grad(loss_detached, ze,
                                         allow_unused=True)[0]
        self.assertIsNone(g_detached)

    def test_live_caption_runs_only_on_new_bd_output_and_resumes(self):
        self._baseline()
        with self.assertRaisesRegex(ValueError, 'requires corrected'):
            self.invoke('trimodal_live_caption_bd_fixture', arms=('B', 'D'),
                        resume=True, live_caption_gradient=True)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.invoke('trimodal_semantic_correction_bd_fixture', arms=('B','D'),
                        resume=True, semantic_correction=True, live_caption_gradient=True)
        name = 'trimodal_live_caption_bd_fixture'
        first = self.invoke(name, arms=('B','D'), resume=True,
                            semantic_correction=True, live_caption_gradient=True,
                            max_epochs_per_arm=1)
        self.assertEqual(first, [])
        metadata = self.metadata(name)
        self.assertEqual(metadata['status'], 'paused_epoch_limit')
        self.assertTrue(metadata['signature']['live_caption_gradient'])
        for arm in 'BD':
            checkpoint = torch.load(self.root / name / f'sub-01_s0_{arm}_latest.pt',
                                    weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertTrue(checkpoint['signature']['live_caption_gradient'])
        with patch.object(torch.optim.AdamW, 'step',
                          side_effect=AssertionError('Ceiling trained twice')):
            self.invoke(name, arms=('B','D'), resume=True,
                        semantic_correction=True, live_caption_gradient=True,
                        max_epochs_per_arm=1)
        # Switching off the live-caption objective on the same saved path
        # is rejected by the fresh-output-name guard before signature checks.
        with self.assertRaisesRegex(ValueError, 'separate.*output directory'):
            self.invoke(name, arms=('B','D'), resume=True,
                        semantic_correction=True, live_caption_gradient=False)
        # With the same experiment name, the signed training protocol still
        # rejects an altered loss weight (and may not update stored files).
        run_record = self.root / name / 'run.json'
        before_record = run_record.read_bytes()
        with patch.object(runner, 'JOINT_WEIGHT', runner.JOINT_WEIGHT + .01):
            with self.assertRaisesRegex(ValueError, 'signature|Resume'):
                self.invoke(name, arms=('B','D'), resume=True,
                            semantic_correction=True, live_caption_gradient=True)
        self.assertEqual(run_record.read_bytes(), before_record)

    def test_invalid_execution_options_are_rejected_before_reading_inputs(self):
        options = [dict(arms=()), dict(arms=('A', 'A')), dict(arms=('E',)),
                   dict(max_seconds=-1), dict(max_seconds=float('nan')),
                   dict(max_seconds=float('inf')), dict(max_epochs_per_arm=0),
                   dict(max_epochs_per_arm=1.5)]
        with patch.object(runner, 'baseline_preflight',
                          side_effect=AssertionError('Invalid options read inputs')):
            for option in options:
                with self.subTest(option=option), self.assertRaises(ValueError):
                    self.invoke('invalid', **option)


if __name__ == '__main__':
    unittest.main()
