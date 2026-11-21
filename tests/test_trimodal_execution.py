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
