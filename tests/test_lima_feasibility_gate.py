"""Synthetic CPU evidence for the Issue #5 gate, not THINGS measurements."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from diagnostics import lima_feasibility_gate as gate


class LIMAGateTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(55)

    def tensors(self):
        eeg = torch.randn(7, 5, generator=self.rng)
        images = torch.randn(13, 5, generator=self.rng)
        captions = torch.randn(9, 3, generator=self.rng)
        fitted = (torch.randn(5, 3, generator=self.rng),
                  torch.randn(1, 5, generator=self.rng),
                  torch.randn(1, 3, generator=self.rng))
        return eeg, images, captions, fitted

    def test_chunked_marginal_equals_full_dense_reference(self):
        eeg, images, captions, fitted = self.tensors()
        w, mx, my = fitted
        normalized = F.normalize(images, dim=-1)
        mapped = F.normalize((normalized-mx) @ w+my, dim=-1)
        pi = ((F.normalize(eeg, dim=-1) @ normalized.T)/.07).softmax(-1)
        q = ((mapped @ F.normalize(captions, dim=-1).T)/.07).softmax(-1)
        expected = pi @ q
        for chunk in (1, 3, 100):
            actual = gate.mediated_scores(eeg, images, captions, fitted, query_batch_size=chunk)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(actual.sum(-1), torch.ones(len(eeg)))

    def test_same_concept_mask_matches_dense_reference(self):
        eeg, images, captions, fitted = self.tensors()
        mask = torch.ones(7, 13, dtype=torch.bool)
        mask[:, ::2] = False
        w, mx, my = fitted
        zi = F.normalize(images, dim=-1)
        q = ((F.normalize((zi-mx)@w+my, dim=-1) @ F.normalize(captions, dim=-1).T)/.07).softmax(-1)
        logits = (F.normalize(eeg, dim=-1) @ zi.T)/.07
        expected = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ q
        actual = gate.mediated_scores(eeg, images, captions, fitted,
                                     query_batch_size=2, allowed_mediators=mask)
        torch.testing.assert_close(actual, expected)
        mask[0] = False
        with self.assertRaisesRegex(ValueError, 'permitted'):
            gate.mediated_scores(eeg, images, captions, fitted, allowed_mediators=mask)

    def test_teacher_readonly_and_detached(self):
        values = self.tensors()
        eeg, images, captions, fit = values
        tensors = [eeg, images, captions, *fit]
        before = [value.clone() for value in tensors]
        for value in tensors:
            value.requires_grad_(True)
        result = gate.mediated_scores(*values)
        self.assertFalse(result.requires_grad)
        for value, original in zip(tensors, before):
            self.assertIsNone(value.grad)
            torch.testing.assert_close(value, original)

    def test_exact_split_rejects_leakage_reordering_or_duplicates(self):
        rows = torch.arange(16540)
        fit, query = rows[rows % 10 != 9], rows[rows % 10 == 9]
        gate.validate_split(fit, query)
        bad = fit.clone()
        bad[0] = query[0]
        for invalid in (bad, fit.flip(0), fit[:-1]):
            with self.assertRaisesRegex(ValueError, 'historical'):
                gate.validate_split(invalid, query)
        with self.assertRaisesRegex(ValueError, 'historical'):
            gate.validate_split(fit, query, 16539)

    def test_duplicate_caption_positive_and_deterministic_ties(self):
        scores = torch.tensor([[.4, .4, .2], [.1, .1, .8]])
        hits, _ = gate.caption_hits(scores, ['cat', 'dog'],
                                    ['cat', 'cat', 'dog'], [10, 11, 12])
        self.assertEqual(hits.tolist(), [1., 1.])

    def test_fixed_real_control_fits_never_receive_dev_captions(self):
        # A small mocked split tests data flow; validate_split itself is tested
        # above on the full canonical 16,540-row partition.
        fit = torch.tensor([0, 1, 10, 11])
        query = torch.tensor([9, 19])
        text = torch.randn(20, 3, generator=self.rng)
        images = torch.randn(4, 5, generator=self.rng)
        eeg = torch.randn(2, 5, generator=self.rng)
        identities = [str(i) for i in range(20)]
        original = gate.fit_bridge
        received = []
        def record_fit(x, y, rows, permutation=None):
            received.append((x.clone(), y.clone(), rows.clone(), permutation))
            return original(x, y, rows, permutation)
        with patch.object(gate, 'validate_split'), patch.object(gate, 'fit_bridge', side_effect=record_fit):
            report = gate.evaluate(eeg, images, text, identities, fit, query)
        self.assertEqual(len(received), 2)
        for x, y, rows, _ in received:
            torch.testing.assert_close(x, images)
            torch.testing.assert_close(y, text[fit])
            torch.testing.assert_close(rows, torch.arange(len(fit)))
        self.assertIsNone(received[0][-1])
        permutation = received[1][-1]
        self.assertEqual(sorted(permutation.tolist()), list(range(len(fit))))
        self.assertTrue((fit//10 != fit[permutation]//10).all())
        self.assertEqual(report['image_mediators'], 4)
        self.assertEqual(report['caption_gallery'], 2)
        self.assertIn('same_concept_excluded_sensitivity', report)
        self.assertEqual(report['decision']['gpu_decision'], 'NO_GO')

    def test_gate_threshold_and_positive_ci_are_both_required(self):
        null = torch.zeros(100)
        real = null.clone()
        real[:5] = 1
        with patch.object(gate, 'bootstrap_ci', return_value=[.01, 9.]):
            report = gate.gate_decision(real, null)
            self.assertEqual(report['numerical_gate'], 'PASS')
            self.assertEqual(report['gpu_decision'], 'NO_GO')
            real[4] = 0
            self.assertEqual(gate.gate_decision(real, null)['numerical_gate'], 'FAIL')
        with patch.object(gate, 'bootstrap_ci', return_value=[0., 9.]):
            real[:] = 1
            self.assertEqual(gate.gate_decision(real, null)['numerical_gate'], 'FAIL')

    def test_missing_private_data_is_not_a_measurement_or_load_attempt(self):
        with tempfile.TemporaryDirectory() as temporary, patch('torch.load') as loader:
            root, cache = Path(temporary)/'run', Path(temporary)/'cache'
            report = gate.run(root, cache)
            self.assertEqual(report['status'], 'NOT_MEASURED')
            self.assertEqual(report['gpu_decision'], 'NO_GO')
            self.assertFalse(report['all_required_files_present'])
            self.assertEqual(len(report['missing_inputs']), 16)
            self.assertTrue(any('sub-01_train_z16.npy' in p for p in report['missing_inputs']))
            loader.assert_not_called()
            self.assertFalse(root.exists())
            self.assertFalse(cache.exists())

    def test_inspection_rejects_official_test_before_reading_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'train_only_manifest.json').write_text(json.dumps({
                'partition': 'official_test', 'official_test_opened': False,
                'eeg_file': 'sub-01_test.npy', 'feature_parts': list(gate.PARTS)}))
            with patch('torch.load') as loader, self.assertRaisesRegex(ValueError, 'official-training'):
                gate.inspect_inputs(root, root/'cache')
            loader.assert_not_called()

    def test_exclusive_report_preserves_old_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, cache, output = base/'run', base/'cache', base/'new.json'
            with self.assertRaisesRegex(ValueError, 'outside'):
                gate.write_report({}, root/'artifact.json', root, cache)
            gate.write_report({'status': 'NOT_MEASURED'}, output, root, cache)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                gate.write_report({'status': 'fake'}, output, root, cache)
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_or_nonfinite_inputs_rejected(self):
        values = self.tensors()
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            gate.mediated_scores(*values, query_batch_size=0)
        values[0][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            gate.mediated_scores(*values)
        with self.assertRaisesRegex(ValueError, 'binary'):
            gate.gate_decision(torch.tensor([float('nan')]), torch.zeros(1))

    def baseline_fixture(self, directory):
        from phase1.historical_06b import _code_hashes
        base = Path(directory)
        paths = {name: base/name for name in ('manifest', 'baseline_record', 'checkpoint',
                                             'caption_features', 'caption_ids', 'caption_provenance')}
        for path in paths.values():
            path.write_text('{}')
        rows = torch.arange(16540)
        record = {'status': 'complete_training_only', 'official_test_opened': False,
                  'checkpoint': paths['checkpoint'].name,
                  'checkpoint_sha256': gate.sha256(paths['checkpoint']),
                  'signature': {'manifest_sha256': gate.sha256(paths['manifest']),
                                'code_hashes': _code_hashes()},
                  'fit_rows': rows[rows%10 != 9].tolist(),
                  'validation_rows': rows[rows%10 == 9].tolist(), 'selected_epoch': 25}
        paths['baseline_record'].write_text(json.dumps(record))
        return paths, record

    def test_provenance_failures_stop_before_tensor_loading(self):
        for failure in ('checkpoint', 'manifest', 'code', 'missing_code_set'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                paths, record = self.baseline_fixture(temporary)
                if failure == 'checkpoint':
                    record['checkpoint_sha256'] = 'bad'
                elif failure == 'manifest':
                    record['signature']['manifest_sha256'] = 'bad'
                elif failure == 'code':
                    record['signature']['code_hashes']['models.py'] = 'bad'
                else:
                    record['signature']['code_hashes'] = {}
                paths['baseline_record'].write_text(json.dumps(record))
                availability = {'missing_inputs': []}
                with patch.object(gate, 'inspect_inputs', return_value=(paths, availability)), \
                     patch('phase1.historical_06b.preflight') as preflight, \
                     patch('torch.load') as tensor_load, \
                     self.assertRaisesRegex(ValueError, 'provenance|source'):
                    gate.run(Path(temporary), Path(temporary)/'cache')
                preflight.assert_not_called()
                tensor_load.assert_not_called()

    def test_signed_linux_source_hashes_accept_only_line_ending_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, record = self.baseline_fixture(temporary)
            root = Path(gate.__file__).resolve().parents[1]/'phase1'
            record['signature']['code_hashes'] = {
                name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                for name in record['signature']['code_hashes']}
            gate.validate_baseline_record(record, paths['manifest'], paths['checkpoint'])

    def test_mocked_run_records_schema_and_never_encodes_dev_images(self):
        # Exercise orchestration only. Fake tensors and results are never
        # persisted as real scientific evidence and do not invoke the GPU.
        from phase1.historical_06b import _load_training
        class FrozenFake(torch.nn.Module):
            def img(self, layers):
                return layers[0]
            def eeg(self, eeg):
                return eeg.flatten(1)
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = self.baseline_fixture(temporary)
            all_rows = torch.arange(16540)
            fit, query = all_rows[all_rows%10 != 9], all_rows[all_rows%10 == 9]
            row_values = all_rows.float()[:, None].repeat(1, 2)
            def evaluate_check(eeg_queries, train_images, texts, ids, got_fit, got_query):
                torch.testing.assert_close(eeg_queries[:, 0], query.float())
                torch.testing.assert_close(train_images[:, 0], fit.float())
                self.assertFalse((train_images[:, 0].long()%10 == 9).any())
                self.assertTrue(torch.isfinite(texts).all())
                return {'decision': {'numerical_gate': 'FAIL', 'gpu_decision': 'NO_GO'}}
            with patch.object(gate, 'inspect_inputs', return_value=(paths, {'missing_inputs': []})), \
                 patch('phase1.historical_06b.preflight'), \
                 patch('phase1.trimodal_06b.preflight_text_inputs', return_value={
                     'status': 'cpu_caption_preflight_passed', 'unique_caption_identities': 16540}), \
                 patch('phase1.historical_06b._load_training', return_value=(
                     {}, row_values[:, None, :], [row_values], fit, query, None)), \
                 patch('phase1.models.Aligner', return_value=FrozenFake()), \
                 patch('torch.load', return_value={}), \
                 patch('phase1.trimodal_06b._load_text', return_value=(
                     row_values, [str(i) for i in all_rows.tolist()])), \
                 patch.object(gate, 'evaluate', side_effect=evaluate_check):
                result = gate.run(Path(temporary), Path(temporary)/'cache')
            self.assertEqual(result['status'], 'MEASURED_CPU_ONLY')
            self.assertFalse(result['official_test_opened'])
            self.assertFalse(result['gpu_training_executed'])
            self.assertEqual(result['selected_baseline_epoch'], 25)
            self.assertEqual(result['input_sha256_before'], result['input_sha256_after'])
            self.assertIn('phase1/core.py', result['source_sha256'])
            self.assertIn('phase1/joint_model.py', result['source_sha256'])
            json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
