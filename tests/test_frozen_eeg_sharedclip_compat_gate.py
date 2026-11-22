"""CPU-only synthetic checks for frozen EEG -> shared-CLIP compatibility gate.

These tests validate implementation/contract, not EEG scientific performance.
"""
import importlib.util
import unittest
from pathlib import Path

import torch

script = Path(__file__).resolve().parents[1]/'diagnostics'/'frozen_eeg_sharedclip_compat_gate.py'
spec = importlib.util.spec_from_file_location('frozen_eeg_sharedclip_compat_gate', script)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FrozenEEGSharedClipCompatibilityTests(unittest.TestCase):
    def test_fit_first_exemplar_to_dev_last_exemplar(self):
        fit = [10*i for i in range(256)]
        query = gate.split_first_last(fit)
        self.assertEqual(len(query),256)
        self.assertEqual(query[0],9)
        self.assertEqual(query[-1],2559)
        self.assertTrue(all(q % 10 == 9 for q in query))
        self.assertFalse(set(fit) & set(query))

    def test_invalid_training_row_rejected(self):
        rows = [10*i for i in range(256)]
        rows[7] += 9
        with self.assertRaisesRegex(ValueError, 'first-exemplar'):
            gate.split_first_last(rows)

    def test_duplicate_training_concept_rejected(self):
        rows = [10*i for i in range(256)]
        rows[2] = rows[1]
        with self.assertRaisesRegex(ValueError, 'unique'):
            gate.split_first_last(rows)

    def test_out_of_training_bounds_rejected(self):
        rows = [10*i for i in range(255)]+[16540]
        with self.assertRaisesRegex(ValueError, 'first-exemplar'):
            gate.split_first_last(rows)

    @staticmethod
    def mock_features():
        base = torch.eye(256)
        eeg = torch.nn.functional.pad(base,(0,768))
        vectors = torch.nn.functional.pad(base,(0,256))
        ids = [f'synthetic caption {i}' for i in range(256)]
        return eeg, vectors, ids

    def test_paired_retrieval_and_negative_control(self):
        eeg, clip, ids = self.mock_features()
        torch.set_num_threads(min(4,torch.get_num_threads()))
        result = gate.compare_readouts(eeg,eeg,clip,clip,clip,clip,clip,ids,
                                       list(range(9,2560,10)))
        self.assertEqual(result['shared_visual_teacher_frozen_eeg_to_dev_caption']['hits_top1'],256)
        self.assertEqual(result['direct_frozen_eeg_to_shared_caption']['hits_top1'],256)
        self.assertEqual(result['legacy_frozen_eeg_to_original_caption']['hits_top1'],256)
        self.assertEqual(result['cross_concept_shuffled_visual_teacher']['hits_top1'],0)
        self.assertEqual(result['preregistered_cpu_compatibility_gate'],
                         'NO_GO_OR_INCONCLUSIVE_SHARED_VISUAL_TRANSFER')
        self.assertFalse(result['gpu_training_authorized'])

    def test_caption_identity_duplicates_allowed(self):
        eeg, clip, ids = self.mock_features()
        ids[1] = ids[0]
        torch.set_num_threads(min(4,torch.get_num_threads()))
        result = gate.compare_readouts(eeg,eeg,clip,clip,clip,clip,clip,ids,
                                       list(range(9,2560,10)))
        self.assertEqual(result['shared_visual_teacher_frozen_eeg_to_dev_caption']['hits_top1'],256)

    def test_mismatched_dimensions_refused(self):
        eeg, clip, ids = self.mock_features()
        with self.assertRaisesRegex(ValueError, 'shape'):
            gate.compare_readouts(eeg[:255],eeg,clip,clip,clip,clip,clip,ids,
                                  list(range(9,2560,10)))

    def test_nonfinite_feature_refused(self):
        eeg, clip, ids = self.mock_features()
        clip[0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            gate.compare_readouts(eeg,eeg,clip,clip,clip,clip,clip,ids,
                                  list(range(9,2560,10)))


if __name__ == '__main__':
    unittest.main()
