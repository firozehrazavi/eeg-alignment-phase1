import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from phase1.input_diagnostics import eeg_summary, image_summary


class InputDiagnosticsTests(unittest.TestCase):
    def test_both_containers_report_mismatched_time_axis_without_fixing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            for direct in (False, True):
                path = Path(tmp) / 'eeg.npy'
                data = {'preprocessed_eeg_data': np.full((2, 1, 3, 250), 123456, dtype=np.float32),
                        'times': np.arange(300) / 250 - .2, 'ch_names': ['a', 'b', 'c']}
                if direct:
                    with path.open('wb') as f: pickle.dump(data, f)
                else:
                    np.save(path, data)
                before = path.read_bytes()
                result = eeg_summary(path, 10**9)
                self.assertEqual(result['eeg_shape'], [2, 1, 3, 250])
                self.assertEqual(result['time_shape'], [300])
                self.assertEqual(result['time_count_minus_eeg_samples'], 50)
                self.assertEqual(result['nearest_zero_index_value'], [50, 0.0])
                self.assertAlmostEqual(result['interval_min_median_max'][1], .004)
                self.assertNotIn('123456', json.dumps(result))
                self.assertEqual(before, path.read_bytes())

    def test_ram_guard_does_not_load_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eeg.npy'; path.write_bytes(b'untouched')
            with mock.patch('phase1.input_diagnostics.np.load') as load:
                self.assertIn('skipped', eeg_summary(path, 0)['status'])
                load.assert_not_called()
            self.assertFalse(eeg_summary(Path(tmp) / 'absent.npy', 0)['exists'])

    def test_nonfinite_and_nonnumeric_time_metadata_are_safe_to_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            for times in ([0., np.nan], ['unknown'], [[0., 1.]], []):
                path = Path(tmp) / 'eeg.npy'
                np.save(path, {'preprocessed_eeg_data': np.zeros((1, 2)), 'times': times})
                json.dumps(eeg_summary(path, 10**9), allow_nan=False)

    def test_directory_scan_handles_nested_layout_without_reading_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / 'images_set/images_set'
            train = images / 'training_images/concept'; train.mkdir(parents=True)
            pixel = train / 'example.jpg'; pixel.write_bytes(b'not actual pixels')
            (images / 'image_metadata.npy').write_bytes(b'metadata not loaded')
            result = image_summary([root])[0]
            self.assertEqual(result['image_files_seen'], 1)
            self.assertEqual(result['metadata_paths'], [str(images / 'image_metadata.npy')])
            self.assertEqual(pixel.read_bytes(), b'not actual pixels')
            self.assertTrue(image_summary([root], max_dirs=1)[0]['scan_truncated'])
            self.assertFalse(image_summary([root / 'missing'])[0]['exists'])

    def test_standalone_notebook_matches_tested_source(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/10a_cpu_input_diagnostics.ipynb').read_text())
        code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 1)
        self.assertEqual(''.join(code[0]['source']), (root / 'phase1/input_diagnostics.py').read_text())
        self.assertEqual(code[0]['outputs'], [])
        compile(''.join(code[0]['source']), 'diagnostic_cell', 'exec')


if __name__ == '__main__':
    unittest.main()
