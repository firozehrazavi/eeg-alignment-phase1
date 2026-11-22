"""Synthetic CPU tests for one fixed train-only shared CLIP teacher gate."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

path = Path(__file__).resolve().parents[1]/'diagnostics'/'shared_clip_teacher_gate.py'
spec = importlib.util.spec_from_file_location('shared_clip_teacher_gate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SharedClipTeacherTests(unittest.TestCase):

    def catalog(self):
        concepts = [f'{c+1:05d}_item_{c+1}' for c in range(1654) for _ in range(10)]
        filenames = [f'item_{c+1}_image_{j+1}.jpg' for c in range(1654) for j in range(10)]
        meta = {'train_rel': [f'images_set/training_images/{concepts[i]}/{f}'
                              for i,f in enumerate(filenames)],
                'train_files': list(filenames)}
        official = {'train_img_concepts': concepts, 'train_img_files': list(filenames)}
        return meta, official

    def test_fixed_train_only_rows(self):
        meta, official = self.catalog()
        rows, names, paths = mod.select_training_rows(meta, official)
        self.assertEqual(len(rows), 256)
        self.assertEqual(len(set(i//10 for i in rows)), 256)
        self.assertTrue(all(i%10 == 0 for i in rows))
        self.assertEqual(len(set(names)), 256)
        self.assertTrue(all(p.startswith('images_set/training_images/') for p in paths))
        self.assertEqual(rows, mod.select_training_rows(meta, official)[0])

    def test_rejects_swapped_original_filenames(self):
        meta, official = self.catalog()
        official['train_img_files'][3], official['train_img_files'][4] = (
            official['train_img_files'][4], official['train_img_files'][3])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            mod.select_training_rows(meta, official)

    def test_rejects_test_partition_paths(self):
        meta, official = self.catalog()
        meta['train_rel'][0] = 'images_set/test_images/00001_test.jpg'
        with self.assertRaisesRegex(ValueError, 'test or traversal'):
            mod.select_training_rows(meta, official)

    def test_rejects_path_traversal(self):
        meta, official = self.catalog()
        meta['train_rel'][0] = 'images_set/training_images/../../test_images/000.jpg'
        with self.assertRaises(ValueError):
            mod.select_training_rows(meta, official)

    def test_rejects_misordered_concept_blocks(self):
        meta, official = self.catalog()
        official['train_img_concepts'][10] = '00001_item_1'
        with self.assertRaisesRegex(ValueError, 'concept block'):
            mod.select_training_rows(meta, official)

    def test_fixed_positive_vs_shuffled(self):
        v = torch.eye(256)
        captions = [f'caption_{i}' for i in range(256)]
        names = [f'concept_{i}' for i in range(256)]
        result = mod.score_conditions(v, v, v, captions, names, list(range(0, 2560, 10)))
        self.assertEqual(result['image_to_existing_blip_caption']['real']['top1_hits'], 256)
        self.assertEqual(result['image_to_existing_blip_caption']['shuffled_vectors']['top1_hits'], 0)
        self.assertEqual(result['image_to_canonical_concept_text']['real']['top1_hits'], 256)
        self.assertEqual(result['teacher_preflight'], 'PASS_SHARED_TEACHER_ONLY_NOT_GPU_APPROVAL')
        self.assertFalse(result['gpu_training_authorized'])

    def test_noisy_image_teacher_is_no_go(self):
        v = torch.eye(256)
        c = v.roll(1, dims=0)
        names = [f'concept_{i}' for i in range(256)]
        r = mod.score_conditions(v, c, c, names, names, list(range(256)))
        self.assertEqual(r['teacher_preflight'], 'NO_GO_SHARED_TEACHER')
        self.assertFalse(r['gpu_training_authorized'])

    def test_duplicate_captions_are_not_artificially_wrong(self):
        v = torch.eye(256)
        ids = [f'caption_{i}' for i in range(256)]
        ids[1] = ids[0]  # Both rows are acceptable for this target text.
        names = [f'concept_{i}' for i in range(256)]
        report = mod.score_conditions(v, v, v, ids, names, list(range(256)))
        self.assertEqual(report['image_to_existing_blip_caption']['real']['top1_hits'], 256)

    def test_nonfinite_and_shape_refused(self):
        v = torch.eye(256)
        v[0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            mod.normalized_scores(v, torch.eye(256))
        with self.assertRaisesRegex(ValueError, 'matching finite'):
            mod.normalized_scores(torch.eye(256), torch.eye(255))

    def test_duplicate_concepts_refused(self):
        v = torch.eye(256)
        labels = [f'concept_{i}' for i in range(256)]
        labels[2] = labels[1]
        with self.assertRaisesRegex(ValueError, 'Nonunique'):
            mod.score_conditions(v, v, v, labels, labels, list(range(256)))


if __name__ == '__main__':
    unittest.main()
