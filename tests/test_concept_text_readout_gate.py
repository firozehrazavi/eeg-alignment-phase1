"""Read-only CPU regressions for the concept-text diagnostic; synthetic data only."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / 'diagnostics' / 'concept_text_readout_gate.py'
spec = importlib.util.spec_from_file_location('concept_gate', SRC)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture():
    names = [f'{i+1:05d}_concept_{i+1}' for i in range(1654)]
    concept_rows = [s for s in names for _ in range(10)]
    image_names = [f'example_{i+1:05d}_{j}.jpg' for i in range(1654) for j in range(10)]
    captions = [f'a photo of concept {i+1}' for i in range(1654) for _ in range(10)]
    return names, concept_rows, image_names, captions


class ConceptSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names, cls.concepts, cls.files, cls.captions = fixture()

    def test_canonical_train_concept_blocks(self):
        labels = gate.canonical_concepts(self.concepts, self.files)
        self.assertEqual(len(labels), 1654)
        self.assertEqual(labels[0], 'concept 1')
        self.assertEqual(labels[-1], 'concept 1654')

    def test_wrong_concept_order_fails_closed(self):
        damaged = list(self.concepts)
        damaged[100] = '00077_incorrect'
        with self.assertRaises(ValueError):
            gate.canonical_concepts(damaged, self.files)

    def test_wrong_image_duplicate_fails_closed(self):
        damaged = list(self.files)
        damaged[4] = damaged[0]
        with self.assertRaises(ValueError):
            gate.canonical_concepts(self.concepts, damaged)

    def _build_sources(self, root):
        root = Path(root)
        metadata = root/'image_metadata.npy'
        np.save(metadata, {'train_img_files': self.files,
                           'train_img_concepts': self.concepts})
        meta = root/'meta_v6.pt'
        torch.save({'train_files': self.files,
                    'train_rel': ['images_set/training_images/'+self.concepts[i]+'/'+n
                                  for i, n in enumerate(self.files)]}, meta)
        source = root/'captions_blip_large.json'
        ids = root/'caption_ids_train_only.json'
        source.write_text(json.dumps({'train': self.captions}), encoding='utf-8')
        ids.write_text(json.dumps(self.captions), encoding='utf-8')
        prov = root/'caption_provenance_train_only.json'
        prov.write_text(json.dumps({'partition': 'official_training', 'used_test': False,
                                    'encoder_revision': 'unavailable',
                                    'source_sha256': {meta.name: gate.digest(meta),
                                                      source.name: gate.digest(source)}}))
        return metadata, meta, source, ids, prov

    def test_exact_source_provenance_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_sources(directory)
            report, labels = gate.source_audit(*paths)
            self.assertEqual(report['status'], 'SOURCE_ORDER_AND_TEXT_VERIFIED')
            self.assertTrue(report['blip_text_equals_caption_ids_all_rows'])
            self.assertEqual(report['eeg_stimulus_identity'], 'NOT_INDEPENDENTLY_VERIFIED')
            self.assertEqual(len(labels), 1654)

    def test_caption_ids_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_sources(directory)
            ids = json.loads(paths[3].read_text())
            ids[0] = 'fabricated target'
            paths[3].write_text(json.dumps(ids))
            with self.assertRaisesRegex(ValueError, 'do not exactly match'):
                gate.source_audit(*paths)

    def test_image_metadata_order_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_sources(directory)
            data = np.load(paths[0], allow_pickle=True).item()
            data['train_img_files'][0], data['train_img_files'][1] = (data['train_img_files'][1],
                                                                     data['train_img_files'][0])
            np.save(paths[0], data)
            with self.assertRaisesRegex(ValueError, 'differ from official'):
                gate.source_audit(*paths)

    def test_provenance_source_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._build_sources(directory)
            prov = json.loads(paths[4].read_text())
            prov['source_sha256']['meta_v6.pt'] = '0'*64
            paths[4].write_text(json.dumps(prov))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                gate.source_audit(*paths)

    def test_lexical_diagnostic_is_not_accuracy(self):
        names = ['concept'] * 1654
        caps = ['photo of a concept'] * 10 + ['an unrelated description'] * 16530
        summary = gate.literal_name_audit(names, caps)
        self.assertEqual(summary['literal_name_mention_rows'], 10)
        self.assertEqual(summary['concepts_without_literal_mention'], 1653)
        self.assertIn('NOT', summary['note'])

    def test_encoder_only_uses_cpu_and_exact_prompt_template(self):
        class MockEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
            def encode_text(self, tokens):
                return torch.ones((len(tokens), 512))
        seen = []
        def tokenizer(strings):
            seen.extend(strings)
            return torch.zeros((len(strings), 77), dtype=torch.long)
        fake = types.SimpleNamespace(
            create_model_and_transforms=lambda model, pretrained, device:
                (MockEncoder(), None, None),
            get_tokenizer=lambda model: tokenizer,
            __version__='test-only')
        with patch.dict(sys.modules, {'open_clip': fake}):
            reps, metadata = gate.encode_concept_prompts(['aardvark'] * 1654)
        self.assertEqual(tuple(reps.shape), (1654, 512))
        self.assertEqual(seen[0], 'a photo of a aardvark')
        self.assertEqual(metadata['legacy_encoder_exact_weights_match'],
                         'NOT_VERIFIABLE_ORIGINAL_REVISION_UNPINNED')
        self.assertTrue(torch.isfinite(reps).all())
        self.assertFalse(reps.requires_grad)


if __name__ == '__main__':
    unittest.main()
