import json
import os
from pathlib import Path
import unittest
from unittest import mock
from phase1.extraction_environment import configure

class ExtractionEnvironmentTests(unittest.TestCase):
    def test_backend_settings_override_auto(self):
        with mock.patch.dict(os.environ,{'USE_FLAX':'AUTO','USE_TF':'1'}):
            configure()
            for k,v in {'USE_FLAX':'0','USE_TF':'0','USE_TORCH':'1','USE_TORCH_XLA':'0'}.items():
                self.assertEqual(os.environ[k],v)
    def test_smoke_precedes_probe_and_extractor_configures_before_import(self):
        root=Path(__file__).resolve().parents[1]
        cells=json.loads((root/'notebooks/10_rpa_patch_pilot.ipynb').read_text())['cells']
        setup=''.join(next(c['source'] for c in cells if c.get('id')=='proplus-02'))
        self.assertLess(setup.index('phase1.extraction_environment'),setup.index('PREPARED_INPUTS_CHECKED = True'))
        source=(root/'phase1/patch_cache.py').read_text()
        self.assertLess(source.index('    configure()'),source.index('    from transformers import'))
