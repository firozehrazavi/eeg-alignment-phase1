import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

class TrainingRuntimeTests(unittest.TestCase):
    def source(self,id):
        p=Path(__file__).resolve().parents[1]/'notebooks/11_training_only_profile.ipynb'
        return ''.join(next(c['source'] for c in json.loads(p.read_text())['cells'] if c.get('id')==id))
    def test_resource_setup_imports_budget_meter_in_fresh_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.source('training-1').replace('APPROVE_GPU_SESSION = False','APPROVE_GPU_SESSION = True').replace('AVAILABLE_CU = 0.0','AVAILABLE_CU = 500.0').replace('CU_PER_HOUR = 0.0','CU_PER_HOUR = 1.0')
            source=source.replace('/content/drive/MyDrive/eeg_results/patch_phase1',root.as_posix())
            colab=ModuleType('google.colab');colab.drive=SimpleNamespace(mount=mock.Mock())
            env={'TRAINING_PREFLIGHT_OK':True,'REPO':root}
            with mock.patch.dict('sys.modules',{'google.colab':colab}),mock.patch.dict(os.environ,{},clear=False),mock.patch('torch.cuda.is_available',return_value=True),mock.patch('torch.cuda.mem_get_info',return_value=(10*2**30,24*2**30)),mock.patch('torch.cuda.get_device_name',return_value='fake L4'),mock.patch('torch.cuda.is_bf16_supported',return_value=True),mock.patch('phase1.patch_resources.runtime_profile',return_value={}),mock.patch('subprocess.run'),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source,'gpu_setup','exec'),env)
            self.assertTrue(env['RESOURCE_SETUP_OK'])
            self.assertIsInstance(env['meter'],env['BudgetMeter'])
            self.assertTrue(callable(env['monitored']))
            self.assertTrue(list(root.glob('runtime_*.json')))
    def test_skipped_predecessors_fail_before_work(self):
        for id,message in [('training-1','cell 1'),('training-2','cell 2'),('training-3','cell 3')]:
            with self.subTest(id=id),self.assertRaisesRegex(AssertionError,message):
                exec(compile(self.source(id),id,'exec'),{})
