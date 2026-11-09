import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
from phase1.training_preflight import check

class TrainingEntryTests(unittest.TestCase):
    def test_completed_tokens_and_disk_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m={'catalog':'catalog.json','tokens':'tokens_L28','pooled':{'train':[],'test':[]},'subjects':{}}
            (root/'dataset.json').write_text(json.dumps(m));(root/'config.json').write_text('{}')
            for name in ['preparation_state.json','catalog.json','tokens_L28/manifest.json','tokens_L28/train.npy','tokens_L28/test.npy','tokens_L28/train_done.npy','tokens_L28/test_done.npy']:
                p=root/name;p.parent.mkdir(exist_ok=True);p.write_text('{}')
            token=SimpleNamespace(meta={'layer':28,'width':3200},arrays={'train':SimpleNamespace(shape=(2000,1024,3200)),'test':SimpleNamespace(shape=(200,1024,3200))})
            with mock.patch('phase1.training_preflight.preflight',return_value={}),mock.patch('phase1.training_preflight.TokenCache',return_value=token),mock.patch('phase1.training_preflight.shutil.disk_usage') as disk:
                disk.return_value.free=100000
                self.assertEqual(check(root/'dataset.json',root/'config.json',root/'new',root)['checkpoint_status'],'new')
                disk.return_value.free=0
                with self.assertRaises(OSError):check(root/'dataset.json',root/'config.json',root/'new',root)
    def test_notebook_has_no_extraction_or_full_training_command(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/11_training_only_profile.ipynb'
        cells=json.loads(p.read_text())['cells'];sources=[''.join(c['source']) for c in cells if c['cell_type']=='code']
        for source in sources:compile(source,'training_notebook','exec')
        self.assertNotIn("'--revision'", ''.join(sources))
        self.assertNotIn('from_pretrained', ''.join(sources))
        self.assertIn("'--profile-one-epoch'",sources[-1])
        self.assertIn('APPROVE_ONE_EPOCH_PROFILE = False',sources[-1])
        self.assertIn("os.environ['CUDA_VISIBLE_DEVICES']=''",sources[0])
