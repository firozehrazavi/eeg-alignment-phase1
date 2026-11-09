import json
from pathlib import Path
import unittest
class FiveEpochNotebookTests(unittest.TestCase):
    def test_only_train_only_is_exposed(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/12_controlled_five_epoch_training.ipynb'
        cells=json.loads(p.read_text())['cells'];src=[''.join(c['source']) for c in cells if c['cell_type']=='code']
        for s in src:compile(s,'cell','exec')
        self.assertIn('APPROVE_FIVE_EPOCH_TRAINING = False',src[-1])
        self.assertIn("'--train-only'",src[-1]);self.assertNotIn('--profile-one-epoch',src[-1])
        self.assertIn('Checkpoint code mismatch',src[0])
        self.assertIn('controlled_sub01_five_epochs_v1',src[0])
        self.assertNotIn('from_pretrained',''.join(src))
