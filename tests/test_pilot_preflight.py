import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from phase1.pilot_preflight import preflight
from phase1.patch_cache import digest

class PreflightTests(unittest.TestCase):
    def test_headers_and_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cat={};m={'catalog':'catalog.json','subjects':{'sub-01':{}},'pooled':{},
                'preprocessing':{'regime':'benchmark_preprocessed'}};done={}
            for s,n in [('train',20),('test',200)]:
                cat[s]=[dict(id=f'{s}{i}',concept=f'{s}c{i}',sha256=f'{s}hash{i}') for i in range(n)]
                m['pooled'][s]=[]
                for k in range(5):
                    name=f'{s}{k}.npy';np.save(root/name,np.zeros((n,3200),dtype='float32'))
                    m['pooled'][s].append(name);done[name]='header-only'
                name=f'{s}_eeg.npy';np.save(root/name,np.zeros((n,17,250),dtype='float32'))
                m['subjects']['sub-01'][s]=name;done[name]='header-only'
            for name,obj in [('dataset.json',m),('catalog.json',cat)]:
                (root/name).write_text(json.dumps(obj));done[name]=digest(root/name)
            (root/'preparation_state.json').write_text(json.dumps({'completed':done}))
            cfg=Path(__file__).resolve().parents[1]/'configs/patch_pilot.json'
            result=preflight(root/'dataset.json',cfg)
            self.assertEqual(result['status'],'header_preflight_passed')
            np.save(root/'test_eeg.npy',np.zeros((200,17,249),dtype='float32'))
            with self.assertRaises(ValueError):preflight(root/'dataset.json',cfg)
            (root/'test_eeg.npy').unlink()
            with self.assertRaises(FileNotFoundError):preflight(root/'dataset.json',cfg)
