import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from phase1.pilot_inputs import CHANNELS, inspect_eeg, inventory, validate_eeg_times


class ObservedInputsTests(unittest.TestCase):
    def test_verification_notebook_is_existing_only_and_cpu_only(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/10b_cpu_verify_observed_inputs.ipynb'
        cells=json.loads(path.read_text())['cells']
        code=[c for c in cells if c['cell_type']=='code']
        self.assertEqual(len(code),1)
        source=''.join(code[0]['source']);compile(source,'cpu_verifier','exec')
        self.assertIn('assert not torch.cuda.is_available()',source)
        self.assertIn("'--verify-existing-sub01'",source)
        self.assertNotIn('--download-sub01',source)
        self.assertNotIn('phase1.patch_prepare',source)
        self.assertNotIn('phase1.patch_train',source)
        self.assertEqual(code[0]['outputs'],[])

    def test_observed_float64_dictionary_and_scalar_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'eeg.npy'
            for repetitions in (4,80):
                # Reduced image count; actual repetition/channel/time axes and dtype.
                shape=(2,repetitions,63,250)
                data={'preprocessed_eeg_data':np.zeros(shape,dtype=np.float64),
                      'ch_names':CHANNELS,'times':np.arange(300)/250-.2}
                for direct in (True,False):
                    if direct:
                        with path.open('wb') as f:pickle.dump(data,f,protocol=4)
                    else:np.save(path,data)
                    before=path.read_bytes();result=inspect_eeg(path,shape,check_ram=False)
                    self.assertEqual(path.read_bytes(),before)
                    self.assertEqual(result['time_metadata_offset_samples'],50)
                    self.assertEqual(result['time_samples'],250)
                    self.assertAlmostEqual(result['time_start_seconds'],0)
                    self.assertAlmostEqual(result['time_end_seconds'],.996)
                    self.assertTrue(result['finite'] and result['channels_metadata_verified'])
                    self.assertEqual(result['dtype'],'float64')

    def test_rejects_unsupported_time_conventions(self):
        t=np.arange(300)/250-.2
        bad=[t+.004,t[:-1],t[::-1],t*1000,t.reshape(3,100),np.where(np.arange(300)==42,np.nan,t)]
        jitter=t.copy();jitter[100]+=.0001;bad.append(jitter)
        for times in bad:
            with self.subTest(shape=times.shape):
                with self.assertRaises(ValueError):validate_eeg_times(times,250)
        with self.assertRaises(ValueError):validate_eeg_times(t,249)
        self.assertEqual(validate_eeg_times(np.arange(250)/250,250)['time_metadata_offset_samples'],0)

    def test_inventory_full_counts_roots_order_and_late_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir();images=root/'images_set';images.mkdir()
            raw={};meta={};features={};existing=set()
            for split,key,n in [('train','tr',16540),('test','te',200)]:
                names=[f'{split}_{i}.jpg' for i in range(n)]
                rels=[f'images_set/{split}ing_images/concept/{name}' for name in names]
                raw[f'{split}_img_files']=names;meta[f'{split}_files']=names;meta[f'{split}_rel']=rels
                existing.update(str(root/p) for p in rels)
                features[key]={(layer,'mean'):torch.zeros(1).expand(n,3200) for layer in [20,24,28,32,36]}
            np.save(images/'image_metadata.npy',raw)
            torch.save(meta,cache/'meta_v6.pt');torch.save(features,cache/'feat6_internvit6b.pt')
            original=Path.is_file;seen=set()
            def is_file(path):
                if path.suffix=='.jpg':seen.add(str(path));return str(path) in existing
                return original(path)
            with mock.patch.object(Path,'is_file',is_file):
                for candidate in (root,images):
                    result=inventory(candidate,cache)
                    self.assertTrue(result['ready_images_and_features'])
                    self.assertEqual(result['image_root'],str(root))
                    self.assertEqual(result['image_roots'][0]['splits']['train']['paths_checked'],16540)
                self.assertEqual(len(seen),16740)
                missing=str(root/meta['train_rel'][-1]);existing.remove(missing)
                result=inventory(root,cache)
                self.assertFalse(result['ready_images_and_features'])
                self.assertIn('train_referenced_images_missing',result['image_roots'][0]['reasons'])
                existing.add(missing)
                meta['train_files']=list(reversed(meta['train_files']));torch.save(meta,cache/'meta_v6.pt')
                result=inventory(root,cache)
                self.assertIn('train_metadata_count_or_order_mismatch',result['image_roots'][0]['reasons'])
                self.assertIsNone(result['image_root'])


if __name__=='__main__':unittest.main()
