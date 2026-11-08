import hashlib
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from phase1.pilot_inputs import FILES,CHANNELS,fetch_files,inspect_eeg,inventory,verify_bytes,load_dictionary


class PilotInputTests(unittest.TestCase):
    def write_eeg(self,path,data,format):
        if format=='direct_dict':
            with open(path,'wb') as f:pickle.dump(data,f,protocol=4)
        else:np.save(path,data)

    def test_inspect_eeg_direct_dict_and_scalar_object_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape=(2,1,63,4)
            data={'preprocessed_eeg_data':np.ones(shape,dtype=np.float32),
                  'ch_names':CHANNELS,'times':np.arange(4)/250}
            reports=[]
            for format in ['direct_dict','scalar_object_array']:
                with self.subTest(format=format):
                    path=Path(tmp)/(format+'.npy');self.write_eeg(path,data,format)
                    loaded=np.load(path,allow_pickle=True)
                    self.assertIsInstance(loaded,dict if format=='direct_dict' else np.ndarray)
                    before=path.read_bytes();reports.append(inspect_eeg(path,shape,check_ram=False))
                    self.assertEqual(before,path.read_bytes())
            self.assertEqual(reports[0],reports[1])
            self.assertTrue(reports[0]['finite'] and reports[0]['channels_metadata_verified'] and reports[0]['time_metadata_verified'])

    def test_both_formats_preserve_all_eeg_validation(self):
        shape=(2,1,63,4)
        original={'preprocessed_eeg_data':np.ones(shape,dtype=np.float32),'ch_names':CHANNELS,'times':np.arange(4)/250}
        with tempfile.TemporaryDirectory() as tmp:
            for format in ['direct_dict','scalar_object_array']:
                cases={
                    'shape':{**original,'preprocessed_eeg_data':np.ones((1,1,63,4),dtype=np.float32)},
                    'dtype':{**original,'preprocessed_eeg_data':np.ones(shape,dtype=np.int32)},
                    'finite':{**original,'preprocessed_eeg_data':np.full(shape,np.nan,dtype=np.float32)},
                    'channels':{**original,'ch_names':list(reversed(CHANNELS))},
                    'times':{**original,'times':np.arange(4)/100}}
                for name,data in cases.items():
                    with self.subTest(format=format,validation=name):
                        path=Path(tmp)/'bad.npy';self.write_eeg(path,data,format)
                        with self.assertRaises(ValueError):inspect_eeg(path,shape,check_ram=False)

    def test_dictionary_loader_rejects_other_containers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in [np.ones(2),np.array([{}],dtype=object),np.array('bad',dtype=object),['not a dictionary']]:
                path=Path(tmp)/'bad.npy'
                with open(path,'wb') as f:pickle.dump(value,f)
                with self.assertRaises(ValueError):load_dictionary(path)

    def test_existing_only_mode_cannot_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);download=mock.Mock();data=b'existing';plan=self.plan(data)
            with self.assertRaises(FileNotFoundError):fetch_files(root,plan,download,allow_download=False)
            download.assert_not_called();self.assertFalse((root/'.pilot_downloads').exists())
            for name in FILES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            fetch_files(root,plan,download,inspect=lambda p,s:{'shape':list(s)},allow_download=False)
            download.assert_not_called()

    def plan(self,data):
        return {'revision':'a'*40,'files':{n:{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()} for n in FILES}}

    def test_resume_download_and_skip_existing_without_other_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'complete EEG bytes';plan=self.plan(data);calls=[]
            def download(**kw):
                calls.append(kw['filename'])
                path=Path(kw['local_dir'])/kw['filename'];path.parent.mkdir(parents=True,exist_ok=True)
                partial=path.with_suffix('.npy.incomplete')
                if len(calls)==1:
                    partial.write_bytes(data[:4]);raise ConnectionError('simulated disconnect')
                if partial.exists():self.assertEqual(partial.read_bytes(),data[:4])
                path.write_bytes(data);return str(path)
            check=lambda p,shape:{'shape':list(shape),'finite':True}
            with self.assertRaises(ConnectionError):fetch_files(root,plan,download,check)
            report=fetch_files(root,plan,download,check)
            self.assertEqual(set(report),set(FILES));self.assertTrue(all(v['remote_sha256_verified'] for v in report.values()))
            before=len(calls);fetch_files(root,plan,download,check);self.assertEqual(before,len(calls))
            self.assertTrue(all('/sub-01/' in n for n in calls))
            self.assertTrue(all((root/n).read_bytes()==data for n in FILES))

    def test_existing_corrupt_file_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name=next(iter(FILES));target=root/name;target.parent.mkdir(parents=True)
            target.write_bytes(b'OLD');plan=self.plan(b'NEW')
            download=mock.Mock()
            with self.assertRaises(ValueError):fetch_files(root,plan,download)
            self.assertEqual(target.read_bytes(),b'OLD');download.assert_not_called()
            with self.assertRaises(ValueError):verify_bytes(target,{'size':4,'sha256':None})

    def test_subject_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan(b'abc');plan['files']['Preprocessed_data_250Hz/sub-02/data.npy']={}
            with self.assertRaises(ValueError):fetch_files(tmp,plan,mock.Mock())

    def test_shape_metadata_finiteness_and_ram_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'eeg.npy';shape=(2,1,63,4)
            data={'preprocessed_eeg_data':np.ones(shape,dtype=np.float32),'ch_names':CHANNELS,'times':np.arange(4)/250}
            np.save(p,data)
            r=inspect_eeg(p,shape,check_ram=False)
            self.assertTrue(r['finite'] and r['channels_metadata_verified'] and r['time_metadata_verified'])
            with self.assertRaises(ValueError):inspect_eeg(p,(3,1,63,4),check_ram=False)
            data['preprocessed_eeg_data'][0,0,0,0]=np.nan;np.save(p,data)
            with self.assertRaises(ValueError):inspect_eeg(p,shape,check_ram=False)
            with mock.patch.dict('sys.modules',{'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=0))}):
                with self.assertRaises(MemoryError):inspect_eeg(p,shape)

    def test_inventory_is_read_only_and_not_ready_with_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir();sentinel=cache/'sub-01_test_z16.npy'
            sentinel.write_bytes(b'historical data')
            result=inventory(root,cache)
            self.assertFalse(result['ready_images_and_features']);self.assertTrue(result['missing'])
            self.assertEqual(sentinel.read_bytes(),b'historical data')
            self.assertFalse((root/'Preprocessed_data_250Hz').exists())

    def test_notebook_cpu_phase_precedes_gpu_gate(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/10_rpa_patch_pilot.ipynb'
        cells=json.loads(p.read_text())['cells']
        sources=[''.join(c['source']) for c in cells if c['cell_type']=='code']
        cpu=next(i for i,s in enumerate(sources) if "'--download-sub01'" in s)
        gpu=next(i for i,s in enumerate(sources) if 'APPROVE_GPU_SESSION = False' in s)
        self.assertLess(cpu,gpu)
        self.assertIn("assert not torch.cuda.is_available()",sources[cpu])
        self.assertIn("CUDA_VISIBLE_DEVICES=''",sources[cpu])
        self.assertIn('VERIFY_EXISTING_EEG_ONLY = True',sources[cpu])
        self.assertIn('/content/eeg-patch-pr2-cpu-dictfix',sources[cpu])
        for source in sources:compile(source,'notebook','exec')


if __name__=='__main__':unittest.main()
