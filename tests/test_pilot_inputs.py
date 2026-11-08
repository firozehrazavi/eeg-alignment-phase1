import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from phase1.pilot_inputs import FILES,CHANNELS,fetch_files,inspect_eeg,inventory,verify_bytes


class PilotInputTests(unittest.TestCase):
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
        for source in sources:compile(source,'notebook','exec')


if __name__=='__main__':unittest.main()
