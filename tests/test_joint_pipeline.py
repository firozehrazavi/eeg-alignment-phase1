import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from phase1.joint_smoke import fixture
from phase1.joint_data import load_training, split_three, derangement
from phase1.joint_experiment import evaluate, run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.manifest,self.config=fixture(self.root/'data')

    def test_train_loader_never_opens_unused_subject_or_test(self):
        m=json.loads(self.manifest.read_text())
        m['subjects']['sub-02']['path']='PROTECTED_OFFICIAL_TEST.npy'
        self.manifest.write_text(json.dumps(m))
        loaded=load_training(self.manifest,['sub-01'])
        self.assertEqual(set(loaded[3]),{'sub-01'})

    def test_split_disjoint_and_derangement(self):
        c=[f'c{i//10}' for i in range(16540)]
        parts=split_three(c)
        for i in range(3):
            for j in range(i):
                self.assertFalse({c[k] for k in parts[i]} & {c[k] for k in parts[j]})
        p=derangement(c,3).tolist()
        self.assertEqual(sorted(p),list(range(len(c))))
        self.assertTrue(all(c[i]!=c[j] for i,j in enumerate(p)))

    def test_hash_and_provenance_fail_closed(self):
        m=json.loads(self.manifest.read_text()); m['text']['sha256']='wrong'
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'SHA256'): load_training(self.manifest,['sub-01'])
        m['partition']='official_test'; self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'official_training'): load_training(self.manifest,['sub-01'])

    def test_full_four_arm_cpu_pipeline(self):
        cfg=json.loads(self.config.read_text()); cfg['epochs']=1
        self.config.write_text(json.dumps(cfg))
        rows=run(self.manifest,self.config,self.root/'run')
        self.assertEqual([r['arm'] for r in rows],list('ABCD'))
        self.assertEqual(len({r['parameters'] for r in rows}),1)
        self.assertTrue(all(r['candidates']==4 for r in rows))
        saved=json.loads((self.root/'run/provenance.json').read_text())
        self.assertFalse(saved['official_test_opened'])
        self.assertEqual(saved['status'],'complete_development_only')
        with self.assertRaises(FileExistsError): run(self.manifest,self.config,self.root/'run')

    def test_interrupted_pipeline_resumes_from_matching_epoch_checkpoint(self):
        cfg=json.loads(self.config.read_text()); cfg['epochs']=2
        self.config.write_text(json.dumps(cfg))
        output=self.root/'resumable'
        reference=run(self.manifest,self.config,self.root/'reference')
        original_evaluate=evaluate
        calls={'count':0}
        def interrupt_after_one_epoch(*args, **kwargs):
            calls['count']+=1
            if calls['count']==2:
                raise RuntimeError('injected interruption')
            return original_evaluate(*args, **kwargs)
        with patch('phase1.joint_experiment.evaluate', side_effect=interrupt_after_one_epoch):
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                run(self.manifest,self.config,output)
        latest=torch.load(output/'sub-01_s0_A_latest.pt',map_location='cpu',weights_only=True)
        self.assertEqual(latest['epoch'],1)
        rows=run(self.manifest,self.config,output,resume=True)
        self.assertEqual([r['arm'] for r in rows],list('ABCD'))
        self.assertEqual([(r['arm'],float(r['image_top1']),float(r['text_top1'])) for r in reference],
                         [(r['arm'],float(r['image_top1']),float(r['text_top1'])) for r in rows])
        for arm in 'ABCD':
            expected=torch.load(self.root/f'reference/sub-01_s0_{arm}_holdout_scores.pt',
                                map_location='cpu',weights_only=True)
            actual=torch.load(output/f'sub-01_s0_{arm}_holdout_scores.pt',
                              map_location='cpu',weights_only=True)
            torch.testing.assert_close(actual['image_scores'],expected['image_scores'],rtol=0,atol=0)
            torch.testing.assert_close(actual['text_scores'],expected['text_scores'],rtol=0,atol=0)
        saved=json.loads((output/'provenance.json').read_text())
        self.assertEqual(saved['status'],'complete_development_only')
        log=[json.loads(line) for line in (output/'epochs.jsonl').read_text().splitlines()]
        self.assertEqual([(r['run'],r['epoch']) for r in log if r['run']=='sub-01_s0_A'],
                         [('sub-01_s0_A',1),('sub-01_s0_A',2)])
        self.assertEqual(len(run(self.manifest,self.config,output,resume=True)),4)
        cfg['epochs']=3
        self.config.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            run(self.manifest,self.config,output,resume=True)


if __name__=='__main__': unittest.main()
