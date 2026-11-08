import copy
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import torch
from phase1.patch_adapter import ResidualPatchAdapter, PatchAligner
from phase1.patch_cache import TokenCache, digest
from phase1.patch_experiment import enforce_provenance, fit_eeg_norm, schedule, run
from phase1.patch_prepare import prepare, POST17
from phase1.patch_resources import estimate_cu, retry_oom, BudgetMeter, BudgetExceeded


class PatchTests(unittest.TestCase):
    def test_budget_and_oom_safeguards(self):
        self.assertEqual(estimate_cu(7200,7.5),15.)
        with self.assertRaises(ValueError):estimate_cu(1,0)
        attempts=[]
        def operation(chunk):
            attempts.append(chunk)
            if chunk>2:raise torch.cuda.OutOfMemoryError('synthetic OOM')
            return 123
        self.assertEqual(retry_oom(operation,8),(123,2))
        self.assertEqual(attempts,[8,4,2])
        with mock.patch.dict('os.environ',{'PHASE1_SESSION_START':'0','PHASE1_CU_RATE':'10','PHASE1_CU_LIMIT':'1',
                                          'PHASE1_RESOURCE_LOG':''}):
            with self.assertRaises(BudgetExceeded):BudgetMeter().check('synthetic_budget')

    def test_residual_formula_and_gradient(self):
        torch.manual_seed(0)
        m=ResidualPatchAdapter(32,(2,2),16,32,8)
        t=torch.randn(3,4,32,requires_grad=True)
        x=t.transpose(1,2).reshape(3,32,2,2)
        expected=m.projection((x+m.branch(x)).mean((2,3)))
        torch.testing.assert_close(m(t),expected)
        m(t).square().mean().backward()
        self.assertGreater(float(t.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None for p in m.parameters()))
        with self.assertRaises(ValueError):m(torch.randn(3,5,32))

    def test_zero_branch_is_mean_control(self):
        m=ResidualPatchAdapter(32,(2,2),16,32,8)
        for p in m.branch.parameters():p.data.zero_()
        x=torch.randn(3,4,32)
        torch.testing.assert_close(m(x),m.projection(x.mean(1)))

    def test_checkpoint_chunking_preserves_gradients(self):
        a=PatchAligner(12,[32]*5,32,(2,2),'patch',bottleneck=32,groups=8)
        b=copy.deepcopy(a); x=torch.randn(5,4,32)
        y=a.visual(tokens=x,chunk=2); z=b.visual(tokens=x,chunk=5)
        torch.testing.assert_close(y,z)
        y.square().mean().backward();z.square().mean().backward()
        for p,q in zip(a.img.parameters(),b.img.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)

    def test_matched_eeg_initialization(self):
        states=[]
        for arm in ['baseline','mean','patch']:
            torch.manual_seed(7)
            m=PatchAligner(12,[32]*5,32,(2,2),arm,bottleneck=32,groups=8)
            states.append(m.eeg.state_dict())
        for k in states[0]:
            torch.testing.assert_close(states[0][k],states[1][k])
            torch.testing.assert_close(states[0][k],states[2][k])

    def test_source_moments_and_strict_guard(self):
        x=np.arange(48,dtype=np.float32).reshape(4,3,4)
        norm=fit_eeg_norm([x], [0,1])
        torch.testing.assert_close(norm.mean,torch.tensor(x[:2].mean((0,2),keepdims=True)))
        cfg={'regime':'strict','objective':'diagonal'}
        with self.assertRaises(ValueError):
            enforce_provenance({'preprocessing':{'regime':'benchmark_preprocessed'}},cfg)
        enforce_provenance({'preprocessing':{'regime':'strict','upstream_target_independent':True,
                                            'audit_evidence':'test fixture'}},cfg)

    def test_ten_fold_schedule(self):
        subjects=[f'sub-{i:02d}' for i in range(1,11)]
        plan=schedule({'setting':'loso','seeds':[0,1,2]},subjects)
        self.assertEqual(len(plan),30)
        for target,sources,seed in plan:
            self.assertEqual(len(sources),9);self.assertNotIn(target,sources)

    def test_prepare_authoritative_rows_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw';cache=root/'oldcache';cache.mkdir()
            (raw/'images_set').mkdir(parents=True)
            eegdir=raw/'Preprocessed_data_250Hz/sub-01';eegdir.mkdir(parents=True)
            meta={};old={};features={'tr':{},'te':{}}
            for s,n in [('train',20),('test',200)]:
                files=[f'{s}_{i}.jpg' for i in range(n)]
                meta[f'{s}_img_files']=files
                meta[f'{s}_img_concepts']=[f'{s}_c{i//2 if s=="train" else i}' for i in range(n)]
                old[f'{s}_files']=files;old[f'{s}_rel']=[f'images_set/{p}' for p in files]
                for name in files:(raw/'images_set'/name).write_bytes(name.encode())
                for l in [20,24,28,32,36]:features['tr' if s=='train' else 'te'][(l,'mean')]=torch.ones(n,32)*l
                x=np.arange(n*2*63*4,dtype=np.float32).reshape(n,2,63,4)
                np.save(eegdir/f'preprocessed_eeg_{"training" if s=="train" else "test"}.npy',{'preprocessed_eeg_data':x})
            np.save(raw/'images_set/image_metadata.npy',meta)
            torch.save(old,cache/'meta_v6.pt');torch.save(features,cache/'feat6_internvit6b.pt')
            before=digest(cache/'feat6_internvit6b.pt')
            prepare(raw,cache,root/'new',['sub-01'],concepts=3)
            manifest=json.loads((root/'new/dataset.json').read_text())
            x=np.load(eegdir/'preprocessed_eeg_training.npy',allow_pickle=True).item()['preprocessed_eeg_data']
            expected=x[manifest['selection']['train']].mean(1)[:,POST17]
            np.testing.assert_array_equal(np.load(root/'new/sub-01_train_prez.npy'),expected)
            self.assertEqual(before,digest(cache/'feat6_internvit6b.pt'))
            with self.assertRaises(FileExistsError):prepare(raw,cache,root/'new',['sub-01'])
            prepared_hash=digest(root/'new/dataset.json')
            prepare(raw,cache,root/'new',['sub-01'],concepts=3,resume=True)
            self.assertEqual(prepared_hash,digest(root/'new/dataset.json'))
            with mock.patch('phase1.patch_prepare.np.lib.format.open_memmap',side_effect=RuntimeError('simulated interruption')):
                with self.assertRaises(RuntimeError):prepare(raw,cache,root/'partial',['sub-01'],concepts=3)
            prepare(raw,cache,root/'partial',['sub-01'],concepts=3,resume=True)
            self.assertEqual(prepared_hash,digest(root/'partial/dataset.json'))
            with self.assertRaises(ValueError):prepare(raw,cache,root/'partial',['sub-01'],concepts=4,resume=True)

    def test_end_to_end_synthetic_pilot_and_cache_guards(self):
        # This fixture tests execution/identity contracts, NOT EEG performance.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'tokens';cache.mkdir()
            rng=np.random.default_rng(12)
            cat={s:[dict(id=f'{s}-{i}',concept=f'{s}-c{i//2 if s=="train" else i}',
                         sha256=f'{s}-hash-{i}',path='unused') for i in range(n)]
                 for s,n in [('train',20),('test',200)]}
            cp=root/'catalog.json';cp.write_text(json.dumps(cat))
            manifest=dict(catalog='catalog.json',tokens='tokens',subjects={},pooled={},
                          preprocessing={'regime':'benchmark_preprocessed'})
            shapes={}
            for s,n in [('train',20),('test',200)]:
                t=rng.normal(size=(n,4,32)).astype('float16');np.save(cache/f'{s}.npy',t)
                np.save(cache/f'{s}_done.npy',np.ones(n,dtype=bool));shapes[s]=list(t.shape)
                manifest['pooled'][s]=[]
                for l in range(5):
                    p=f'{s}_{l}.npy';np.save(root/p,t.astype('float32').mean(1));manifest['pooled'][s].append(p)
                for sub in ['sub-01','sub-02']:
                    p=f'{sub}_{s}.npy';np.save(root/p,rng.normal(size=(n,3,4)).astype('float32'))
                    manifest['subjects'].setdefault(sub,{})[s]=p
            (cache/'manifest.json').write_text(json.dumps(dict(catalog_sha256=digest(cp),shapes=shapes,grid=[2,2],width=32,layer=28)))
            dp=root/'dataset.json';dp.write_text(json.dumps(manifest))
            cfg=dict(setting='loso',targets=['sub-02'],regime='benchmark_preprocessed',seeds=[0],split_seed=0,
                     arms=['baseline','mean','patch'],objective='diagonal',epochs=2,batch=8,lr=1e-4,
                     weight_decay=1e-4,adapter_chunk=2,bottleneck=32,groups=8,patch_layer=28,device='cpu',report_csls=True)
            config=root/'config.json';config.write_text(json.dumps(cfg))
            # A held-out subject's training file must never be needed.
            (root/'sub-02_train.npy').unlink()
            results=run(dp,config,root/'run1')
            repeated=run(dp,config,root/'run2')
            self.assertEqual(results,repeated)
            # Profile uses the final scheduler, writes epoch state, and never tests.
            self.assertEqual(run(dp,config,root/'profile',profile_one_epoch=True),[])
            self.assertFalse((root/'profile/COMPLETE').exists())
            self.assertFalse((root/'profile/results.json').exists())
            self.assertEqual(results,run(dp,config,root/'profile',resume=True))
            self.assertEqual(results,run(dp,config,root/'run1',resume=True))
            conv={**cfg,'encoder':'conv','arms':['baseline'],'objective':'multi_positive',
                  'epochs':1,'save_validation_scores':True,'text_weight':0.}
            convpath=root/'conv.json';convpath.write_text(json.dumps(conv))
            conv_result=run(dp,convpath,root/'conv')
            self.assertEqual(len(conv_result),2)
            saved=torch.load(root/'conv/loso_sub-02_s0_baseline_multi_positive_source_validation.pt',weights_only=False)
            self.assertEqual(set(saved),{'sub-01'})
            self.assertEqual(saved['sub-01']['role'],'source_validation')
            self.assertEqual(len(results),6)
            self.assertTrue((root/'run1/COMPLETE').is_file())
            self.assertTrue(all('inductive' not in r['setting'] for r in results))
            with self.assertRaises(FileExistsError):run(dp,config,root/'run1')
            done=np.ones(20,dtype=bool);done[0]=False;np.save(cache/'train_done.npy',done)
            with self.assertRaises(ValueError):TokenCache(cache,cp)
            np.save(cache/'train_done.npy',np.ones(20,dtype=bool))
            cp.write_text(json.dumps(cat)+'\n')
            with self.assertRaises(ValueError):TokenCache(cache,cp)


if __name__=='__main__':unittest.main()
