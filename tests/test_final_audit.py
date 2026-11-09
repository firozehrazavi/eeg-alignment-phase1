import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from phase1.final_audit import audit,evaluate,metrics,selected

class FinalAuditTests(unittest.TestCase):
    def test_approval_precedes_all_reads(self):
        with mock.patch('phase1.final_audit.audit') as read:
            with self.assertRaises(PermissionError):evaluate('missing','missing','missing')
            read.assert_not_called()
    def test_metrics_and_tie_selection(self):
        ids=[str(i) for i in range(200)]
        r=metrics(torch.eye(200),ids)
        self.assertEqual(r,{'top1':100.,'top5':100.,'mrr':1.})
        h=[dict(epoch=i,loss=1.,source_val_top1=4.) for i in range(1,6)]
        self.assertEqual(selected(h)['epoch'],1)
        with self.assertRaises(ValueError):selected(h[:-1])
    def test_audit_fifteen_records_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'TRAINING_COMPLETE').write_text('done')
            p={'config':{'setting':'within_subject','seeds':[0],'epochs':5,'arms':['baseline','mean','patch'],'objective':'diagonal'},'split':{'fit':[0,1],'validation':[2,3]}}
            (root/'provenance.json').write_text(json.dumps(p))
            h=[dict(epoch=i,loss=1/i,source_val_top1=float(i),seconds=1.,peak_gpu_allocated_bytes=100) for i in range(1,6)]
            for arm in p['config']['arms']:
                torch.save(dict(epoch=5,history=h,best_epoch=5,best=5.,best_state={'w':torch.ones(1)}),root/f'within_subject_sub-01_s0_{arm}_diagonal_latest.pt')
            before={f.name:f.read_bytes() for f in root.iterdir()}
            result=audit(root);self.assertEqual(result['verified_epoch_records'],15)
            self.assertEqual(result['chance_top1_percent'],50.)
            self.assertEqual(before,{f.name:f.read_bytes() for f in root.iterdir()})

    def test_frozen_evaluation_idempotent_synthetic(self):
        import numpy as np
        from phase1.core import concept_split
        from phase1.patch_cache import digest
        class FakeAligner(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();self.arm=args[4];self.dummy=torch.nn.Parameter(torch.ones(1))
            def eeg(self,x):return x.flatten(1)[:,:32]*self.dummy
            def visual(self,pooled=None,tokens=None,chunk=1):
                return (pooled[0] if pooled is not None else tokens.mean(1))*self.dummy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tr=root/'training';tr.mkdir();tok=root/'tokens';tok.mkdir()
            rng=np.random.default_rng(5);cat={};m={'catalog':'catalog.json','tokens':'tokens','pooled':{},'subjects':{'sub-01':{}}}
            shapes={}
            for split,n in [('train',20),('test',200)]:
                cat[split]=[dict(id=f'{split}{i}',concept=f'{split}c{i}',sha256=f'h{split}{i}') for i in range(n)]
                x=rng.normal(size=(n,4,32)).astype('float16');np.save(tok/f'{split}.npy',x)
                np.save(tok/f'{split}_done.npy',np.ones(n,dtype=bool));shapes[split]=list(x.shape)
                m['pooled'][split]=[]
                for layer in range(5):
                    name=f'{split}_{layer}.npy';np.save(root/name,x.mean(1).astype('float32'));m['pooled'][split].append(name)
                name=f'{split}_eeg.npy';np.save(root/name,rng.normal(size=(n,17,250)).astype('float32'));m['subjects']['sub-01'][split]=name
            (root/'catalog.json').write_text(json.dumps(cat));(root/'dataset.json').write_text(json.dumps(m))
            tm={'catalog_sha256':digest(root/'catalog.json'),'shapes':shapes,'grid':[2,2],'width':32,'layer':28}
            (tok/'manifest.json').write_text(json.dumps(tm))
            cfg={'setting':'within_subject','seeds':[0],'epochs':5,'arms':['baseline','mean','patch'],'objective':'diagonal','split_seed':0,'val_fraction':.1,'bottleneck':32,'groups':8}
            fit,val=concept_split([r['concept'] for r in cat['train']],0,.1)
            prov={'config':cfg,'split':{'fit':fit,'validation':val},'dataset_sha256':digest(root/'dataset.json'),'catalog_sha256':digest(root/'catalog.json'),'token_manifest':tm,'input_hashes':{}}
            (tr/'provenance.json').write_text(json.dumps(prov));(tr/'TRAINING_COMPLETE').write_text('done')
            h=[dict(epoch=i,loss=1.,source_val_top1=float(i)) for i in range(1,6)]
            for arm in cfg['arms']:
                torch.save(dict(epoch=5,history=h,best_epoch=5,best=5.,best_state={'dummy':torch.ones(1)}),tr/f'within_subject_sub-01_s0_{arm}_diagonal_latest.pt')
            before={p.name:digest(p) for p in tr.iterdir()}
            with mock.patch('phase1.final_audit.PatchAligner',FakeAligner):
                result=evaluate(root/'dataset.json',tr,root/'out',approve=True)
                self.assertEqual(set(result),set(cfg['arms']))
                self.assertEqual(result,evaluate(root/'dataset.json',tr,root/'out',approve=True))
            self.assertEqual(before,{p.name:digest(p) for p in tr.iterdir()})
            self.assertTrue((root/'out/COMPLETE').is_file())

    def test_notebook_gate_is_first_statement(self):
        import ast
        root=Path(__file__).resolve().parents[1]/'notebooks'
        n=json.loads((root/'14_gated_final_test.ipynb').read_text())
        sources=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        self.assertNotIn('--approve-final-test',sources[0])
        code=ast.parse(sources[1]);first=ast.Module(body=code.body[:2],type_ignores=[])
        with self.assertRaises(AssertionError):exec(compile(first,'gate','exec'),{})
        for name in ['13_readonly_training_audit.ipynb','14_gated_final_test.ipynb']:
            for c in json.loads((root/name).read_text())['cells']:
                if c['cell_type']=='code':compile(''.join(c['source']),name,'exec')
