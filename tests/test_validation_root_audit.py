import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from phase1.validation_root_audit import prefix_field,wilson,describe_pool,norm_eeg,diagnose
from phase1.core import concept_split,positive_mask,clip_loss,multi_positive_loss,retrieval_metrics,Standardizer

class RootAuditTests(unittest.TestCase):
    def test_training_prefix_does_not_consume_test_values(self):
        prefix='{"train":[{"id":"a","concept":"c","sha256":"h"}]'
        class Guarded(io.StringIO):
            def read(self,n=-1):
                if self.tell()>=len(prefix):raise AssertionError('Attempted to read beyond training prefix')
                return super().read(n)
        with mock.patch('builtins.open',return_value=Guarded(prefix+',"test":SECRET')):
            self.assertEqual(prefix_field('unused',['train'])[0]['id'],'a')
    def test_prefix_nested_mapping_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metadata.json'
            p.write_text('{"catalog":"x", "selection":{"train":[0,2,12],"test":BROKEN}}')
            self.assertEqual(prefix_field(p,['selection','train']),[0,2,12])
            p.write_text('{"test":BROKEN,"train":[]}')
            with self.assertRaises(ValueError):prefix_field(p,['train'])
    def test_image_retrieval_not_concept_classification(self):
        ids=[f'image{i:03}' for i in range(200)];concepts=[i//10 for i in range(200)]
        scores=torch.tensor([[float(a==b) for b in concepts] for a in concepts])
        self.assertEqual(retrieval_metrics(scores,ids,ids)['top1'],10.)
        # Every winning image is in the correct concept, but only 20 exact matches.
        self.assertEqual(sum(concepts[int(j)]==concepts[i] for i,j in enumerate(scores.argmax(1))),200)
    def test_actual_pool_and_normalization_do_not_fit_validation(self):
        records=[dict(id=f'i{i}',concept=f'c{i//10}',sha256=f'h{i}') for i in range(2000)]
        fit,val=concept_split([r['concept'] for r in records],seed=0,fraction=.1)
        r=describe_pool(records,fit,val)
        self.assertEqual((r['fit_images'],r['validation_images'],r['validation_concepts']),(1800,200,20))
        self.assertEqual(r['exact_image_chance_percent'],.5)
        self.assertEqual(r['same_concept_top_image_chance_percent'],5.)
        x=np.random.default_rng(0).normal(size=(2000,3,4)).astype('float32')
        a=norm_eeg(x,fit);x[val]=1e6;b=norm_eeg(x,fit)
        torch.testing.assert_close(a.mean,b.mean);torch.testing.assert_close(a.std,b.std)
    def test_unique_stimuli_have_no_duplicate_false_negatives(self):
        torch.manual_seed(0);a=torch.randn(8,5,requires_grad=True);b=torch.randn(8,5,requires_grad=True)
        ids=list(range(8))
        torch.testing.assert_close(clip_loss(a,b,10.),multi_positive_loss(a,b,10.,ids,ids))
        # Same concept is not the same stimulus: both remain distinct retrieval answers.
        self.assertEqual(int(positive_mask(['c_img1','c_img2'],['c_img1','c_img2']).sum()),2)
    def test_mean_affine_commutes_and_scale_can_lower_loss_without_new_rank(self):
        torch.manual_seed(0);x=torch.randn(3,7,5);norm=Standardizer(torch.randn(1,5),torch.rand(1,5)+.1)
        torch.testing.assert_close(((x-norm.mean[:,None,:])/norm.std[:,None,:]).mean(1),norm.transform(x.mean(1)))
        a=torch.eye(8)
        self.assertLess(float(clip_loss(a,a,20.)),float(clip_loss(a,a,2.)))
        self.assertEqual(retrieval_metrics(a,list(range(8)),list(range(8)),8),retrieval_metrics(20*a,list(range(8)),list(range(8)),8))
    def test_uncertainty_and_effective_batch_counts(self):
        np.testing.assert_allclose(wilson(17,200),[5.374575,13.189587],atol=1e-5)
        np.testing.assert_allclose(wilson(18,200),[5.768745,13.776572],atol=1e-5)
        batches=np.array_split(np.arange(1800),math.ceil(1800/64))
        self.assertEqual(sorted(map(len,batches)),[62]*27+[63]*2)
        self.assertEqual(len(batches)*5,145)

    def test_stats_audit_opens_training_arrays_only_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training';training.mkdir()
            cfg=dict(setting='within_subject',seeds=[0],split_seed=0,val_fraction=.1,objective='diagonal',bottleneck=256,groups=32)
            (training/'provenance.json').write_text('{"config":'+json.dumps(cfg)+',"image_ids":UNREADABLE}')
            records=[dict(id=f'i{i}',concept=f'c{i}',sha256=f'h{i}') for i in range(20)]
            (root/'catalog.json').write_text('{"train":'+json.dumps(records)+',"test":UNREADABLE}')
            (root/'dataset.json').write_text('{"selection":{"train":'+json.dumps(list(range(20)))+',"test":UNREADABLE}}')
            pooled=np.random.default_rng(2).normal(size=(20,3200)).astype('float16')
            eeg=np.random.default_rng(3).normal(size=(20,17,250)).astype('float32')
            def load(path,*args,**kwargs):
                path=Path(path)
                if path.name=='sub-01_train_prez.npy':return eeg
                if path.name in [f'train_L{i}_mean.npy' for i in [20,24,28,32,36]]:return pooled.astype('float32')
                if path.name=='train_done.npy':return np.ones(20,dtype=bool)
                if path.name=='train.npy':return np.broadcast_to(pooled[:,None,:],(20,1024,3200))
                raise AssertionError('Forbidden array access: '+str(path))
            for arm in ['baseline','mean','patch']:
                torch.save({'epoch':5,'history':[{'epoch':i,'loss':1.,'source_val_top1':50.} for i in range(1,6)],
                    'best_epoch':1,'best':50.,'best_state':{'log_scale':torch.tensor(math.log(1/.07))}},
                    training/f'within_subject_sub-01_s0_{arm}_diagonal_latest.pt')
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with mock.patch('phase1.validation_root_audit.np.load',side_effect=load):
                report=diagnose(root,training)
            self.assertEqual(report['pool']['validation_images'],2)
            self.assertFalse(report['original_EEG_identity_verified'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
