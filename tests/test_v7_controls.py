import copy
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from phase1.core import multi_positive_loss, retrieval_metrics
from phase1.patch_experiment import enforce_provenance
from phase1.v7_controls import (ConvEEG,source_training_rows,stratified_rows,augment_eeg,
    margin_multi_positive_loss,margin_at_epoch,select_score_ensemble,apply_score_ensemble)


class V7Tests(unittest.TestCase):
    def test_global_stratified_rows_exclude_validation(self):
        fit=[0,2,4,6];val=[1,3,5,7,8,9]
        rows=source_training_rows(fit,val,10,['s1','s2'],'s3')
        batches=stratified_rows(rows,np.repeat([0,1],10),4,np.random.default_rng(3))
        joined=np.concatenate(batches)
        self.assertEqual(set(joined),set(rows));self.assertEqual(len(joined),len(rows))
        self.assertTrue(all(int(r)%10 in fit for r in joined))
        self.assertTrue(all(len(b)<=4 for b in batches))
        self.assertTrue(all(len(set((b//10).tolist()))==2 for b in batches))
        with self.assertRaises(ValueError):source_training_rows(fit,[2],10,['s1'])
        with self.assertRaises(ValueError):source_training_rows(fit,val,10,['s1'],'s1')

    def test_conv_shape_gradient_and_inductive_batch_independence(self):
        torch.manual_seed(2);torch.set_num_threads(1)
        m=ConvEEG(3,16,embedding=32)
        x=torch.randn(4,3,16);before=x.clone()
        y=m(x);self.assertEqual(y.shape,(4,32));y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        m.eval()
        torch.testing.assert_close(m(x[:1]),m(x)[:1],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(x,before)
        with self.assertRaises(ValueError):m(torch.randn(4,2,16))

    def test_augmentation_is_explicit_and_does_not_mutate(self):
        x=torch.ones(3,4,20);state=torch.get_rng_state()
        self.assertIs(augment_eeg(x),x);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        y=augment_eeg(x,cut_fraction=.2)
        self.assertTrue(((y==0).sum(-1)==4).all());self.assertTrue((x==1).all())
        torch.manual_seed(2);a=augment_eeg(x,noise_std=.2)
        torch.manual_seed(2);b=augment_eeg(x,noise_std=.2)
        torch.testing.assert_close(a,b)
        with self.assertRaises(ValueError):augment_eeg(x,channel_drop=1.)

    def test_margin_all_positives_and_permutation(self):
        torch.manual_seed(0);z=torch.randn(4,8,requires_grad=True);v=torch.randn(4,8,requires_grad=True)
        ids=['a','b','a','c'];perm=[2,0,3,1]
        torch.testing.assert_close(margin_multi_positive_loss(z,v,3.,ids,ids,0),multi_positive_loss(z,v,3.,ids,ids))
        a=margin_multi_positive_loss(z,v,3.,ids,ids,.15)
        b=margin_multi_positive_loss(z,v[perm],3.,ids,[ids[i] for i in perm],.15)
        torch.testing.assert_close(a,b);a.backward();self.assertTrue(torch.isfinite(z.grad).all())
        cfg={'objective':'multi_positive','epochs':5,'margin_stage':{'start_epoch':4,'margin':.15}}
        self.assertEqual(margin_at_epoch(cfg,3),0.);self.assertEqual(margin_at_epoch(cfg,4),.15)

    def test_text_cannot_switch_on_from_configuration(self):
        m={'preprocessing':{'regime':'benchmark_preprocessed'}}
        c={'regime':'benchmark_preprocessed','objective':'multi_positive','text_weight':0}
        enforce_provenance(m,c)
        with self.assertRaises(ValueError):enforce_provenance(m,{**c,'text_weight':.3})

    def test_ensemble_uses_only_complete_source_validation_and_ids(self):
        ids=['a','b','c','d']
        def bundle(scores):return dict(role='source_validation',score_type='cosine',scores=scores,query_ids=ids,gallery_ids=ids)
        validation={m:{s:bundle(torch.eye(4)*(1 if m=='good' else -1)) for s in ['s1','s2']} for m in ['good','bad']}
        candidates=[{'good':0.,'bad':1.},{'good':1.,'bad':0.},{'good':.5,'bad':.5}]
        selection=select_score_ensemble(validation,candidates,['s1','s2'],'s3')
        self.assertEqual(selection['weights'],{'good':1.,'bad':0.})
        test={m:copy.deepcopy(validation[m]['s1']) for m in validation}
        order=[2,0,3,1]
        for b in test.values():
            b['role']='test';b['scores']=b['scores'][:,order];b['gallery_ids']=[ids[i] for i in order]
        scores,q,g=apply_score_ensemble(selection,test)
        self.assertEqual(retrieval_metrics(scores,q,g,4)['top1'],100.)
        bad=copy.deepcopy(validation);bad['good']['s1']['role']='test'
        with self.assertRaises(ValueError):select_score_ensemble(bad,candidates,['s1','s2'],'s3')
        with self.assertRaises(ValueError):select_score_ensemble(validation,candidates,['s1','s2'],'s2')
        with self.assertRaises(ValueError):apply_score_ensemble(selection,{'good':test['good']})

    def test_ablation_configs_change_one_factor(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        baseline=json.loads((root/'v7_A0_corrected.json').read_text())
        for name,key in [('v7_A1_conv_only','encoder'),('v7_B1_noise_only','augmentation'),('v7_C1_margin_only','margin_stage')]:
            cfg=json.loads((root/f'{name}.json').read_text())
            changed={k for k in set(cfg)|set(baseline) if cfg.get(k)!=baseline.get(k)}
            self.assertEqual(changed,{key})


if __name__=='__main__':unittest.main()
