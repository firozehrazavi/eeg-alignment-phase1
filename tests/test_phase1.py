import ast
import json
import math
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
from phase1.core import (clip_loss, multi_positive_loss, retrieval_metrics,
    positive_mask, concept_split, validate_catalog, AccessPolicy,
    select_source_config, Standardizer, csls, validate_disjoint_catalogs)
from phase1.semantic import SemanticHead


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_unique_matches_baseline_loss_and_gradient(self):
        a = torch.randn(7, 5, requires_grad=True)
        b = torch.randn(7, 5, requires_grad=True)
        old = clip_loss(a, b, 3.)
        new = multi_positive_loss(a, b, 3., list(range(7)), list(range(7)))
        torch.testing.assert_close(old, new)
        for x, y in zip(torch.autograd.grad(old, (a, b)), torch.autograd.grad(new, (a, b))):
            torch.testing.assert_close(x, y)

    def test_duplicate_identity_removes_false_negative_penalty(self):
        a = torch.ones(4, 3, requires_grad=True)
        old = clip_loss(a, a, 2.)
        new = multi_positive_loss(a, a, 2., ['a']*4, ['a']*4)
        self.assertAlmostEqual(old.item(), math.log(4), places=6)
        self.assertAlmostEqual(new.item(), 0., places=6)
        new.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_rectangular_reordered_positive_indexing(self):
        a = torch.eye(3)[[0, 1, 0, 2]]
        b = torch.eye(3)[[2, 0, 1]]
        loss = multi_positive_loss(a, b, 20., ['a','b','a','c'], ['c','a','b'])
        self.assertLess(loss.item(), 1e-6)
        p = positive_mask(['a','b','a'], ['b','a'])
        self.assertEqual(p.tolist(), [[False,True],[True,False],[False,True]])

    def test_missing_positive_rejected(self):
        with self.assertRaises(ValueError):
            multi_positive_loss(torch.eye(2), torch.eye(2), 1, ['a','b'], ['a','c'])

    def test_permutation_invariant(self):
        a, b = torch.randn(4, 7), torch.randn(4, 7)
        ids = ['a','b','a','c']; p = [2,0,3,1]
        x = multi_positive_loss(a,b,2,ids,ids)
        y = multi_positive_loss(a,b[p],2,ids,[ids[i] for i in p])
        torch.testing.assert_close(x,y)

    def test_baseline_architecture_and_loss_are_unchanged(self):
        nb = json.loads((Path(__file__).parents[1]/'notebooks/06b_internvit_sota.ipynb').read_text(encoding='utf8'))
        original = ast.parse(''.join(nb['cells'][5]['source']))
        import phase1.models as models
        import inspect
        for name in ['EEGProject', 'ImageSide', 'Aligner']:
            node = next(n for n in original.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(node), ast.dump(ast.parse(inspect.getsource(getattr(models,name))).body[0]))


class MetricTests(unittest.TestCase):
    def test_known_top1_top5(self):
        ids = [str(i) for i in range(200)]
        s = torch.zeros(3,200)
        s[0,0]=2
        s[1,0]=3; s[1,1]=2
        s[2,:6]=torch.tensor([9.,8.,1.,7.,6.,5.])
        m = retrieval_metrics(s,['0','1','2'],ids)
        self.assertAlmostEqual(m['top1'],100/3)
        self.assertAlmostEqual(m['top5'],200/3)

    def test_ties_are_not_all_correct_and_permutation_safe(self):
        ids = ['a','b','c','d','e','f']
        s = torch.zeros(6,6)
        m = retrieval_metrics(s,ids,ids,6)
        self.assertAlmostEqual(m['top1'],100/6)
        self.assertAlmostEqual(m['top5'],500/6)
        self.assertEqual(m,retrieval_metrics(s[:,[5,4,3,2,1,0]],ids,ids[::-1],6))

    def test_invalid_gallery(self):
        for ids in [['a','a'],['b','c']]:
            with self.assertRaises(ValueError):
                retrieval_metrics(torch.ones(1,2),['a'],ids,2)
        with self.assertRaises(ValueError):
            retrieval_metrics(torch.tensor([[float('nan')]]),['a'],['a'],1)
        with self.assertRaises(ValueError):
            retrieval_metrics(torch.eye(2),['a','b'],['a','b'])


class ProtocolTests(unittest.TestCase):
    def test_concept_partition(self):
        concepts = ['dog','cat','dog','cup','cat','cup','car']
        fit,val = concept_split(concepts,3,.3)
        self.assertFalse({concepts[i] for i in fit} & {concepts[i] for i in val})
        self.assertEqual(sorted(fit+val),list(range(len(concepts))))
        self.assertEqual((fit,val),concept_split(concepts,3,.3))

    def test_duplicate_content_detected(self):
        with self.assertRaises(ValueError):
            validate_catalog(['a','b'],['dog','cat'],['same','same'])

    def test_test_concepts_and_content_cannot_overlap(self):
        train=dict(image_ids=['a'],concept_ids=['dog'],content_hashes=['h1'])
        for test in [dict(image_ids=['b'],concept_ids=['dog'],content_hashes=['h2']),
                     dict(image_ids=['b'],concept_ids=['cat'],content_hashes=['h1'])]:
            with self.assertRaises(ValueError): validate_disjoint_catalogs(train,test)

    def test_source_selection(self):
        rows = [dict(config=c,subject=s,score=v,role='source_validation')
                for c,v in [('a',4),('b',5)] for s in ['s1','s2']]
        self.assertEqual(select_source_config(rows,['s1','s2'],'target'),'b')
        for bad in [rows+[dict(config='b',subject='target',score=99,role='source_validation')], rows[:-1], rows+rows[:1]]:
            with self.assertRaises(ValueError):
                select_source_config(bad,['s1','s2'],'target')

    def test_normalization_independent_of_other_queries(self):
        normalizer = Standardizer.fit(torch.arange(24.).reshape(4,2,3),True)
        q = torch.randn(1,2,3)
        before = normalizer.transform(q)
        after = normalizer.transform(torch.cat([q,torch.ones_like(q)*1e9]))[:1]
        torch.testing.assert_close(before,after)
        self.assertTrue(torch.isfinite(Standardizer.fit(torch.ones(3,2)).transform(torch.ones(1,2))).all())

    def test_policy_rejects_hidden_transduction(self):
        for p in [AccessPolicy('loso_inductive',test_query_statistics=True),
                  AccessPolicy('labeled_calibration',0), AccessPolicy('loso_transductive',2),
                  AccessPolicy('loso_inductive',target_session_statistics=True)]:
            with self.assertRaises(ValueError): p.validate()
        with self.assertRaises(ValueError): csls(torch.eye(4),AccessPolicy('loso_inductive'))
        p=AccessPolicy('loso_transductive',test_query_statistics=True)
        self.assertEqual(csls(torch.eye(4),p).shape,(4,4))


class SemanticTests(unittest.TestCase):
    def test_frozen_inputs_and_finite_head_gradients(self):
        head=SemanticHead(5,6,7,4)
        e=torch.randn(3,5,requires_grad=True)
        t=torch.randn(3,6,requires_grad=True)
        v=torch.randn(3,7,requires_grad=True)
        pos=torch.eye(3,dtype=torch.bool)
        loss=head(e,t,v,pos,~pos,adaptive=False)
        loss.backward()
        self.assertIsNone(t.grad); self.assertIsNone(v.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_zero_weight_and_ambiguity(self):
        head=SemanticHead(5,6,7,4)
        args=(torch.randn(3,5),torch.randn(3,6),torch.randn(3,7),torch.eye(3,dtype=torch.bool))
        zero=torch.zeros(3,3,dtype=torch.bool)
        self.assertEqual(head(*args,zero,weight=0).item(),0)
        self.assertLessEqual(head(*args,~args[-1],adaptive=False).item(),head(*args,zero,adaptive=False).item())


if __name__ == '__main__':
    unittest.main()
