"""Regression checks for real duplicate captions in THINGS-EEG2 development."""
import unittest
import torch
from phase1.joint_experiment import duplicate_aware_text_metrics
from phase1.joint_model import JointAligner

class DuplicateAwareCaptionTests(unittest.TestCase):
    def test_duplicate_caption_candidates_count_as_same_answer(self):
        scores=torch.tensor([[1.,5.,0.],[6.,2.,0.],[0.,0.,9.]])
        text=['same','same','other'];image=['img-a','img-b','img-c']
        r=duplicate_aware_text_metrics(scores,text,text,image)
        self.assertEqual(r['top1'],100.0)
        self.assertEqual(r['top5'],100.0)

    def test_gallery_row_permutation_invariant(self):
        scores=torch.tensor([[1.,5.,0.],[6.,2.,0.],[0.,0.,9.]])
        text=['same','same','other'];image=['img-a','img-b','img-c']
        perm=[2,0,1]
        r0=duplicate_aware_text_metrics(scores,text,text,image)
        r1=duplicate_aware_text_metrics(scores[:,perm],text,[text[i] for i in perm],[image[i] for i in perm])
        self.assertEqual(r0,r1)

    def test_duplicate_caption_student_loss_finite_with_gradients(self):
        torch.manual_seed(7)
        model=JointAligner(8,[6]*5,4,rank=3)
        eeg=torch.randn(3,8)
        visuals=[torch.randn(3,6) for _ in range(5)]
        ze,zi=model.encode(eeg,visuals)
        text=torch.randn(3,4);text[1]=text[0]
        residual=torch.randn(3,4)
        terms=model.loss_terms(ze,zi,text,residual,['a','b','c'],caption_ids=['same','same','other'])
        self.assertTrue(torch.isfinite(terms['total']).item())
        terms['total'].backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_missing_positive_rejected(self):
        with self.assertRaises(ValueError):
            duplicate_aware_text_metrics(torch.eye(2),['unseen','b'],['a','b'],['i1','i2'])

if __name__=='__main__':unittest.main()
