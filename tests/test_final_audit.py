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
