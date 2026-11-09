import ast
import json
from pathlib import Path
import unittest

class RootNotebookTests(unittest.TestCase):
    def test_gate_and_no_training_or_final_test_entry_points(self):
        root=Path(__file__).resolve().parents[1]
        n=json.loads((root/'notebooks/15_validation_root_cause_cpu.ipynb').read_text())
        code=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),1);self.assertEqual(code[0]['outputs'],[])
        source=''.join(code[0]['source']);tree=ast.parse(source)
        with self.assertRaises(AssertionError):exec(compile(ast.Module(body=tree.body[:2],type_ignores=[]),'gate','exec'),{})
        self.assertIn('RUN_CPU_VALIDATION_INFERENCE = False',source)
        self.assertIn("CUDA_VISIBLE_DEVICES=''",source)
        self.assertIn('assert not torch.cuda.is_available()',source)
        self.assertIn('phase1.validation_root_audit',source)
        for forbidden in ['phase1.final_audit','phase1.patch_experiment','--approve-final-test','--output','from_pretrained']:
            self.assertNotIn(forbidden,source)
        compile(source,'audit_notebook','exec')
