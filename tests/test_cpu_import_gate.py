import ast
import json
from pathlib import Path
import unittest

class CPUImportGateTests(unittest.TestCase):
    def test_pinned_cpu_smoke_and_gpu_gate(self):
        n=json.loads((Path(__file__).resolve().parents[1]/'notebooks/10_rpa_patch_pilot.ipynb').read_text())
        cells={c.get('id'):''.join(c['source']) for c in n['cells']}
        cpu=cells['cpu-environment-smoke'];gpu=cells['proplus-01'];setup=cells['proplus-02']
        for s in (cpu,gpu,setup):compile(s,'cell','exec')
        pin='a9e03c549a5af55025920de6d70cd50a9d5b73b3'
        self.assertIn(pin,cpu);self.assertIn(pin,gpu)
        self.assertIn('assert not torch.cuda.is_available()',cpu)
        self.assertIn("'checkout','--detach',CODE_REVISION",cpu)
        self.assertIn("'checkout','--detach',CODE_REVISION",setup)
        self.assertNotIn('audit/phase1-reproducible-alignment',setup)
        self.assertLess(cpu.index("'pip','install'"),cpu.index("'phase1.extraction_environment'"))
        self.assertLess(cpu.index("'phase1.extraction_environment'"),cpu.index('tmp.replace(EVIDENCE)'))
        self.assertIn("'Child exit code: 0' in text",cpu)
        self.assertLess(gpu.index('assert EVIDENCE.is_file()'),gpu.index('APPROVE_GPU_SESSION'))
        self.assertIn("CPU_SMOKE['requirements_sha256']",setup)
        self.assertIn('phase1.extraction_environment',setup)
        self.assertNotIn('phase1.patch_prepare',cpu)
        self.assertNotIn('phase1.patch_cache',cpu)
