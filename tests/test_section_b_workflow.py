import ast
import json
from pathlib import Path
import unittest

class SectionBWorkflowTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/10_rpa_patch_pilot.ipynb'
        self.cells={c.get('id'):c for c in json.loads(p.read_text())['cells']}
    def test_fresh_gpu_setup_defines_all_paths_without_section_a(self):
        source=''.join(self.cells['proplus-01']['source'])
        names={'RAW','CACHE','WORK','PREP','TOKEN_DIR','PREPARED_INPUTS_CHECKED'}
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
        env={'Path':Path}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'setup_paths','exec'),env)
        self.assertTrue(names<=env.keys())
        self.assertEqual(str(env['PREP']).replace('\\','/'),'/content/drive/MyDrive/eeg_results/patch_phase1/pilot_sub01_200concepts_v1')
        self.assertEqual(env['TOKEN_DIR'],env['PREP']/'tokens_L28')
        self.assertFalse(env['PREPARED_INPUTS_CHECKED'])
    def test_setup_includes_validation_and_probe_fails_clearly_if_skipped(self):
        source=''.join(self.cells['proplus-02']['source'])
        self.assertIn('phase1.pilot_preflight',source)
        self.assertGreater(source.index('PREPARED_INPUTS_CHECKED = True'),source.index('phase1.pilot_preflight'))
        self.assertNotIn('phase1.patch_prepare',source)
        self.assertNotIn('phase1.pilot_inputs',source)
        probe=ast.parse(''.join(self.cells['proplus-06']['source']))
        guard=compile(ast.Module(body=[probe.body[0]],type_ignores=[]),'probe_guard','exec')
        with self.assertRaisesRegex(AssertionError,'setup'):exec(guard,{})
        exec(guard,{'PREPARED_INPUTS_CHECKED':True})
        for c in self.cells.values():
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
