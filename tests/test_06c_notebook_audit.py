"""Exercise the 06c bootstrap without networking, Drive mounts, or training."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import pathlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/06c_checkpointed_06b_and_trimodal.ipynb'


class Notebook06CBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.bootstrap = ''.join(self.notebook['cells'][1]['source'])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name) / 'checkout'
        shutil.copytree(ROOT / 'phase1', self.checkout / 'phase1',
                        ignore=shutil.ignore_patterns('__pycache__'))

    def execute(self, branch='research/semantic-correction-bd-20261121', dirty='', stale_phase1=False):
        def git_output(command, **kwargs):
            if command[-2:] == ['branch', '--show-current']:
                return branch
            if command[-2:] == ['status', '--porcelain']:
                return dirty
            if command[-2:] == ['rev-parse', 'HEAD']:
                return 'verified-commit-for-test'
            self.fail(f'Unexpected git read: {command}')
        # Other unittest modules may already import phase1; model a fresh kernel
        # here, except in the explicit stale-import regression.
        modules = {name: module for name, module in sys.modules.copy().items()
                   if name != 'phase1' and not name.startswith('phase1.')}
        modules['google.colab'] = None
        if stale_phase1:
            modules['phase1'] = types.ModuleType('phase1')
        with patch.dict(os.environ, {'EEG_06C_REPO': str(self.checkout)}, clear=False), \
                patch.dict(sys.modules, modules, clear=True), \
                patch.object(sys, 'path', copy.copy(sys.path)), \
                patch.object(subprocess, 'run') as run, \
                patch.object(subprocess, 'check_output', side_effect=git_output), \
                contextlib.redirect_stdout(io.StringIO()):
            namespace = {}
            exec(compile(self.bootstrap, str(NOTEBOOK), 'exec'), namespace)
            return namespace, run.call_args_list

    def test_current_sources_are_pinned_and_cpu_paths_are_configurable(self):
        namespace, commands = self.execute()
        self.assertEqual(namespace['REPO'], self.checkout)
        self.assertEqual(namespace['EXPECTED_SOURCE_SHA256'], {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted((ROOT / 'phase1').rglob('*.py'))})
        self.assertEqual(commands[0].args[0][-3:],
                         ['--ff-only', 'origin', 'research/semantic-correction-bd-20261121'])
        self.assertIn("os.environ.get('EEG_CACHE'", ''.join(self.notebook['cells'][3]['source']))
        self.assertIn("os.environ.get('EEG_06C_RUN_ROOT'", ''.join(self.notebook['cells'][3]['source']))

    def test_updated_branch_cannot_silently_change_audited_code(self):
        with (self.checkout / 'phase1/historical_06b.py').open('a') as handle:
            handle.write('\n# changed after notebook publication\n')
        with self.assertRaisesRegex(RuntimeError, 'source hash mismatch'):
            self.execute()

    def test_reused_wrong_branch_or_dirty_checkout_is_preserved(self):
        for branch, dirty in [('main', ''), ('research/semantic-correction-bd-20261121', ' M local.py')]:
            with self.subTest(branch=branch, dirty=dirty):
                with self.assertRaisesRegex(RuntimeError, 'another branch or has local changes'):
                    self.execute(branch, dirty)


    def test_cached_phase1_cannot_bypass_verified_disk_sources(self):
        with self.assertRaisesRegex(RuntimeError, 'already imported.*Restart'):
            self.execute(stale_phase1=True)

    def test_cpu_cells_execute_real_preflight_and_profile_without_training(self):
        # Tiny independent fixture: actual loaders/forward/backward run; neither
        # baseline nor tri-modal full training may be entered by these cells.
        from phase1 import historical_06b as baseline, trimodal_06b as trimodal
        cache = Path(self.temp.name) / 'cache'
        output = Path(self.temp.name) / 'run'
        parts = ('feat6_internvit6b_parts/train_000.pt',
                 'feat6_internvit6b_parts/train_001.pt')
        (cache / 'feat6_internvit6b_parts').mkdir(parents=True)
        rng = np.random.default_rng(42)
        np.save(cache / 'sub-01_train_z16.npy', rng.normal(size=(6, 63, 2)).astype(np.float16))
        generator = torch.Generator().manual_seed(42)
        for name, rows in zip(parts, (4, 2)):
            torch.save({key: torch.randn(rows, 5, generator=generator).half()
                        for key in baseline.SELECTED_KEYS}, cache / name)
        namespace = {'pathlib': pathlib, 'json': json, 'os': os, 'torch': torch}
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, prior_threads)
        with patch.multiple(baseline, TRAIN_IMAGES=6, TRAIN_CONCEPTS=2,
                            PER_CONCEPT=3, FEATURE_CHUNK=4, FEATURE_WIDTH=5,
                            PARTS=parts), \
                patch.dict(os.environ, {'EEG_CACHE': str(cache),
                                        'EEG_06C_RUN_ROOT': str(output)}), \
                patch.object(baseline, 'run_baseline',
                             side_effect=AssertionError('Full baseline training entered')) as baseline_run, \
                patch.object(trimodal, 'run',
                             side_effect=AssertionError('Full tri-modal training entered')) as trimodal_run, \
                contextlib.redirect_stdout(io.StringIO()):
            output.mkdir()
            baseline.create_manifest(cache, output / 'train_only_manifest.json')
            for index in (3, 4, 6, 8, 9):
                source = ''.join(self.notebook['cells'][index]['source'])
                source = source.replace('RUN_CPU_PREFLIGHT=False', 'RUN_CPU_PREFLIGHT=True')
                source = source.replace('RUN_CPU_PROFILE=False', 'RUN_CPU_PROFILE=True')
                exec(compile(source,
                             f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        baseline_run.assert_not_called()
        trimodal_run.assert_not_called()
        report = json.loads((output / 'budgeted_cpu_checks_20261121/cpu_preflight.json').read_text())
        profile = json.loads((output / 'budgeted_cpu_checks_20261121/cpu_profile.json').read_text())
        self.assertEqual(report['status'], 'cpu_preflight_passed')
        self.assertEqual(report['fit_images_per_epoch'], 4)
        self.assertFalse(report['official_test_opened'])
        self.assertEqual(profile['device'], 'cpu')
        self.assertGreaterEqual(profile['profile_step_seconds'], 0)
        self.assertFalse(profile['official_test_opened'])
        self.assertEqual(namespace['caption_files'], [cache / 'captions_train_only.npy',
                                                     cache / 'caption_ids_train_only.json',
                                                     cache / 'caption_provenance_train_only.json'])


    def test_default_continuation_skips_completed_work_and_preserves_inputs(self):
        from phase1 import historical_06b as baseline, trimodal_06b as trimodal
        output = Path(self.temp.name) / 'existing_run'
        cache = Path(self.temp.name) / 'existing_cache'
        (output / 'baseline').mkdir(parents=True)
        cache.mkdir()
        protected = [output / 'train_only_manifest.json', output / 'cpu_preflight.json',
                     output / 'cpu_profile.json', output / 'baseline/run.json',
                     output / 'baseline/within_route_c_sub-01_s0_best.pt',
                     cache / 'captions_train_only.npy', cache / 'caption_ids_train_only.json',
                     cache / 'caption_provenance_train_only.json']
        for path in protected:
            path.write_bytes(b'completed artifact retained verbatim')
        before = {path: path.read_bytes() for path in protected}
        namespace = {'pathlib': pathlib, 'json': json, 'os': os, 'torch': torch}
        with patch.dict(os.environ, {'EEG_CACHE': str(cache), 'EEG_06C_RUN_ROOT': str(output)}), \
                patch.object(baseline, 'preflight', side_effect=AssertionError('Repeated preflight')), \
                patch.object(baseline, 'profile_cpu', side_effect=AssertionError('Repeated CPU update')), \
                patch.object(baseline, 'run_baseline', side_effect=AssertionError('Baseline rerun')), \
                patch.object(trimodal, 'preflight_text_inputs', side_effect=AssertionError('Repeated caption preflight')), \
                patch.object(trimodal, 'run', side_effect=AssertionError('Unexpected GPU run')), \
                contextlib.redirect_stdout(io.StringIO()):
            for index in (3, 4, 6, 8, 9):
                exec(''.join(self.notebook['cells'][index]['source']), namespace)
        self.assertEqual(before, {path: path.read_bytes() for path in protected})
        self.assertFalse((output / 'budgeted_cpu_checks_20261121').exists())
        self.assertFalse(namespace['TRIMODAL_OUTPUT'].exists())
        self.assertEqual(namespace['TRIMODAL_OUTPUT'], output / 'trimodal_semantic_correction_bd_20261121')

    def test_enabled_gpu_controls_resume_one_new_output_in_both_modes(self):
        from phase1 import trimodal_06b as trimodal
        output = Path(self.temp.name) / 'existing_run'
        cache = Path(self.temp.name) / 'existing_cache'
        cache.mkdir()
        caption_files = [cache / name for name in ('captions_train_only.npy',
                         'caption_ids_train_only.json', 'caption_provenance_train_only.json')]
        for path in caption_files:
            path.write_bytes(b'input existence fixture; runner is mocked')
        for profile, arms in [(True, ('B','D')), (True, ('B',)), (False, ('B','D'))]:
            with self.subTest(profile=profile, arms=arms):
                namespace = {'pathlib': pathlib, 'json': json, 'os': os, 'torch': torch,
                             'RUN_ROOT': output, 'MANIFEST': output / 'train_only_manifest.json',
                             'BASELINE_OUTPUT': output / 'baseline', 'caption_files': caption_files,
                             'TEXT_FEATURES': caption_files[0], 'CAPTION_IDS': caption_files[1],
                             'CAPTION_PROVENANCE': caption_files[2]}
                source = ''.join(self.notebook['cells'][9]['source'])
                source = source.replace('RUN_TRIMODAL=False', 'RUN_TRIMODAL=True')
                source = source.replace('PROFILE=True', f'PROFILE={profile}')
                source = source.replace("ARMS=('B','D')", f'ARMS={arms!r}')
                with patch.object(torch.cuda, 'is_available', return_value=True), \
                        patch.object(trimodal, 'run', return_value=[]) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    namespace['run_trimodal'] = run
                    exec(source, namespace)
                self.assertEqual(run.call_args.args[1], output / 'baseline')
                self.assertEqual(run.call_args.args[5], output / 'trimodal_semantic_correction_bd_20261121')
                self.assertEqual(run.call_args.kwargs,
                                 {'resume': True, 'arms': arms, 'max_seconds': 900.0,
                                  'max_epochs_per_arm': 5 if profile else None,
                                   'semantic_correction': True})

    def test_invalid_budget_cannot_enter_runner(self):
        from unittest.mock import Mock
        source = ''.join(self.notebook['cells'][9]['source']).replace('RUN_TRIMODAL=False', 'RUN_TRIMODAL=True')
        for budget in ["float('inf')", "float('nan')", '-1', '0']:
            with self.subTest(budget=budget):
                run = Mock(side_effect=AssertionError('Invalid budget entered runner'))
                namespace = {'RUN_ROOT': Path(self.temp.name), 'run_trimodal': run}
                with self.assertRaisesRegex(ValueError, 'finite positive'):
                    exec(source.replace('TIME_BUDGET_SECONDS=900.0', f'TIME_BUDGET_SECONDS={budget}'), namespace)
                run.assert_not_called()

    def test_gpu_actions_remain_off_and_cuda_determinism_precedes_import(self):
        self.assertLess(self.bootstrap.index("os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG'"),
                        self.bootstrap.index('import numpy as np, torch'))
        self.assertIn('RUN_BASELINE=False', ''.join(self.notebook['cells'][6]['source']))
        self.assertIn('RUN_TRIMODAL=False', ''.join(self.notebook['cells'][9]['source']))
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(NOTEBOOK), 'exec')


if __name__ == '__main__':
    unittest.main()

