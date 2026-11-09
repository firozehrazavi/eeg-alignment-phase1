import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from phase1.process_log import run_logged

class ProcessLogTests(unittest.TestCase):
    def test_early_failure_is_visible_and_saved_without_token_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);screen=io.StringIO()
            with contextlib.redirect_stdout(screen):
                with self.assertRaisesRegex(RuntimeError,'saved log'):
                    run_logged([sys.executable,'-c',"print('hello',flush=True); raise RuntimeError('early sentinel')"],root,root/'logs')
            logs=list((root/'logs').glob('*.log'));self.assertEqual(len(logs),1)
            for text in (screen.getvalue(),logs[0].read_text()):
                self.assertIn('hello',text);self.assertIn('Traceback',text);self.assertIn('early sentinel',text)
            self.assertFalse((root/'tokens_L28').exists())
    def test_success_and_launch_failure_preserve_distinct_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                p=run_logged([sys.executable,'-c',"print('ok')"],root,root/'logs')
                with self.assertRaises(OSError):run_logged([str(root/'missing-executable')],root,root/'logs')
            self.assertIn('ok',p.read_text());self.assertEqual(len(list((root/'logs').glob('*.log'))),2)
