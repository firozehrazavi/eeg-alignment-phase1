import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase1 import historical_06b as baseline


class Historical06BTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.concepts, self.per, self.chunk, self.width = 6, 2, 3, 4, 5
        self.parts = ("feat6_internvit6b_parts/train_000.pt",
                      "feat6_internvit6b_parts/train_001.pt")
        (self.root / "feat6_internvit6b_parts").mkdir()
        rng = np.random.default_rng(42)
        eeg = rng.normal(size=(self.images, 63, 2)).astype(np.float16)
        np.save(self.root / "sub-01_train_z16.npy", eeg)
        for i, name in enumerate(self.parts):
            rows = min(self.chunk, self.images - i * self.chunk)
            tensors = {
                (layer, pool): torch.randn(rows, self.width).half()
                for layer in baseline.FEATURE_LAYERS for pool in ("mean", "cls")
            }
            torch.save(tensors, self.root / name)
        self.patches = patch.multiple(
            baseline,
            TRAIN_IMAGES=self.images,
            TRAIN_CONCEPTS=self.concepts,
            PER_CONCEPT=self.per,
            FEATURE_CHUNK=self.chunk,
            FEATURE_WIDTH=self.width,
            PARTS=self.parts,
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.manifest_path = self.root / "manifest.json"
        baseline.create_manifest(self.root, self.manifest_path)

    def test_preflight_validates_train_only_rows_and_work_profile(self):
        report = baseline.preflight(self.manifest_path)
        self.assertEqual(report["status"], "cpu_preflight_passed")
        self.assertFalse(report["official_test_opened"])
        self.assertEqual(report["route_layers"], [20, 24, 28, 32, 36])
        self.assertEqual(report["fit_images_per_epoch"], 4)
        self.assertEqual(report["validation_images"], 2)
        self.assertEqual(report["total_optimizer_updates"],
                         baseline.EPOCHS * math.ceil(4 / baseline.BATCH))
        for path in report["input_sha256"]:
            self.assertNotIn("test", path.lower())
        profile = baseline.profile_cpu(self.manifest_path, batch_size=32)
        self.assertEqual(profile["profile_batch_size"], 4)
        self.assertEqual(profile["requested_profile_batch_size"], 32)
        self.assertIsNone(profile["linear_cpu_hours_for_full_schedule"])
        self.assertEqual(profile["device"], "cpu")
        self.assertGreaterEqual(profile["profile_step_seconds"], 0)
        self.assertFalse(profile["official_test_opened"])

    def test_preflight_rejects_modified_and_test_named_inputs(self):
        path = self.root / self.parts[0]
        tensors = torch.load(path, weights_only=True)
        torch.save(tensors, path)
        data = json.loads(self.manifest_path.read_text())
        data["source_hashes"][self.parts[0]] = "bad"
        self.manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "changed"):
            baseline.preflight(self.manifest_path)
        data["source_hashes"][self.parts[0]] = baseline._sha256(path)
        data["feature_parts"][0] = "feat6_internvit6b_parts/test_000.pt"
        self.manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            baseline.preflight(self.manifest_path)

    def test_epoch_checkpoint_resume_matches_uninterrupted_run(self):
        with patch.multiple(baseline, EPOCHS=3, BATCH=4, EVAL_EVERY=1):
            reference = baseline.run_baseline(self.manifest_path, self.root / "reference", "cpu")
            changed_runtime = baseline._runtime_signature(torch.device("cpu"), False)
            changed_runtime["float32_matmul_precision"] = "different_precision"
            with patch.object(baseline, "_runtime_signature", return_value=changed_runtime):
                with self.assertRaisesRegex(ValueError, "environment"):
                    baseline.run_baseline(self.manifest_path, self.root / "reference",
                                          "cpu", resume=True)
            original_validation = baseline._validation_top1
            calls = {"n": 0}

            def interrupt_on_second_validation(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("injected interruption")
                return original_validation(*args, **kwargs)

            with patch.object(baseline, "_validation_top1",
                              side_effect=interrupt_on_second_validation):
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    baseline.run_baseline(self.manifest_path, self.root / "resumed", "cpu")
            latest = torch.load(
                self.root / "resumed/within_route_c_sub-01_s0_latest.pt",
                map_location="cpu", weights_only=True)
            self.assertEqual(latest["epoch"], 1)
            resumed = baseline.run_baseline(
                self.manifest_path, self.root / "resumed", "cpu", resume=True)
            self.assertEqual(reference["selected_epoch"], resumed["selected_epoch"])
            self.assertEqual(reference["validation_top1"], resumed["validation_top1"])
            self.assertEqual(reference["checkpoint_keys"], resumed["checkpoint_keys"])
            expected = torch.load(self.root / "reference/within_route_c_sub-01_s0_best.pt",
                                  map_location="cpu", weights_only=True)
            actual = torch.load(self.root / "resumed/within_route_c_sub-01_s0_best.pt",
                                map_location="cpu", weights_only=True)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_runtime_signature_uses_requested_cuda_device(self):
        # No GPU work: metadata mocks ensure cuda:1 is not mistaken for cuda:0.
        with patch.object(torch.cuda, "get_device_name", return_value="GPU one") as name, \
                patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)) as capability, \
                patch.object(torch.cuda, "device_count", return_value=2):
            signature = baseline._runtime_signature(torch.device("cuda:1"), True)
        name.assert_called_once_with(1)
        capability.assert_called_once_with(1)
        self.assertEqual(signature["device_index"], 1)
        self.assertEqual(signature["device_capability"], [8, 0])
        self.assertEqual(signature["autocast_dtype"], "bfloat16")
        self.assertNotEqual(signature, baseline._runtime_signature("cpu", False))
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            baseline._runtime_signature("cpu", True)


if __name__ == "__main__":
    unittest.main()
