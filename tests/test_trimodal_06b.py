import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase1 import historical_06b as baseline
from phase1 import trimodal_06b as trimodal


class Historical06BTrimodalTests(unittest.TestCase):
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
        np.save(self.root / "sub-01_train_z16.npy",
                rng.normal(size=(6, 63, 2)).astype(np.float16))
        for i, name in enumerate(self.parts):
            rows = min(self.chunk, self.images - i * self.chunk)
            torch.save({
                (layer, pool): torch.randn(rows, self.width).half()
                for layer in baseline.FEATURE_LAYERS for pool in ("mean", "cls")
            }, self.root / name)
        self.base_patch = patch.multiple(
            baseline, TRAIN_IMAGES=self.images, TRAIN_CONCEPTS=self.concepts,
            PER_CONCEPT=self.per, FEATURE_CHUNK=self.chunk,
            FEATURE_WIDTH=self.width, PARTS=self.parts,
            EPOCHS=2, BATCH=4, EVAL_EVERY=1)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.trimodal_patch = patch.multiple(
            trimodal, TRAIN_IMAGES=self.images, PER_CONCEPT=self.per,
            FEATURE_WIDTH=self.width, TEXT_WIDTH=4, RANK=2,
            EPOCHS=2, BATCH=4, EVAL_EVERY=1)
        self.trimodal_patch.start()
        self.addCleanup(self.trimodal_patch.stop)
        self.manifest = self.root / "manifest.json"
        baseline.create_manifest(self.root, self.manifest)

        text = rng.normal(size=(self.images, 4)).astype(np.float32)
        text[5] = text[4]
        self.text_path = self.root / "captions_train_only.npy"
        np.save(self.text_path, text)
        self.ids_path = self.root / "caption_ids_train_only.json"
        self.ids_path.write_text(json.dumps(["a", "b", "c", "d", "same", "same"]))
        self.provenance_path = self.root / "caption_provenance_train_only.json"
        self.provenance_path.write_text(json.dumps({
            "partition": "official_training",
            "caption_generator": "train-only fixture",
            "caption_encoder": "fixture",
            "encoder_revision": "fixed",
            "embedding_normalization": "fit on training rows",
            "source_image_partition": "official_training",
            "used_test": False,
        }))

    def _baseline(self):
        return baseline.run_baseline(self.manifest, self.root / "baseline", "cpu")

    def test_caption_preflight_requires_train_only_provenance(self):
        report = trimodal.preflight_text_inputs(
            self.text_path, self.ids_path, self.provenance_path)
        self.assertEqual(report["status"], "cpu_caption_preflight_passed")
        self.assertEqual(report["unique_caption_identities"], 5)
        ids = ["a", "b", "c", "d", "caption-1", "caption-2"]
        self.ids_path.write_text(json.dumps(ids))
        accepted = trimodal.preflight_text_inputs(
            self.text_path, self.ids_path, self.provenance_path)
        self.assertEqual(accepted["unique_caption_identities"], 6)
        ids[-1] = ids[-2]
        self.ids_path.write_text(json.dumps(ids))
        features = np.load(self.text_path)
        features[-1, 0] += 1
        np.save(self.text_path, features)
        with self.assertRaisesRegex(ValueError, "inconsistent embeddings"):
            trimodal.preflight_text_inputs(
                self.text_path, self.ids_path, self.provenance_path)
        data = json.loads(self.provenance_path.read_text())
        data["used_test"] = True
        self.provenance_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "exclusively"):
            trimodal.preflight_text_inputs(
                self.text_path, self.ids_path, self.provenance_path)

    def test_matched_arms_warmstart_and_direct_text_evaluation(self):
        self._baseline()
        rows = trimodal.run(
            self.manifest, self.root / "baseline", self.text_path, self.ids_path,
            self.provenance_path, self.root / "trimodal", device_name="cpu")
        self.assertEqual([row["arm"] for row in rows], list("ABCD"))
        self.assertTrue(all(row["setting"] == "historical_06b_warm_start"
                            for row in rows))
        self.assertTrue(all(row["candidates"] == 2 for row in rows))
        self.assertTrue(all("text_top1" in row and "image_top1" in row
                            for row in rows))
        run = json.loads((self.root / "trimodal/run.json").read_text())
        self.assertFalse(run["official_test_opened"])
        self.assertIn("no image candidate features", run["direct_text_inference"])

    def test_interrupted_abcd_resumes_identically(self):
        self._baseline()
        common = (self.manifest, self.root / "baseline", self.text_path,
                  self.ids_path, self.provenance_path)
        reference = trimodal.run(*common, self.root / "trimodal_reference",
                                 device_name="cpu")
        original = trimodal._validation_metrics
        calls = {"n": 0}

        def interrupt_after_first_epoch(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected tri-modal interruption")
            return original(*args, **kwargs)

        with patch.object(trimodal, "_validation_metrics",
                          side_effect=interrupt_after_first_epoch):
            with self.assertRaisesRegex(RuntimeError, "injected tri-modal"):
                trimodal.run(*common, self.root / "trimodal_resumed",
                             device_name="cpu")
        latest = torch.load(
            self.root / "trimodal_resumed/sub-01_s0_A_latest.pt",
            map_location="cpu", weights_only=True)
        self.assertEqual(latest["epoch"], 1)
        resumed = trimodal.run(*common, self.root / "trimodal_resumed",
                               device_name="cpu", resume=True)
        for expected_row, actual_row in zip(reference, resumed):
            self.assertEqual(expected_row["arm"], actual_row["arm"])
            self.assertEqual(expected_row["image_top1"], actual_row["image_top1"])
            self.assertEqual(expected_row["text_top1"], actual_row["text_top1"])
        for arm in "ABCD":
            expected = torch.load(
                self.root / f"trimodal_reference/sub-01_s0_{arm}_best.pt",
                map_location="cpu", weights_only=True)
            actual = torch.load(
                self.root / f"trimodal_resumed/sub-01_s0_{arm}_best.pt",
                map_location="cpu", weights_only=True)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_bfloat16_backbone_outputs_train_all_float32_heads(self):
        """Exercise the real runner's AMP boundary without requiring a GPU."""
        self._baseline()
        seen = []
        original = trimodal.JointAligner.text_scores

        def checked_text_scores(model, ze, text):
            seen.append(ze.dtype)
            self.assertEqual(ze.dtype, torch.float32)
            return original(model, ze, text)

        with patch.object(trimodal, "_autocast", side_effect=lambda *args:
                          torch.autocast("cpu", dtype=torch.bfloat16)), \
             patch.object(trimodal.JointAligner, "text_scores", checked_text_scores):
            records = trimodal.run(
                self.manifest, self.root / "baseline", self.text_path,
                self.ids_path, self.provenance_path, self.root / "trimodal_amp",
                device_name="cpu")
        self.assertEqual([row["arm"] for row in records], list("ABCD"))
        self.assertTrue(seen)
        latest = torch.load(self.root / "trimodal_amp/sub-01_s0_B_latest.pt",
                            map_location="cpu", weights_only=True)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in latest["model"].values()))

    def test_fresh_resume_saves_reconstructable_preprocessing_and_checks_artifacts(self):
        self._baseline()
        common = (self.manifest, self.root / "baseline", self.text_path,
                  self.ids_path, self.provenance_path)
        output = self.root / "trimodal_fresh_resume"
        # The Colab entry uses resume=True even on the first invocation.
        records = trimodal.run(*common, output, device_name="cpu", resume=True)
        self.assertEqual([row["arm"] for row in records], list("ABCD"))
        metadata = json.loads((output / "run.json").read_text())
        prep_path = output / metadata["preprocessing"]
        self.assertEqual(baseline._sha256(prep_path), metadata["preprocessing_sha256"])
        prep = torch.load(prep_path, map_location="cpu", weights_only=True)
        model = trimodal.JointAligner(**prep["model_dimensions"]).eval()
        model.load_state_dict(torch.load(output / "sub-01_s0_B_best.pt",
                                         map_location="cpu", weights_only=True))
        saved_scores = torch.load(output / "sub-01_s0_B_validation_scores.pt",
                                  map_location="cpu", weights_only=True)
        self.assertEqual(prep["eeg_normalization"], "historical_z16_no_additional_fit")
        # Reconstruct direct text scores solely from a checkpoint, its frozen
        # preprocessing, EEG queries and raw caption candidates. No fitting or
        # image feature access is allowed on this inference path.
        original_load = np.load
        allowed = {self.root / "sub-01_train_z16.npy", self.text_path}

        def text_only_load(path, *args, **kwargs):
            self.assertIn(Path(path), allowed)
            return original_load(path, *args, **kwargs)

        with patch.object(np, "load", side_effect=text_only_load), \
             patch.object(trimodal, "_load_training", side_effect=AssertionError("Training loaded")), \
             patch.object(trimodal.Residualizer, "fit", side_effect=AssertionError("Refitted")), \
             patch.object(trimodal.Residualizer, "transform", side_effect=AssertionError("Image residual used")), \
             patch.object(trimodal.JointAligner, "image_scores", side_effect=AssertionError("Image scores used")), \
             torch.no_grad():
            rows = prep["validation_rows"]
            eeg = torch.from_numpy(np.load(self.root / "sub-01_train_z16.npy", allow_pickle=False))
            eeg = eeg[:, prep["eeg_source_channels"], :][rows].float()
            raw_text = torch.from_numpy(np.load(self.text_path, allow_pickle=False))
            candidates = ((raw_text - prep["text_mean"]) / prep["text_std"])[rows]
            actual = model.text_scores(model.backbone.eeg(eeg), candidates)
        torch.testing.assert_close(actual, saved_scores["text_scores"], rtol=0, atol=0)

        for filename, message in (
                ("sub-01_s0_A_best.pt", "Completed arm artifact"),
                ("sub-01_s0_B_validation_scores.pt", "Completed arm artifact"),
                ("preprocessing.pt", "preprocessing artifact")):
            artifact = output / filename
            original_bytes = artifact.read_bytes()
            for changed in (False, True):
                with self.subTest(artifact=filename, changed=changed):
                    if changed:
                        artifact.write_bytes(original_bytes + b"corruption")
                    else:
                        artifact.unlink()
                    try:
                        with patch.object(torch.optim.AdamW, "step",
                                          side_effect=AssertionError("Unexpected training")), \
                             self.assertRaisesRegex(ValueError, message):
                            trimodal.run(*common, output, device_name="cpu", resume=True)
                    finally:
                        artifact.write_bytes(original_bytes)


if __name__ == "__main__":
    unittest.main()
