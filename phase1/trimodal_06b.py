"""Matched warm-start A/B/C/D comparison for historical 06b route_c.

All rows are from the official training partition. Direct text retrieval ranks
caption embeddings without any image candidate features at inference time.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .core import clip_loss, retrieval_metrics
from .experiment import seed_all
from .historical_06b import (
    BATCH, EPOCHS, EVAL_EVERY, FEATURE_WIDTH, LR, PER_CONCEPT, POST17, ROUTE_INIT,
    ROUTE_LAYERS, WEIGHT_DECAY, _autocast, _code_hashes, _load_training,
    TRAIN_IMAGES, POST17_INDICES, _runtime_signature, _sha256, _torch_save_atomic,
    _write_json_atomic, _write_epoch_log,
    preflight as baseline_preflight,
)
from .joint_experiment import duplicate_aware_text_metrics
from .joint_model import JointAligner, Residualizer, symmetric_identity_nce

TEXT_WIDTH = 512
RANK = 64
RIDGE = 1.
JOINT_WEIGHT = .1
DISTILL_WEIGHT = .1
READOUT_WEIGHT = .1


def _safe_input(path):
    p = Path(path).resolve()
    if "test" in p.name.casefold() or "eval" in p.name.casefold():
        raise ValueError(f"Test/evaluation-looking caption input refused: {p.name}")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def preflight_text_inputs(feature_path, caption_ids_path, provenance_path):
    """Validate a separately prepared, training-only full-set caption cache."""
    feature_path = _safe_input(feature_path)
    caption_ids_path = _safe_input(caption_ids_path)
    provenance_path = _safe_input(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = ("partition", "caption_generator", "caption_encoder", "encoder_revision",
                "embedding_normalization", "source_image_partition", "used_test")
    if any(key not in provenance for key in required):
        raise ValueError("Caption provenance must document generator, encoder, revision, normalization and partition")
    if any(not provenance[key] for key in required if key != "used_test"):
        raise ValueError("Caption provenance values must be nonempty")
    if (provenance["partition"] != "official_training"
            or provenance["source_image_partition"] != "official_training"
            or provenance["used_test"] is not False):
        raise ValueError("Caption inputs must be derived exclusively from official training images")
    if "train" not in provenance["embedding_normalization"].casefold():
        raise ValueError("Caption embedding normalization must be documented as training-only")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if (features.shape != (TRAIN_IMAGES, TEXT_WIDTH)
            or not np.issubdtype(features.dtype, np.floating)):
        raise ValueError(f"Expected training-only text embeddings [16540, {TEXT_WIDTH}]")
    for start in range(0, len(features), 1024):
        if not np.isfinite(features[start:start + 1024]).all():
            raise ValueError("Nonfinite train caption embedding")
    caption_ids = json.loads(caption_ids_path.read_text(encoding="utf-8"))
    if (not isinstance(caption_ids, list) or len(caption_ids) != TRAIN_IMAGES
            or not all(isinstance(value, str) and value for value in caption_ids)):
        raise ValueError("Expected 16540 nonempty exact-caption identity strings")
    # Duplicate exact captions must share both identity and embedding. Equal
    # vectors alone cannot establish that two distinct captions are identical.
    first = {}
    for index, caption_id in enumerate(caption_ids):
        vector = np.asarray(features[index], dtype=np.float32)
        if caption_id in first and not np.array_equal(vector, first[caption_id]):
            raise ValueError("Identical caption identities have inconsistent embeddings")
        first.setdefault(caption_id, vector)
    return {
        "status": "cpu_caption_preflight_passed",
        "train_only": True,
        "caption_feature_shape": list(features.shape),
        "unique_caption_identities": len(set(caption_ids)),
        "feature_sha256": _sha256(feature_path),
        "caption_ids_sha256": _sha256(caption_ids_path),
        "provenance_sha256": _sha256(provenance_path),
        "provenance": provenance,
    }


def _load_text(feature_path, caption_ids_path, device):
    values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    text = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)
    caption_ids = json.loads(Path(caption_ids_path).read_text(encoding="utf-8"))
    return text, caption_ids


def _validation_metrics(model, eeg, visual, text, rows, caption_ids, device, batch):
    model.eval()
    with torch.no_grad():
        ze = torch.cat([model.backbone.eeg(eeg[ix].float())
                        for ix in rows.split(batch)])
        zi = model.backbone.img([x[rows] for x in visual])
        image_scores = model.image_scores(ze, zi).cpu()
        text_scores = model.text_scores(ze, text[rows]).cpu()
    image_ids = [str(index) for index in rows.cpu().tolist()]
    identities = [caption_ids[index] for index in rows.cpu().tolist()]
    image_metrics = retrieval_metrics(image_scores, image_ids, image_ids, len(image_ids))
    text_metrics = duplicate_aware_text_metrics(text_scores, identities, identities, image_ids)
    return ({**{"image_" + key: value for key, value in image_metrics.items()},
             **{"text_" + key: value for key, value in text_metrics.items()}},
            {"image_scores": image_scores, "text_scores": text_scores,
             "image_ids": image_ids, "caption_ids": identities})


def _atomic_results(records, path):
    if not records:
        return
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _verify_completed_artifacts(output, record):
    """A CSV row alone is not evidence that a completed arm is recoverable."""
    name = f"sub-01_s0_{record['arm']}"
    for field, suffix in (("checkpoint", "best.pt"),
                          ("validation_scores", "validation_scores.pt")):
        expected = f"{name}_{suffix}"
        path = output / expected
        if (record.get(field) != expected or not path.is_file()
                or record.get(field + "_sha256") != _sha256(path)):
            raise ValueError(f"Completed arm artifact missing or changed: {expected}")


def run(manifest_path, baseline_output, text_features, caption_ids_path,
        caption_provenance, output, device_name="cuda", resume=False):
    """Warm-start all four arms from the verified 06b best-validation checkpoint."""
    base_report = baseline_preflight(manifest_path)
    text_report = preflight_text_inputs(text_features, caption_ids_path, caption_provenance)
    baseline_output = Path(baseline_output).resolve()
    baseline_record_path = baseline_output / "run.json"
    if not baseline_record_path.is_file():
        raise ValueError("A completed 06b baseline run.json is required before tri-modal training")
    baseline_record = json.loads(baseline_record_path.read_text())
    if (baseline_record.get("status") != "complete_training_only"
            or baseline_record.get("official_test_opened") is not False
            or baseline_record.get("subject") != "sub-01"
            or baseline_record.get("seed") != 0
            or baseline_record.get("layers") != list(ROUTE_LAYERS)
            or baseline_record.get("pool") != "mean"
            or baseline_record.get("channels") != list(POST17)
            or baseline_record.get("input_sha256") != base_report["input_sha256"]
            or baseline_record.get("signature", {}).get("manifest_sha256")
            != base_report["manifest_sha256"]
            or baseline_record.get("checkpoint")
            != "within_route_c_sub-01_s0_best.pt"):
        raise ValueError("Baseline checkpoint provenance is not the required 06b sub-01 route_c run")
    baseline_path = baseline_output / baseline_record["checkpoint"]
    if _sha256(baseline_path) != baseline_record["checkpoint_sha256"]:
        raise ValueError("Historical 06b checkpoint SHA256 mismatch")
    backbone_state = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if list(backbone_state) != baseline_record["checkpoint_keys"]:
        raise ValueError("Historical 06b checkpoint keys differ from its run record")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    initial_seed = 10000
    # Record the effective runtime, including seed_all's deterministic setting.
    seed_all(initial_seed)

    signature = dict(
        baseline_sha256=baseline_record["checkpoint_sha256"],
        baseline_manifest_sha256=base_report["manifest_sha256"],
        text_feature_sha256=text_report["feature_sha256"],
        caption_ids_sha256=text_report["caption_ids_sha256"],
        caption_provenance_sha256=text_report["provenance_sha256"],
        code_hashes={**_code_hashes(),
                     **{name: _sha256(Path(__file__).parent / name)
                        for name in ("trimodal_06b.py", "joint_model.py", "experiment.py",
                                     "joint_data.py", "joint_experiment.py")}},
        python=str(platform.python_version()), torch=str(torch.__version__),
        numpy=str(np.__version__), subject="sub-01", seed=0,
        runtime=_runtime_signature(device, amp),
        layers=list(ROUTE_LAYERS), epochs=EPOCHS, batch=BATCH, lr=LR,
        weight_decay=WEIGHT_DECAY, rank=RANK, ridge=RIDGE,
        joint_weight=JOINT_WEIGHT, distill_weight=DISTILL_WEIGHT,
        readout_weight=READOUT_WEIGHT, arms=list("ABCD"))
    output = Path(output).resolve()
    prior = {}
    if output.exists():
        if not resume or not (output / "run.json").is_file():
            raise FileExistsError(str(output))
        prior = json.loads((output / "run.json").read_text())
        if prior.get("signature") != signature:
            raise ValueError("Resume baseline, inputs, code, environment or protocol changed")
    else:
        output.mkdir(parents=True)
        _write_json_atomic({"signature": signature, "status": "running",
                            "official_test_opened": False}, output / "run.json")

    manifest, eeg, visual, train_rows, val_rows, visual_norms = _load_training(
        manifest_path, device)
    raw_text, caption_ids = _load_text(text_features, caption_ids_path, device)
    # Historical image targets retain 06b's train-partition normalization;
    # the added caption normalization is fit on optimizer rows only.
    text_mean = raw_text[train_rows].mean(0, keepdim=True)
    text_std = raw_text[train_rows].std(0, keepdim=True) + 1e-4
    text = (raw_text - text_mean) / text_std
    fit_indices = train_rows.detach().cpu().tolist()
    residualizer = Residualizer.fit(visual, text, fit_indices, rank=RANK,
                                   ridge=RIDGE, seed=0)
    residual = residualizer.transform(visual, text).to(device)
    n_in = len(POST17) * eeg.shape[2]
    dims = [FEATURE_WIDTH] * len(ROUTE_LAYERS)

    preprocessing_path = output / "preprocessing.pt"
    if resume and preprocessing_path.exists():
        if (prior.get("preprocessing") != preprocessing_path.name
                or prior.get("preprocessing_sha256") != _sha256(preprocessing_path)):
            raise ValueError("Tri-modal preprocessing artifact changed")
    elif resume and prior.get("preprocessing_sha256"):
        raise ValueError("Tri-modal preprocessing artifact missing")
    else:
        _torch_save_atomic({
            "model_dimensions": dict(n_in=n_in, dims=dims, text_dim=TEXT_WIDTH,
                                     rank=RANK, init_logits=list(ROUTE_INIT)),
            "eeg_normalization": "historical_z16_no_additional_fit",
            "eeg_source_channels": list(POST17_INDICES),
            "text_mean": text_mean.detach().cpu(),
            "text_std": text_std.detach().cpu(),
            "visual_mean": [mean.cpu() for mean, _ in visual_norms],
            "visual_std": [std.cpu() for _, std in visual_norms],
            "residualizer": residualizer.state_dict(),
            "fit_rows": train_rows.cpu(), "validation_rows": val_rows.cpu(),
        }, preprocessing_path)
    run_state = json.loads((output / "run.json").read_text())
    run_state.update(preprocessing=preprocessing_path.name,
                     preprocessing_sha256=_sha256(preprocessing_path))
    _write_json_atomic(run_state, output / "run.json")

    seed_all(initial_seed)
    initial = JointAligner(n_in, dims, TEXT_WIDTH, rank=RANK,
                           init_logits=list(ROUTE_INIT))
    initial.load_06b_backbone_state(backbone_state)
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in initial.state_dict().items()}

    concept_rows = [index // PER_CONCEPT for index in fit_indices]
    from .joint_data import derangement
    permutation = derangement(concept_rows, 7000)
    shuffled = residual.clone()
    fit_tensor = torch.as_tensor(fit_indices, dtype=torch.long, device=residual.device)
    shuffled[fit_tensor] = residual[fit_tensor[permutation.to(fit_tensor.device)]]
    generator = torch.Generator(device="cpu").manual_seed(8000)
    pseudo = torch.randn(residual.shape, generator=generator, dtype=residual.dtype)
    pseudo *= residual[fit_tensor].std(0, correction=0).clamp_min(1e-6).cpu()
    pseudo = pseudo.to(residual.device)
    semantic = {"A": None, "B": residual, "C": pseudo, "D": shuffled}
    run_records = []
    records_path = output / "results.csv"
    if resume and records_path.exists():
        with records_path.open(newline="") as handle:
            run_records = list(csv.DictReader(handle))
        done = [(row.get("subject"), int(row.get("seed", -1)), row.get("arm"))
                for row in run_records]
        if len(set(done)) != len(done) or any(
                item not in {("sub-01", 0, arm) for arm in "ABCD"} for item in done):
            raise ValueError("Existing A/B/C/D results have duplicate or unknown runs")
    completed = {(row["subject"], int(row["seed"]), row["arm"])
                 for row in run_records}

    try:
        for record in run_records:
            _verify_completed_artifacts(output, record)
        for arm in "ABCD":
            run_key = ("sub-01", 0, arm)
            if run_key in completed:
                continue
            model = JointAligner(n_in, dims, TEXT_WIDTH, rank=RANK,
                                 init_logits=list(ROUTE_INIT)).to(device)
            model.load_state_dict(initial_state, strict=True)
            seed_all(initial_seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                          weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=LR, total_steps=EPOCHS * math.ceil(
                    len(train_rows) / BATCH), pct_start=.05)
            best_score, best_epoch, best_state = -1., 0, None
            run_name = f"sub-01_s0_{arm}"
            latest_path = output / f"{run_name}_latest.pt"
            first_epoch = 1
            if resume and latest_path.exists():
                saved = torch.load(latest_path, map_location="cpu", weights_only=True)
                if saved.get("signature") != signature or saved.get("run_key") != list(run_key):
                    raise ValueError(f"Resume checkpoint mismatch for arm {arm}")
                model.load_state_dict(saved["model"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                best_score, best_epoch = float(saved["best_score"]), int(saved["best_epoch"])
                best_state = saved["best_state"]
                first_epoch = int(saved["epoch"]) + 1
                torch.set_rng_state(saved["torch_rng"].cpu())
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
            for epoch in range(first_epoch, EPOCHS + 1):
                model.train()
                permutation_rows = train_rows[torch.randperm(len(train_rows), device=device)]
                losses = []
                epoch_started = time.monotonic()
                for start in range(0, len(permutation_rows), BATCH):
                    rows = permutation_rows[start:start + BATCH]
                    if len(rows) < 2:
                        continue
                    eeg_batch = eeg[rows].float()
                    image_batch = [feature[rows] for feature in visual]
                    text_batch = text[rows]
                    identity = [str(value) for value in rows.detach().cpu().tolist()]
                    caption_batch_ids = [caption_ids[value] for value in rows.detach().cpu().tolist()]
                    with _autocast(device, amp):
                        ze = model.backbone.eeg(eeg_batch)
                        zi = model.backbone.img(image_batch)
                    # Autocast covers only the historical backbone. The joint
                    # and text heads below retain float32 weights, so their
                    # inputs must leave the mixed-precision region as float32.
                    ze, zi = ze.float(), zi.float()
                    if arm == "A":
                        loss = clip_loss(ze.float(), zi.float(), model.backbone.scale())
                    else:
                        terms = model.loss_terms(
                            ze, zi, text_batch, semantic[arm][rows], identity,
                            joint_weight=JOINT_WEIGHT, distill_weight=DISTILL_WEIGHT,
                            caption_ids=caption_batch_ids)
                        loss = terms["total"]
                    loss = loss + READOUT_WEIGHT * symmetric_identity_nce(
                        model.text_scores(ze.detach(), text_batch),
                        caption_batch_ids, caption_batch_ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss in arm {arm}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    main = [parameter for name, parameter in model.named_parameters()
                            if not name.startswith(("readout_", "text_log_scale"))]
                    student = [parameter for name, parameter in model.named_parameters()
                               if name.startswith(("readout_", "text_log_scale"))]
                    torch.nn.utils.clip_grad_norm_(main, 1., error_if_nonfinite=True)
                    torch.nn.utils.clip_grad_norm_(student, 1., error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    losses.append(float(loss.detach()))
                score = None
                tune_text_top1 = None
                if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
                    metrics, _ = _validation_metrics(
                        model, eeg, visual, text, val_rows, caption_ids, device, BATCH)
                    score = metrics["image_top1"]
                    tune_text_top1 = metrics["text_top1"]
                    if score > best_score:
                        best_score, best_epoch = score, epoch
                        best_state = {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}
                epoch_record = dict(run=run_name, epoch=epoch, loss=float(np.mean(losses)),
                                    tune_image_top1=score,
                                    tune_text_top1=tune_text_top1,
                                    seconds=time.monotonic() - epoch_started)
                _write_epoch_log(epoch_record, output / "epochs.jsonl")
                _torch_save_atomic({
                    "signature": signature, "run_key": list(run_key),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_score": best_score, "best_epoch": best_epoch,
                    "best_state": best_state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda" else [],
                }, latest_path)
                print(json.dumps(epoch_record), flush=True)
            if best_state is None:
                raise RuntimeError(f"No image-validation checkpoint for arm {arm}")
            model.load_state_dict(best_state, strict=True)
            metrics, scores = _validation_metrics(
                model, eeg, visual, text, val_rows, caption_ids, device, BATCH)
            scores_path = output / f"{run_name}_validation_scores.pt"
            best_path = output / f"{run_name}_best.pt"
            _torch_save_atomic(scores, scores_path)
            _torch_save_atomic(best_state, best_path)
            record = dict(subject="sub-01", seed=0, arm=arm,
                          setting="historical_06b_warm_start", role="training_validation",
                          candidates=len(val_rows),
                          unique_caption_candidates=len(set(
                              caption_ids[index] for index in val_rows.cpu().tolist())),
                          selected_epoch=best_epoch, parameters=sum(
                              parameter.numel() for parameter in model.parameters()),
                          backbone_parameters=sum(
                              parameter.numel() for parameter in model.backbone.parameters()),
                          checkpoint=best_path.name, checkpoint_sha256=_sha256(best_path),
                          validation_scores=scores_path.name,
                          validation_scores_sha256=_sha256(scores_path),
                          **metrics)
            run_records.append(record)
            completed.add(run_key)
            _atomic_results(run_records, records_path)
        result = {
            "signature": signature,
            "status": "complete_training_only",
            "official_test_opened": False,
            "baseline_checkpoint": str(baseline_path),
            "preprocessing": preprocessing_path.name,
            "preprocessing_sha256": _sha256(preprocessing_path),
            "subject": "sub-01",
            "seed": 0,
            "layers": list(ROUTE_LAYERS),
            "arms": run_records,
            "caption_provenance": text_report["provenance"],
            "text_normalization_fit": "optimizer/training rows only",
            "residualizer_fit": "optimizer/training rows only",
            "direct_text_inference": "EEG and candidate text only; no image candidate features",
        }
        _write_json_atomic(result, output / "run.json")
        return run_records
    except Exception as error:
        run_state = json.loads((output / "run.json").read_text())
        run_state.update(status="incomplete",
                         error=type(error).__name__ + ": " + str(error))
        _write_json_atomic(run_state, output / "run.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--caption-ids", required=True)
    parser.add_argument("--caption-provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, args.baseline_output, args.text_features,
                 args.caption_ids, args.caption_provenance, args.output,
                 args.device, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
