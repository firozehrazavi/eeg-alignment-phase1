"""Train-only, resumable reproduction of the historical 06b route_c recipe.

The loader reads only sub-01_train_z16.npy and explicitly named InternViT
train chunks. It never opens the combined feature cache or any test artifact.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core import clip_loss
from .models import Aligner, EMB

TRAIN_IMAGES = 16540
TRAIN_CONCEPTS = 1654
PER_CONCEPT = 10
FEATURE_CHUNK = 2048
FEATURE_WIDTH = 3200
FEATURE_LAYERS = (12, 16, 20, 24, 28, 32, 36)
ROUTE_LAYERS = (20, 24, 28, 32, 36)
ROUTE_INIT = (-2., -1., 0., -1., -2.)
EPOCHS = 60
BATCH = 1024
LR = 1e-4
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 5

CH63 = (
    "Fp1 Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 F2 F4 F6 F8 FT9 FT7 FC5 FC3 FC1 "
    "FCz FC2 FC4 FC6 FT8 FT10 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP9 TP7 CP5 CP3 "
    "CP1 CPz CP2 CP4 CP6 TP8 TP10 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz "
    "PO4 PO8 O1 Oz O2"
).split()
POST17 = "P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
POST17_INDICES = tuple(CH63.index(channel) for channel in POST17)
PARTS = tuple(
    f"feat6_internvit6b_parts/train_{i:03d}.pt"
    for i in range(math.ceil(TRAIN_IMAGES / FEATURE_CHUNK))
)
SELECTED_KEYS = tuple((layer, "mean") for layer in ROUTE_LAYERS)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def _torch_save_atomic(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _resolve(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Input path escapes cache root: {relative}") from error
    if "test" in path.name.casefold() or "eval" in path.name.casefold():
        raise ValueError(f"Test/evaluation-looking input refused: {relative}")
    return path


def create_manifest(cache_root, manifest_path):
    """Hash and record the precise train-only files used by the 06b recipe."""
    root = Path(cache_root).resolve()
    eeg_name = "sub-01_train_z16.npy"
    names = (eeg_name, *PARTS)
    hashes = {}
    for name in names:
        path = _resolve(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Required historical 06b training input missing: {path}")
        hashes[name] = _sha256(path)
    manifest = {
        "cache_root": str(root),
        "protocol": "historical_06b_within_subject_route_c",
        "partition": "official_training",
        "subject": "sub-01",
        "seed": 0,
        "train_images": TRAIN_IMAGES,
        "train_concepts": TRAIN_CONCEPTS,
        "images_per_concept": PER_CONCEPT,
        "eeg_normalization": "historical_train_z16_cache; no additional EEG fit",
        "eeg_file": eeg_name,
        "feature_parts": list(PARTS),
        "feature_chunk_size": FEATURE_CHUNK,
        "feature_layers_available": list(FEATURE_LAYERS),
        "feature_pool": "mean",
        "route_layers": list(ROUTE_LAYERS),
        "route_init_logits": list(ROUTE_INIT),
        "eeg_channels": list(CH63),
        "selected_channels": list(POST17),
        "feature_width": FEATURE_WIDTH,
        "source_hashes": hashes,
        "official_test_opened": False,
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def _expected_part_rows(index):
    start = index * FEATURE_CHUNK
    return min(FEATURE_CHUNK, TRAIN_IMAGES - start)


def preflight(manifest_path):
    """Scan/hashes inputs on CPU; only training EEG/features are opened."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    fixed = {
        "protocol": "historical_06b_within_subject_route_c",
        "partition": "official_training",
        "subject": "sub-01",
        "seed": 0,
        "train_images": TRAIN_IMAGES,
        "train_concepts": TRAIN_CONCEPTS,
        "images_per_concept": PER_CONCEPT,
        "route_layers": list(ROUTE_LAYERS),
        "route_init_logits": list(ROUTE_INIT),
        "feature_width": FEATURE_WIDTH,
        "eeg_file": "sub-01_train_z16.npy",
        "feature_parts": list(PARTS),
        "feature_chunk_size": FEATURE_CHUNK,
        "official_test_opened": False,
    }
    for key, value in fixed.items():
        if manifest.get(key) != value:
            raise ValueError(f"Historical 06b manifest mismatch for {key}")
    if manifest.get("selected_channels") != list(POST17) or manifest.get("feature_pool") != "mean":
        raise ValueError("Historical 06b channel or pooling protocol mismatch")
    if (manifest.get("eeg_channels") != CH63
            or manifest.get("feature_layers_available") != list(FEATURE_LAYERS)):
        raise ValueError("Historical 06b source channel order or feature cache layer mismatch")
    if manifest.get("eeg_normalization") != "historical_train_z16_cache; no additional EEG fit":
        raise ValueError("Historical 06b EEG preprocessing declaration mismatch")
    root = Path(manifest.get("cache_root", manifest_path.parent)).resolve()
    hashes = manifest.get("source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {manifest["eeg_file"], *PARTS}:
        raise ValueError("Incomplete train-only input hashes")
    eeg_path = _resolve(root, manifest["eeg_file"])
    if not eeg_path.is_file() or _sha256(eeg_path) != hashes[manifest["eeg_file"]]:
        raise ValueError("Missing or changed historical training EEG cache")
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    if (eeg.ndim != 3 or eeg.shape[:2] != (TRAIN_IMAGES, len(CH63))
            or eeg.shape[2] < 1 or eeg.dtype != np.float16):
        raise ValueError(f"Expected float16 EEG [{TRAIN_IMAGES}, 63, time], got {eeg.shape}/{eeg.dtype}")
    for start in range(0, TRAIN_IMAGES, 1024):
        if not np.isfinite(eeg[start:start + 1024]).all():
            raise ValueError("Nonfinite historical training EEG")
    time_points = int(eeg.shape[2])
    del eeg

    part_shapes = []
    for index, name in enumerate(PARTS):
        path = _resolve(root, name)
        if not path.is_file() or _sha256(path) != hashes[name]:
            raise ValueError(f"Missing or changed historical training feature chunk: {name}")
        values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict) or not set(SELECTED_KEYS) <= set(values):
            raise ValueError(f"Missing route_c mean-pooled layers in {name}")
        rows = _expected_part_rows(index)
        for key in SELECTED_KEYS:
            tensor = values[key]
            if (not torch.is_tensor(tensor) or tuple(tensor.shape) != (rows, FEATURE_WIDTH)
                    or tensor.dtype != torch.float16 or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid train feature tensor {key} in {name}")
        part_shapes.append(rows)
        del values
    if sum(part_shapes) != TRAIN_IMAGES:
        raise ValueError("Training feature chunks do not cover the full training partition")
    fit_rows = TRAIN_IMAGES - TRAIN_CONCEPTS
    model = Aligner(len(POST17) * time_points, [FEATURE_WIDTH] * len(ROUTE_LAYERS),
                    list(ROUTE_INIT))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    updates_per_epoch = math.ceil(fit_rows / BATCH)
    return {
        "status": "cpu_preflight_passed",
        "partition": "official_training_only",
        "official_test_opened": False,
        "subject": "sub-01",
        "seed": 0,
        "train_images": TRAIN_IMAGES,
        "validation_images": TRAIN_CONCEPTS,
        "fit_images_per_epoch": fit_rows,
        "eeg_shape": [TRAIN_IMAGES, len(CH63), time_points],
        "selected_eeg_channels": list(POST17),
        "route_layers": list(ROUTE_LAYERS),
        "feature_shape_per_layer": [TRAIN_IMAGES, FEATURE_WIDTH],
        "target_pool": "mean",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "updates_per_epoch": updates_per_epoch,
        "total_optimizer_updates": EPOCHS * updates_per_epoch,
        "validation_epochs": list(range(EVAL_EVERY, EPOCHS + 1, EVAL_EVERY)),
        "model_parameters": parameter_count,
        "fp32_adam_parameter_state_bytes_lower_bound": parameter_count * 16,
        "fp32_targets_device_bytes": TRAIN_IMAGES * FEATURE_WIDTH * len(ROUTE_LAYERS) * 4,
        "input_sha256": hashes,
        "manifest_sha256": _sha256(manifest_path),
    }


def _load_training(manifest_path, device):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest.get("cache_root", manifest_path.parent)).resolve()
    eeg_array = np.load(_resolve(root, manifest["eeg_file"]), mmap_mode="r", allow_pickle=False)
    eeg_np = np.ascontiguousarray(eeg_array[:, POST17_INDICES, :])
    eeg = torch.from_numpy(eeg_np).to(device)
    del eeg_array, eeg_np

    layers = []
    normalizers = []
    for key in SELECTED_KEYS:
        pieces = []
        for name in PARTS:
            part = torch.load(_resolve(root, name), map_location="cpu", weights_only=True)
            pieces.append(part[key])
            del part
        raw = torch.cat(pieces).float()
        mean = raw.mean(0, keepdim=True)
        std = raw.std(0, keepdim=True) + 1e-4
        normalized = ((raw - mean) / std).to(device)
        layers.append(normalized)
        normalizers.append((mean, std))
        del pieces, raw

    rows = torch.arange(TRAIN_IMAGES, device=device)
    validation = (rows % PER_CONCEPT) == (PER_CONCEPT - 1)
    val_rows = rows[validation]
    train_rows = rows[~validation]
    return manifest, eeg, layers, train_rows, val_rows, normalizers


def _autocast(device, enabled):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)
    return nullcontext()


@torch.no_grad()
def _validation_top1(model, eeg, targets, val_rows, device, amp):
    model.eval()
    eeg_embeddings = []
    for start in range(0, len(val_rows), 4096):
        indices = val_rows[start:start + 4096]
        with _autocast(device, amp):
            projected = model.eeg(eeg[indices].float())
        eeg_embeddings.append(F.normalize(projected.float(), dim=-1))
    image = model.img([target[val_rows] for target in targets])
    image = F.normalize(image.float(), dim=-1)
    scores = torch.cat(eeg_embeddings) @ image.T
    ranks = (scores > scores.diag().unsqueeze(1)).sum(1)
    return float((ranks < 1).float().mean().item() * 100)


def _encode_numpy_rng(state=None):
    state = np.random.get_state() if state is None else state
    return {
        "name": state[0],
        "state": torch.from_numpy(state[1].copy()),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_rng(saved, device):
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state["name"], state["state"].cpu().numpy().astype(np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def _write_epoch_log(record, path):
    path = Path(path)
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [entry for entry in records
               if (entry["epoch"], entry["run"]) != (record["epoch"], record["run"])]
    records.append(record)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(entry) + "\n" for entry in records))
    os.replace(temporary, path)


def _code_hashes():
    root = Path(__file__).parent
    return {name: _sha256(root / name) for name in ("historical_06b.py", "models.py", "core.py")}


def run_baseline(manifest_path, output, device_name="cuda", resume=False,
                 max_seconds=None):
    """Train exact 06b route_c on training images and select by training validation."""
    report = preflight(manifest_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(output).resolve()
    manifest_sha = report["manifest_sha256"]
    signature = dict(manifest_sha256=manifest_sha, code_hashes=_code_hashes(),
                     python=str(platform.python_version()), torch=str(torch.__version__),
                     numpy=str(np.__version__), subject="sub-01", seed=0,
                     layers=list(ROUTE_LAYERS), init_logits=list(ROUTE_INIT),
                     epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY,
                     eval_every=EVAL_EVERY)
    if output.exists():
        if not resume:
            raise FileExistsError(str(output))
        state_path = output / "run.json"
        if not state_path.is_file():
            raise ValueError("Cannot resume without run.json")
        previous = json.loads(state_path.read_text())
        if previous.get("signature") != signature:
            raise ValueError("Resume inputs, code, environment, or baseline protocol changed")
    else:
        output.mkdir(parents=True)
        _write_json_atomic({"signature": signature, "status": "running",
                            "official_test_opened": False}, output / "run.json")

    manifest, eeg, targets, train_rows, val_rows, normalizers = _load_training(manifest_path, device)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    amp = (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
    model = Aligner(len(POST17) * eeg.shape[2], [FEATURE_WIDTH] * len(ROUTE_LAYERS),
                    list(ROUTE_INIT)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=EPOCHS * math.ceil(len(train_rows) / BATCH),
        pct_start=0.05)
    best_score, best_epoch, best_state = -1., 0, None
    first_epoch = 1
    latest_path = output / "within_route_c_sub-01_s0_latest.pt"
    if resume and latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("signature") != signature:
            raise ValueError("Latest checkpoint signature mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_score, best_epoch = float(saved["best_score"]), int(saved["best_epoch"])
        best_state = saved["best_state"]
        first_epoch = int(saved["epoch"]) + 1
        _restore_rng(saved, device)
    if first_epoch > EPOCHS + 1:
        raise ValueError("Latest checkpoint epoch exceeds configured schedule")

    start_time = time.monotonic()
    run_name = "within_route_c_sub-01_s0"
    try:
        for epoch in range(first_epoch, EPOCHS + 1):
            if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                raise TimeoutError("Experiment wall-time budget exhausted at epoch boundary")
            model.train()
            permutation = train_rows[torch.randperm(len(train_rows), device=device)]
            losses = []
            epoch_started = time.monotonic()
            for start in range(0, len(permutation), BATCH):
                if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                    raise TimeoutError("Experiment wall-time budget exhausted; latest epoch is resumable")
                batch_rows = permutation[start:start + BATCH]
                if len(batch_rows) < 2:
                    continue
                with _autocast(device, amp):
                    ze = model.eeg(eeg[batch_rows].float())
                    zi = model.img([target[batch_rows] for target in targets])
                loss = clip_loss(ze.float(), zi.float(), model.scale())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            score = None
            if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
                score = _validation_top1(model, eeg, targets, val_rows, device, amp)
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}
            entry = dict(run=run_name, epoch=epoch, loss=float(np.mean(losses)),
                         validation_top1=score, seconds=time.monotonic() - epoch_started)
            _write_epoch_log(entry, output / "epochs.jsonl")
            latest = dict(signature=signature, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          epoch=epoch, best_score=best_score, best_epoch=best_epoch,
                          best_state=best_state, python_rng=random.getstate(),
                          numpy_rng=_encode_numpy_rng(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            _torch_save_atomic(latest, latest_path)
            print(json.dumps(entry), flush=True)
        if best_state is None:
            raise RuntimeError("Historical route_c training produced no validation checkpoint")
        best_path = output / "within_route_c_sub-01_s0_best.pt"
        _torch_save_atomic(best_state, best_path)
        final_model = Aligner(len(POST17) * eeg.shape[2],
                              [FEATURE_WIDTH] * len(ROUTE_LAYERS), list(ROUTE_INIT)).to(device)
        final_model.load_state_dict(best_state, strict=True)
        run_record = {
            "signature": signature,
            "status": "complete_training_only",
            "official_test_opened": False,
            "selected_epoch": best_epoch,
            "validation_top1": best_score,
            "subject": "sub-01",
            "seed": 0,
            "layers": list(ROUTE_LAYERS),
            "pool": "mean",
            "channels": list(POST17),
            "n_in": len(POST17) * eeg.shape[2],
            "feature_dims": [FEATURE_WIDTH] * len(ROUTE_LAYERS),
            "target_mean": [mean.tolist() for mean, _ in normalizers],
            "target_std": [std.tolist() for _, std in normalizers],
            "validation_rows": val_rows.detach().cpu().tolist(),
            "fit_rows": train_rows.detach().cpu().tolist(),
            "checkpoint": best_path.name,
            "checkpoint_keys": list(best_state),
            "checkpoint_sha256": _sha256(best_path),
            "input_sha256": manifest["source_hashes"],
        }
        _write_json_atomic(run_record, output / "run.json")
        return run_record
    except Exception as error:
        existing = json.loads((output / "run.json").read_text())
        existing.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
        _write_json_atomic(existing, output / "run.json")
        raise


def profile_cpu(manifest_path, batch_size=32):
    """Measure one representative CPU update and report a linear planning bound."""
    if batch_size < 2:
        raise ValueError("CPU profile batch_size must be at least two")
    report = preflight(manifest_path)
    manifest, eeg, targets, train_rows, _, _ = _load_training(manifest_path, torch.device("cpu"))
    model = Aligner(len(POST17) * eeg.shape[2], [FEATURE_WIDTH] * len(ROUTE_LAYERS),
                    list(ROUTE_INIT))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ids = train_rows[:batch_size]
    timings = []
    for _ in range(2):
        ze = model.eeg(eeg[ids].float())
        zi = model.img([target[ids] for target in targets])
        loss = clip_loss(ze.float(), zi.float(), model.scale())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        timings.append(time.perf_counter())
    seconds_per_step = timings[-1] - timings[-2]
    return {
        **report,
        "profile": "one actual CPU optimizer update after one warm-up update",
        "profile_batch_size": batch_size,
        "profile_step_seconds": seconds_per_step,
        "linear_cpu_hours_for_full_schedule": seconds_per_step
        * report["total_optimizer_updates"] / 3600,
        "device": "cpu",
        "official_test_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("manifest")
    prepare.add_argument("--cache-root", required=True)
    prepare.add_argument("--output", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--manifest", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--manifest", required=True)
    profile.add_argument("--batch-size", type=int, default=32)
    train = subparsers.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.command == "manifest":
        result = create_manifest(args.cache_root, args.output)
    elif args.command == "preflight":
        result = preflight(args.manifest)
    elif args.command == "profile":
        result = profile_cpu(args.manifest, args.batch_size)
    else:
        result = run_baseline(args.manifest, args.output, args.device,
                              args.resume, args.max_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
