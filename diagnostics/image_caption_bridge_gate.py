"""CPU-only, read-only Image->Caption feasibility gate for the historical 06b model.

Fits exactly one fixed-regularization ridge map from frozen 06b InternViT
image embeddings to existing train-only caption embeddings. Real pairing is
compared with cross-concept shuffled pairing on 1,654-image development data.
Also reports an internal held-out-concept check using the same gallery.

No GPU, no model training, no writes, and no official test arrays.
Run from the pinned repo: python diagnostics/image_caption_bridge_gate.py
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RIDGE_RATIO = 0.05  # Fixed a priori; no tuning on validation.
CONTROL_SEED = 7000
BOOTSTRAP_SEED = 20261121


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_bridge(images, captions, train_rows, permutation=None):
    """Affine ridge bridge, with means and ridge strength learned on fit rows."""
    if images.ndim != 2 or captions.ndim != 2 or len(images) != len(captions):
        raise ValueError("Image/caption row shape mismatch")
    fit = torch.as_tensor(train_rows, dtype=torch.long)
    if fit.ndim != 1 or len(fit) < 2 or len(torch.unique(fit)) != len(fit):
        raise ValueError("Training rows must be unique")
    if fit.min() < 0 or fit.max() >= len(images):
        raise ValueError("Training index outside cache")
    x = F.normalize(images[fit].float(), dim=1)
    y = F.normalize(captions[fit].float(), dim=1)
    if permutation is not None:
        p = torch.as_tensor(permutation, dtype=torch.long)
        if p.shape != (len(fit),) or sorted(p.tolist()) != list(range(len(fit))):
            raise ValueError("Shuffled control must permute fit rows only")
        y = y[p]
    mx, my = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    x, y = x - mx, y - my
    gram = x.T @ x
    strength = max(RIDGE_RATIO * float(gram.trace()) / x.shape[1], 1e-8)
    gram.diagonal().add_(strength)
    weights = torch.linalg.solve(gram, x.T @ y)
    return weights, mx, my


def bridge_scores(images, captions, fitted, query_rows, gallery_rows):
    weights, mx, my = fitted
    query = F.normalize(images[query_rows].float(), dim=-1)
    query = F.normalize((query - mx) @ weights + my, dim=-1)
    target = F.normalize(captions[gallery_rows].float(), dim=-1)
    return query @ target.T


def caption_hits(scores, query_ids, gallery_ids, image_ids):
    """Same duplicate-caption positive and stable tie semantics as 06c."""
    if scores.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("Caption score shape mismatch")
    if len(image_ids) != len(gallery_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate or misaligned image IDs")
    if not set(query_ids) <= set(gallery_ids) or not torch.isfinite(scores).all():
        raise ValueError("Missing positive caption or nonfinite score")
    order = sorted(range(len(gallery_ids)), key=lambda i: (str(gallery_ids[i]), str(image_ids[i])))
    ordered = [gallery_ids[j] for j in order]
    ranked = scores[:, order].argsort(dim=1, descending=True, stable=True)
    lookup = {cid: i for i, cid in enumerate(sorted(set(ordered)))}
    pred = torch.tensor([lookup[t] for t in ordered], dtype=torch.long)[ranked]
    target = torch.tensor([lookup[t] for t in query_ids], dtype=torch.long)[:, None]
    match = pred == target
    return match[:, :1].any(1).float(), match[:, :5].any(1).float()


def bootstrap_ci(a, b, seed):
    diff = (a.float() - b.float()).cpu().numpy()
    generator = np.random.default_rng(seed)
    sample = np.array([diff[generator.integers(0, len(diff), len(diff))].mean() * 100
                       for _ in range(1000)])
    return [round(float(x), 4) for x in np.quantile(sample, [0.025, 0.975])]


def evaluate(images, text, caption_ids, fit, query, gallery, seed):
    from phase1.joint_data import derangement
    fit, query, gallery = [torch.as_tensor(x, dtype=torch.long) for x in (fit, query, gallery)]
    if set(fit.tolist()) & set(query.tolist()):
        raise ValueError("Train/validation overlap")
    permutation = derangement((fit // 10).tolist(), seed)
    real_map = fit_bridge(images, text, fit)
    null_map = fit_bridge(images, text, fit, permutation)
    qi, gi = query.tolist(), gallery.tolist()
    ids = [caption_ids[i] for i in qi]
    gallery_ids = [caption_ids[i] for i in gi]
    a1, a5 = caption_hits(bridge_scores(images, text, real_map, query, gallery), ids, gallery_ids, gi)
    b1, b5 = caption_hits(bridge_scores(images, text, null_map, query, gallery), ids, gallery_ids, gi)
    return {
        "queries": len(query), "gallery": len(gallery), "fit_rows": len(fit),
        "unique_gallery_captions": len(set(gallery_ids)),
        "real_top1": round(float(a1.mean()*100), 5),
        "real_top5": round(float(a5.mean()*100), 5),
        "shuffled_top1": round(float(b1.mean()*100), 5),
        "shuffled_top5": round(float(b5.mean()*100), 5),
        "real_minus_shuffle_top1_pp": round(float((a1-b1).mean()*100), 5),
        "real_minus_shuffle_top5_pp": round(float((a5-b5).mean()*100), 5),
        "paired_top1_delta_95ci_pp": bootstrap_ci(a1, b1, seed + 111),
        "paired_top5_delta_95ci_pp": bootstrap_ci(a5, b5, seed + 222),
    }


def run(run_root, cache):
    from phase1.historical_06b import (_load_training, POST17, ROUTE_INIT,
                                       ROUTE_LAYERS, FEATURE_WIDTH, TRAIN_IMAGES)
    from phase1.models import Aligner
    from phase1.trimodal_06b import _load_text, preflight_text_inputs
    started = time.monotonic()
    manifest = run_root / "train_only_manifest.json"
    ckpt = run_root / "baseline/within_route_c_sub-01_s0_best.pt"
    record = json.loads((run_root / "baseline/run.json").read_text())
    m = json.loads(manifest.read_text())
    if (m.get("partition") != "official_training" or m.get("official_test_opened") is not False
            or len(m.get("feature_parts", [])) != 9):
        raise ValueError("Only historical train-only image features are permitted")
    if (record.get("status") != "complete_training_only"
            or record.get("checkpoint") != ckpt.name
            or record.get("checkpoint_sha256") != sha256(ckpt)):
        raise ValueError("Historical 06b baseline checkpoint provenance mismatch")
    captions_path = cache / "captions_train_only.npy"
    ids_path = cache / "caption_ids_train_only.json"
    prov_path = cache / "caption_provenance_train_only.json"
    report = preflight_text_inputs(captions_path, ids_path, prov_path)
    if report.get("train_only") is not True:
        raise ValueError("Caption data are not train only")
    # Historical normalizers include development images: this is NOT official test.
    _, eeg, layers, fit, val, _ = _load_training(manifest, torch.device("cpu"))
    model = Aligner(len(POST17) * eeg.shape[2],
                    [FEATURE_WIDTH] * len(ROUTE_LAYERS), list(ROUTE_INIT))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    with torch.no_grad():
        image = torch.cat([model.img([v[ix] for v in layers]).float()
                           for ix in torch.arange(TRAIN_IMAGES).split(256)], 0)
    del layers, eeg, model
    captions, caption_ids = _load_text(captions_path, ids_path, torch.device("cpu"))
    if (len(image) != len(caption_ids) or len(captions) != len(image)
            or not torch.isfinite(image).all()):
        raise ValueError("Image/text row count or finiteness mismatch")
    mean = captions[fit].mean(0, keepdim=True)
    std = captions[fit].std(0, keepdim=True) + 1e-4
    text = (captions - mean) / std
    fit, val = fit.cpu(), val.cpu()
    if (len(fit) != 14886 or len(val) != 1654
            or (fit % 10 == 9).any() or not (val % 10 == 9).all()):
        raise ValueError("Expected 9/1 historical split per concept")
    seen = evaluate(image, text, caption_ids, fit, val, val, CONTROL_SEED)

    # Additional concept-held-out check: fit on 90% of concepts, query on
    # unseen 10%, but retain the same 1,654-candidate caption gallery.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    held = torch.tensor(np.sort(rng.choice(1654, size=165, replace=False)))
    inner_fit = fit[~torch.isin(fit//10, held)]
    inner_val = val[torch.isin(val//10, held)]
    unseen = evaluate(image, text, caption_ids, inner_fit, inner_val,
                      val, CONTROL_SEED+1)
    return {
        "status": "measured_cpu_only", "official_test_opened": False,
        "scope": "sub-01 seed-0, 1,654-candidate training-partition validation",
        "baseline_checkpoint_sha256": record["checkpoint_sha256"],
        "manifest_sha256": sha256(manifest),
        "caption_ids_sha256": sha256(ids_path),
        "fixed_ridge_ratio": RIDGE_RATIO,
        "seen_concept_dev": seen,
        "heldout_concept_diagnostic": unseen,
        "cpu_wall_seconds": round(time.monotonic()-started, 2),
        "limitations": ("Historical image normalizer includes dev rows; "
                        "EEG-image-caption row identity remains independently unverified; "
                        "not an official 200-way / 10-subject test."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec")
    parser.add_argument("--cache", default="/content/drive/MyDrive/eeg_results/cache")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    print(json.dumps(run(Path(args.root), Path(args.cache)), indent=2), flush=True)


if __name__ == "__main__":
    main()
