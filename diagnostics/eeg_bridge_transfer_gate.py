"""Frozen 06b EEG -> image-anchored Ridge -> caption. CPU only, read-only.

One new diagnostic, not a new model/experiment. Reuses exactly the verified
Image->Caption Ridge fit, caption normalization, IDs and scoring semantics.
No GPU, no training, no saved artifacts, no official test exposure.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.image_caption_bridge_gate import (
    CONTROL_SEED, fit_bridge, caption_hits, bootstrap_ci, sha256)


def bridge_scores_from_embeddings(query, captions, fit):
    weights, mean_image, mean_caption = fit
    query = F.normalize(query.float(), dim=-1)
    bridge = F.normalize((query - mean_image) @ weights + mean_caption, dim=-1)
    return bridge @ F.normalize(captions.float(), dim=-1).T


def check(query, captions, model, identities, image_ids):
    scores = bridge_scores_from_embeddings(query, captions, model)
    top1, top5 = caption_hits(scores, identities, identities, image_ids)
    return {
        "top1_pct": round(float(top1.mean() * 100), 5),
        "top5_pct": round(float(top5.mean() * 100), 5),
        "hits_top1": int(top1.sum()),
        "hits_top5": int(top5.sum())
    }, top1, top5


def compare(eeg_dev, image_dev, text_dev, identities, image_ids, real, shuffled):
    if eeg_dev.shape != image_dev.shape or eeg_dev.ndim != 2 or eeg_dev.shape[1] != 1024:
        raise ValueError("06b EEG and projected-image embeddings must both be [1654,1024]")
    if len(eeg_dev) != 1654 or len(text_dev) != 1654 or len(identities) != 1654:
        raise ValueError("Expected identical 1,654-query/gallery validation rows")
    if not all(torch.isfinite(x).all() for x in (eeg_dev, image_dev, text_dev)):
        raise ValueError("Non-finite representations")
    z = F.normalize(eeg_dev, dim=-1)
    v = F.normalize(image_dev, dim=-1)
    image_correct = ((z @ v.T).argmax(dim=-1) == torch.arange(len(z)))
    oracle, _, _ = check(image_dev, text_dev, real, identities, image_ids)
    if abs(oracle["top1_pct"] - 44.80048) > 0.001:
        raise ValueError("Original 44.80048% Image->Caption reference not reproduced")
    result, a1, a5 = check(eeg_dev, text_dev, real, identities, image_ids)
    null, b1, b5 = check(eeg_dev, text_dev, shuffled, identities, image_ids)
    permutation = torch.as_tensor(np.random.default_rng(20261122).permutation(len(z)))
    shuffled_eeg, c1, _ = check(eeg_dev[permutation], text_dev, real, identities, image_ids)
    delta = float((a1 - b1).mean() * 100)
    return {
        "eeg_to_image_top1_pct": round(float(image_correct.float().mean()*100),5),
        "oracle_image_to_caption": oracle,
        "PRIMARY_frozen_eeg_to_caption": result,
        "CONTROL_shuffled_image_caption_bridge": null,
        "CONTROL_permuted_eeg_rows": shuffled_eeg,
        "real_minus_shuffled_bridge_pp": round(delta,5),
        "real_minus_shuffled_bridge_ci95_pp": bootstrap_ci(a1,b1,20261123),
        "real_minus_permuted_eeg_ci95_pp": bootstrap_ci(a1,c1,20261124),
        "top5_real_minus_shuffled_ci95_pp": bootstrap_ci(a5,b5,20261125),
        "mechanistic_gate": (
            "GO_for_small_architecture_study" if
            result["top1_pct"] >= 4.0 and
            delta > 0 and bootstrap_ci(a1,b1,20261123)[0] > 0 and
            null["top1_pct"] <= 0.3 else
            "NO_GO_or_inconclusive_for_architecture_change"
        )
    }


def direct_eeg_ridge_diagnostic(eeg_all, text_all, caption_ids, train, val,
                                image_bridge_map):
    """Fit caption readout on train EEG only; evaluate original 1,654-way dev.

    This is a diagnostic of information in frozen EEG, NOT joint learning.
    """
    from phase1.joint_data import derangement

    if len(eeg_all) != 16540 or len(text_all) != 16540:
        raise ValueError("Direct EEG readout requires all 16,540 train-partition rows")
    if len(train) != 14886 or len(val) != 1654:
        raise ValueError("Expected historical 9/1 split; no official test")
    if not torch.isfinite(eeg_all).all():
        raise ValueError("Non-finite EEG embeddings")

    ids = [caption_ids[int(i)] for i in val]
    rows = [int(i) for i in val]

    direct = fit_bridge(eeg_all, text_all, train)
    shuffled = fit_bridge(
        eeg_all, text_all, train,
        derangement((train // 10).tolist(), CONTROL_SEED))

    real_scores = bridge_scores_from_embeddings(eeg_all[val], text_all[val], direct)
    shuf_scores = bridge_scores_from_embeddings(eeg_all[val], text_all[val], shuffled)
    image_scores = bridge_scores_from_embeddings(
        image_bridge_map[0], text_all[val], image_bridge_map[1])

    a1, a5 = caption_hits(real_scores, ids, ids, rows)
    b1, b5 = caption_hits(shuf_scores, ids, ids, rows)
    v1, v5 = caption_hits(image_scores, ids, ids, rows)

    rng = np.random.default_rng(20261122)
    permuted_scores = real_scores[torch.as_tensor(rng.permutation(1654))]
    p1, p5 = caption_hits(permuted_scores, ids, ids, rows)

    def summarize(top1, top5):
        return {
            "top1_pct": round(float(top1.mean() * 100), 5),
            "top5_pct": round(float(top5.mean() * 100), 5),
            "hits_top1": int(top1.sum()),
            "hits_top5": int(top5.sum()),
        }

    # Diagnostic held-out concepts for the *readout only*: the frozen EEG
    # backbone itself was trained on these concepts; not concept zero-shot.
    rng2 = np.random.default_rng(20261121)
    held_concepts = torch.as_tensor(np.sort(rng2.choice(1654, size=165, replace=False)))
    inner_train = train[~torch.isin(train // 10, held_concepts)]
    inner_query = val[torch.isin(val // 10, held_concepts)]
    inner_direct = fit_bridge(eeg_all, text_all, inner_train)
    inner_s = bridge_scores_from_embeddings(eeg_all[inner_query], text_all[val], inner_direct)
    inner_ids = [caption_ids[int(i)] for i in inner_query]
    held_1, held_5 = caption_hits(inner_s, inner_ids, ids, rows)

    low_real_vs_shuf = bootstrap_ci(a1, b1, 20261130)[0]
    low_real_vs_img = bootstrap_ci(a1, v1, 20261131)[0]
    primary = summarize(a1, a5)
    result = {
        "PRIMARY_frozen_eeg_trained_ridge_to_caption": primary,
        "CONTROL_shuffled_eeg_caption_train_pairs": summarize(b1, b5),
        "CONTROL_permuted_eeg_dev_queries": summarize(p1, p5),
        "REFERENCE_image_trained_bridge_on_same_frozen_eeg": summarize(v1, v5),
        "readout_heldout_165_concepts_1654_way": summarize(held_1, held_5),
        "real_minus_shuffle_top1_ci95_pp": bootstrap_ci(a1, b1, 20261130),
        "real_minus_image_bridge_top1_ci95_pp": bootstrap_ci(a1, v1, 20261131),
        "real_minus_image_bridge_top5_ci95_pp": bootstrap_ci(a5, v5, 20261132),
        "heldout_concept_readout_fit_rows": len(inner_train),
        "pre_registered_decision": (
            "GO_for_joint_three_way_hypothesis_only" if
            primary["top1_pct"] >= 4.0 and
            low_real_vs_shuf > 0 and
            low_real_vs_img > 0 and
            primary["top5_pct"] > float(v5.mean() * 100) else
            "NO_GO_or_inconclusive_for_GPU"
        ),
        "interpretation_limit": (
            "Readout training sees 9 EEG trials/exemplars per known concept; "
            "held-out concepts are excluded only from the linear readout, "
            "NOT from the 06b EEG backbone. No claim of novel concepts or "
            "joint EEG-Image-Text learning is supported by this diagnostic."
        ),
    }
    return result


def run(root, cache):
    from phase1.historical_06b import (
        _load_training, POST17, ROUTE_INIT, ROUTE_LAYERS, FEATURE_WIDTH)
    from phase1.models import Aligner
    from phase1.trimodal_06b import _load_text, preflight_text_inputs
    from phase1.joint_data import derangement

    t0 = time.monotonic()
    manifest = root/"train_only_manifest.json"
    checkpoint = root/"baseline/within_route_c_sub-01_s0_best.pt"
    record = json.loads((root/"baseline/run.json").read_text())
    m = json.loads(manifest.read_text())
    if (m.get("partition") != "official_training" or
        m.get("official_test_opened") is not False or
        record.get("status") != "complete_training_only" or
        record.get("checkpoint") != checkpoint.name or
        record.get("checkpoint_sha256") != sha256(checkpoint)):
        raise ValueError("Historical 06b train-only checkpoint provenance failed")
    text_file = cache/"captions_train_only.npy"
    ids_file = cache/"caption_ids_train_only.json"
    prov_file = cache/"caption_provenance_train_only.json"
    if not preflight_text_inputs(text_file,ids_file,prov_file).get("train_only"):
        raise ValueError("Caption cache provenance failed")
    _, eeg, layers, train, val, _ = _load_training(manifest,torch.device("cpu"))
    if len(train) != 14886 or len(val) != 1654:
        raise ValueError("Wrong training-partition split")
    model = Aligner(len(POST17)*eeg.shape[2],
                    [FEATURE_WIDTH]*len(ROUTE_LAYERS), list(ROUTE_INIT))
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    model.eval()
    with torch.no_grad():
        images = torch.cat([
            model.img([x[rows] for x in layers]).float()
            for rows in torch.arange(len(eeg)).split(256)],dim=0)
        eeg_all = torch.cat([
            model.eeg(eeg[rows].float()).float()
            for rows in torch.arange(len(eeg)).split(256)],dim=0)
        eeg_dev = eeg_all[val]
    del model,eeg,layers
    captions, caption_ids = _load_text(text_file,ids_file,torch.device("cpu"))
    if len(captions) != len(images) or len(caption_ids) != len(images):
        raise ValueError("Image/EEG/caption row count mismatch")
    mean = captions[train].mean(0,keepdim=True)
    std = captions[train].std(0,keepdim=True)+1e-4
    text = (captions - mean)/std
    train, val = train.cpu(), val.cpu()
    real = fit_bridge(images,text,train)
    permutation = derangement((train//10).tolist(),CONTROL_SEED)
    shuffled = fit_bridge(images,text,train,permutation)
    rows = val.tolist()
    identities = [caption_ids[i] for i in rows]
    report = compare(eeg_dev,images[val],text[val],identities,rows,real,shuffled)
    report["EEG_fitted_readout_gate"] = direct_eeg_ridge_diagnostic(
        eeg_all, text, caption_ids, train, val, (eeg_dev, real))
    report.update({
        "status":"measured_cpu_only",
        "official_test_opened":False,
        "subject":"sub-01","seed":0,"gallery_size":1654,
        "baseline_checkpoint_sha256":sha256(checkpoint),
        "manifest_sha256":sha256(manifest),
        "caption_ids_sha256":sha256(ids_file),
        "cpu_wall_seconds":round(time.monotonic()-t0,2),
        "limitations":[
            "Development partition; NOT an official 200-way/10-subject score",
            "Historical visual normalization fitted including development rows",
            "EEG-image-caption stimulus row IDs not independently verified",
            "Do not treat B-live 1.27% as a paired baseline to frozen 06b"
        ]
    })
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(
        "/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec"))
    p.add_argument("--cache",type=Path,default=Path(
        "/content/drive/MyDrive/eeg_results/cache"))
    args=p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    print(json.dumps(run(args.root,args.cache),indent=2),flush=True)
