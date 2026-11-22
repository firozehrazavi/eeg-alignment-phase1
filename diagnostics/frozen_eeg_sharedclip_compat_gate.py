"""Single frozen-06b EEG -> shared-CLIP visual-teacher compatibility gate (CPU).

Comparison on the SAME 256 held-out within-training-partition development EEG
queries and the SAME caption strings/gallery:
 A EEG -> train-image OpenCLIP visual features -> dev OpenCLIP caption vectors
 B EEG -> train-caption OpenCLIP text vectors -> dev OpenCLIP captions
 C EEG -> legacy stored train-caption vectors -> dev legacy captions
 D cross-concept shuffled training image-target control for A.

All fits are on first image/trial per 256 known concepts. Development uses
the LAST image/trial (row 10*c+9), never a fit target. No EEG/CLIP training,
no test partition, and no dev image feature encoding. No inference image gallery.

This diagnostic is underpowered (256 first-exemplar fits) and is NOT a
scientific demonstration of joint EEG–Image–Text or official SOTA.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.shared_clip_teacher_gate import (
    DEFAULT_CACHE, DEFAULT_ROOT, EXPECTED_OPENCLIP_WEIGHT_SHA,
    load_train_catalog, SAMPLE_SIZE)
from diagnostics.image_caption_bridge_gate import (
    bootstrap_ci, caption_hits, fit_bridge, RIDGE_RATIO)
from diagnostics.eeg_bridge_transfer_gate import bridge_scores_from_embeddings
from phase1.historical_06b import (
    FEATURE_WIDTH, POST17, POST17_INDICES, ROUTE_INIT, ROUTE_LAYERS,
    _resolve)
from phase1.models import Aligner

DEFAULT_RUN = Path(
    '/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec')
SEED = 20261122


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_first_last(rows):
    """One fit/training and one development exemplar for each selected concept."""
    if (len(rows) != SAMPLE_SIZE or len(set(rows)) != SAMPLE_SIZE or
            any(row < 0 or row >= 16540 or row % 10 != 0 for row in rows)):
        raise ValueError('Training rows must be 256 unique first-exemplar train rows')
    dev = [r + 9 for r in rows]
    if any(q % 10 != 9 or q >= 16540 for q in dev):
        raise ValueError('Development rows must be last exemplar per concept')
    if set(dev) & set(rows):
        raise ValueError('Fit and development rows overlap')
    return dev


def read_frozen_eeg(run, cache, fit_rows, dev_rows):
    """Authenticate historical checkpoint and read ONLY selected training EEG."""
    from diagnostics.lima_feasibility_gate import validate_baseline_record
    run, cache = Path(run).resolve(), Path(cache).resolve()
    manifest_path = run/'train_only_manifest.json'
    record_path = run/'baseline/run.json'
    ckpt = run/'baseline/within_route_c_sub-01_s0_best.pt'
    for p in (manifest_path, record_path, ckpt):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if (manifest.get('partition') != 'official_training'
            or manifest.get('official_test_opened') is not False
            or manifest.get('subject') != 'sub-01'
            or manifest.get('seed') != 0
            or manifest.get('eeg_file') != 'sub-01_train_z16.npy'
            or record.get('fit_rows') != [i for i in range(16540) if i % 10 != 9]
            or record.get('validation_rows') != [i for i in range(9,16540,10)]):
        raise ValueError('Not the verified historical subject-01 seed-0 train/dev split')
    validate_baseline_record(record, manifest_path, ckpt)
    eeg_file = _resolve(Path(manifest['cache_root']), manifest['eeg_file'])
    if not eeg_file.is_file():
        raise FileNotFoundError(str(eeg_file))
    pinned = manifest.get('source_hashes', {}).get(manifest['eeg_file'])
    if not pinned or sha256(eeg_file) != pinned:
        raise ValueError('Historical train-only EEG input hash mismatch')
    eeg = np.load(eeg_file, mmap_mode='r', allow_pickle=False)
    if eeg.ndim != 3 or eeg.shape[:2] != (16540, 63):
        raise ValueError('Wrong historical EEG cache dimensions')
    rows = list(fit_rows) + list(dev_rows)
    extracted = np.ascontiguousarray(eeg[rows][:, POST17_INDICES, :])
    if extracted.dtype.kind != 'f' or not np.isfinite(extracted).all():
        raise ValueError('Nonfinite or nonfloat selected EEG data')
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        model = Aligner(len(POST17)*eeg.shape[2],
                        [FEATURE_WIDTH]*len(ROUTE_LAYERS),
                        list(ROUTE_INIT))
        model.load_state_dict(torch.load(ckpt, map_location='cpu',
                                         weights_only=True), strict=True)
        model.eval()
        x = torch.from_numpy(extracted).float()
        projected = torch.cat(
            [model.eeg(batch).float().cpu() for batch in x.split(128)], 0)
    if projected.shape != (2*SAMPLE_SIZE, 1024) or not torch.isfinite(projected).all():
        raise ValueError('Unexpected frozen EEG shape or invalid output')
    return projected[:SAMPLE_SIZE], projected[SAMPLE_SIZE:], {
        'checkpoint_sha256': sha256(ckpt),
        'manifest_sha256': sha256(manifest_path),
        'training_eeg_sha256': pinned,
        'fit_rows': fit_rows,
        'dev_rows': dev_rows,
        'eeg_stimulus_identity_independent': 'NOT_VERIFIED',
        'normalization_note': 'Historical 06b image standardizer included development rows'}


def encode_fit_images_and_all_captions(image_paths, fit_captions, dev_captions):
    """One shared OpenCLIP instance, pinned state dict; no dev image encodings."""
    import open_clip
    from PIL import Image

    if not (len(image_paths) == len(fit_captions) == len(dev_captions) == SAMPLE_SIZE):
        raise ValueError('Incorrect fixed 256-way fit/dev triple lengths')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device='cpu')
    tok = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    weights_sha = digest.hexdigest()
    if weights_sha != EXPECTED_OPENCLIP_WEIGHT_SHA:
        raise ValueError('Frozen OpenCLIP weights differ from verified teacher gate')

    with torch.no_grad():
        images = []
        for j in range(0, SAMPLE_SIZE, 8):
            batch = []
            for path in image_paths[j:j+8]:
                with Image.open(path) as im:
                    batch.append(preprocess(im.convert('RGB')))
            images.append(model.encode_image(torch.stack(batch)).float().cpu())
        def text_encode(strings):
            pieces = []
            for j in range(0, len(strings), 64):
                pieces.append(model.encode_text(tok(strings[j:j+64])).float().cpu())
            return torch.cat(pieces)
        img = torch.cat(images)
        fit_text = text_encode(fit_captions)
        dev_text = text_encode(dev_captions)
    if (img.shape != (SAMPLE_SIZE,512)
            or fit_text.shape != (SAMPLE_SIZE,512)
            or dev_text.shape != (SAMPLE_SIZE,512)
            or not all(torch.isfinite(a).all() for a in (img,fit_text,dev_text))):
        raise ValueError('Bad frozen shared-CLIP embeddings')
    return img, fit_text, dev_text, {
        'text_and_image_same_checkpoint': True,
        'openclip_weights_sha256': weights_sha,
        'openclip_version': getattr(open_clip, '__version__', 'unknown')}


def compare_readouts(eeg_fit, eeg_dev, image_fit, shared_caption_fit,
                     shared_caption_dev, legacy_caption_fit,
                     legacy_caption_dev, dev_ids, dev_rows):
    """Identical 256 train pairs, 256 queries, target strings and fixed Ridge."""
    expected = [
        (eeg_fit, (256, 1024)), (eeg_dev, (256, 1024)),
        (image_fit, (256, 512)), (shared_caption_fit, (256, 512)),
        (shared_caption_dev, (256, 512)),
        (legacy_caption_fit, (256, 512)),
        (legacy_caption_dev, (256, 512))]
    if any(x.shape != shape or not torch.isfinite(x).all()
           for x, shape in expected):
        raise ValueError('Bad frozen readout shape or nonfinite features')
    if len(dev_rows) != SAMPLE_SIZE or len(dev_ids) != SAMPLE_SIZE:
        raise ValueError('Missing 256 dev image/caption identities')

    # One predeclared fixed Ridge per target, no hyperparameter search.
    shared_visual = fit_bridge(eeg_fit, image_fit, torch.arange(256))
    direct_text = fit_bridge(eeg_fit, shared_caption_fit, torch.arange(256))
    legacy_text = fit_bridge(eeg_fit, legacy_caption_fit, torch.arange(256))
    cyclic = torch.roll(image_fit, shifts=1, dims=0)
    shuffled_visual = fit_bridge(eeg_fit, cyclic, torch.arange(256))

    def evaluate(bridge, gallery):
        scores = bridge_scores_from_embeddings(eeg_dev, gallery, bridge)
        top1, top5 = caption_hits(scores, dev_ids, dev_ids, dev_rows)
        return top1, top5

    visual_1, visual_5 = evaluate(shared_visual, shared_caption_dev)
    direct_1, direct_5 = evaluate(direct_text, shared_caption_dev)
    legacy_1, legacy_5 = evaluate(legacy_text, legacy_caption_dev)
    null_1, null_5 = evaluate(shuffled_visual, shared_caption_dev)
    def summary(a, b):
        return {'top1_pct': round(float(a.mean()*100), 5),
                'top5_pct': round(float(b.mean()*100), 5),
                'hits_top1': int(a.sum()), 'hits_top5': int(b.sum())}
    ci_shuffle = bootstrap_ci(visual_1, null_1, SEED)
    ci_direct = bootstrap_ci(visual_1, direct_1, SEED+1)
    ci_legacy = bootstrap_ci(visual_1, legacy_1, SEED+2)
    robust = (summary(visual_1,visual_5)['top1_pct'] >= 5
              and ci_shuffle[0] > 0 and ci_direct[0] > 0 and ci_legacy[0] > 0)
    return {
        'shared_visual_teacher_frozen_eeg_to_dev_caption': summary(visual_1, visual_5),
        'direct_frozen_eeg_to_shared_caption': summary(direct_1,direct_5),
        'legacy_frozen_eeg_to_original_caption': summary(legacy_1,legacy_5),
        'cross_concept_shuffled_visual_teacher': summary(null_1,null_5),
        'shared_visual_minus_shuffle_top1_ci95_pp': ci_shuffle,
        'shared_visual_minus_direct_top1_ci95_pp': ci_direct,
        'shared_visual_minus_legacy_top1_ci95_pp': ci_legacy,
        'preregistered_cpu_compatibility_gate':
            'COMPATIBLE_CANDIDATE_ONLY_NOT_GPU_APPROVAL' if robust else
            'NO_GO_OR_INCONCLUSIVE_SHARED_VISUAL_TRANSFER',
        'gpu_training_authorized': False,
        'limit': '256-way known-concept development, all EEG frozen; this is not joint learning or official 200-way retrieval'
    }


def run(run_root=DEFAULT_RUN, root=DEFAULT_ROOT, cache=DEFAULT_CACHE):
    start = time.monotonic()
    run_root, root, cache = map(Path, (run_root, root, cache))
    catalog = load_train_catalog(root, cache)
    fit_rows = catalog['rows']
    dev_rows = split_first_last(fit_rows)
    all_texts = json.loads((cache/'caption_ids_train_only.json').read_text(encoding='utf-8'))
    # Caption string identities: first images for fit, last images for dev,
    # but NOT the dev image tensors. We only encode dev caption text.
    fit_captions = catalog['captions']
    dev_captions = [all_texts[i] for i in dev_rows]
    eeg_fit, eeg_dev, eeg_meta = read_frozen_eeg(run_root, cache,fit_rows,dev_rows)
    shared_fit_images, shared_fit_captions, shared_dev_captions, clip_meta = (
        encode_fit_images_and_all_captions(
            catalog['image_paths'], fit_captions, dev_captions))
    from phase1.trimodal_06b import preflight_text_inputs
    features_file = cache/'captions_train_only.npy'
    id_file = cache/'caption_ids_train_only.json'
    provenance_file = cache/'caption_provenance_train_only.json'
    if not preflight_text_inputs(features_file,id_file,provenance_file).get('train_only'):
        raise ValueError('Legacy caption-feature preflight failed')
    original_text = np.load(features_file, mmap_mode='r', allow_pickle=False)
    if original_text.shape != (16540,512):
        raise ValueError('Legacy text dimension mismatch')
    # Legacy readout uses same 256 training and 256 development image identities.
    # The source legacy text encoder revision was not pinned; report caveat.
    legacy_fit = torch.tensor(np.asarray(original_text[fit_rows], dtype=np.float32).copy())
    legacy_dev = torch.tensor(np.asarray(original_text[dev_rows], dtype=np.float32).copy())
    metrics = compare_readouts(
        eeg_fit,eeg_dev,shared_fit_images,shared_fit_captions,
        shared_dev_captions,legacy_fit,legacy_dev,dev_captions,dev_rows)
    return {
        'status': 'MEASURED_CPU_ONLY', 'gpu_training_executed': False,
        'official_test_opened': False,
        'subject': 'sub-01', 'seed': 0,
        'scope': '256 distinct known concepts, 1 fit exemplar and 1 development exemplar each; 256-way',
        'image_gallery_at_eeg_inference': False,
        'training_pairs': 256, 'development_queries': 256, 'gallery_size': 256,
        'ridge_ratio': RIDGE_RATIO, 'selection_seed': SEED,
        'pinned_eeg': eeg_meta, 'source_sha256': catalog['source_sha256'],
        'shared_clip_encoder': clip_meta, 'results': metrics,
        'runtime': {'python': platform.python_version(),'numpy': np.__version__,
                    'torch': torch.__version__,
                    'cpu_wall_seconds': round(time.monotonic()-start, 2)},
        'limitations': [
            '256 fit rows only: underpowered compatibility gate, not a full-fit baseline',
            'Known concepts and historic frozen EEG pretrained on these concept categories',
            'Historical EEG model image normalizer included development images',
            'Dev target is image-specific BLIP text; duplicate caption identities are valid positives',
            'Independent EEG/InternViT stimulus row identity not yet verified',
            'Legacy CLIP text exact source revision not pinned',
            'Not an official test, not a 200-way comparison, not evidence of three-way joint learning']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out.exists() or any(out.is_relative_to(p.resolve())
                            for p in (args.run_root,args.root,args.cache))):
        raise ValueError('Report must be exclusive-new outside source/cache/checkpoint directories')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    report = run(args.run_root,args.root,args.cache)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'report':str(out),'status':report['status'],
                      'results':report['results'],'runtime':report['runtime']},
                     ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
