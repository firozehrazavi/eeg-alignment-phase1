"""Frozen-06b EEG -> canonical concept text vs EEG -> BLIP caption.

Read-only CPU development diagnostic. It does NOT train EEG, use official test,
or establish image-specific caption recovery, novelty, or GPU readiness.
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path('/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec')
CACHE = Path('/content/drive/MyDrive/eeg_results/cache')
METADATA = Path('/content/drive/MyDrive/THINGS-EEG2/images_set/image_metadata.npy')
SEED = 20261122


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_concepts(concepts, filenames):
    """Return exactly 1,654 unique labels in the documented ten-row order."""
    if len(concepts) != 16540 or len(filenames) != 16540:
        raise ValueError('Expected exactly 16,540 training image rows')
    labels = []
    for k in range(1654):
        block = concepts[k * 10:(k + 1) * 10]
        if len(set(block)) != 1 or not re.fullmatch(r'\d{5}_[a-zA-Z0-9_\-]+', str(block[0])):
            raise ValueError('Missing or noncontiguous sorted concept block')
        prefix, label = str(block[0]).split('_', 1)
        if int(prefix) != k + 1:
            raise ValueError('Official concept order mismatch')
        if len(set(filenames[k * 10:(k + 1) * 10])) != 10:
            raise ValueError('Duplicate filenames inside concept block')
        labels.append(label.replace('_', ' '))
    if len(set(labels)) != len(labels):
        raise ValueError('Ambiguous duplicate canonical concept names')
    return labels


def source_audit(metadata_path, meta_v6_path, captions_source, caption_ids_path,
                 provenance_path):
    """Verify exact text/cache and feature-metadata catalog order, not EEG IDs."""
    paths = [Path(x) for x in (metadata_path, meta_v6_path, captions_source,
                                 caption_ids_path, provenance_path)]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError('Missing required user-owned train-only source: ' +
                                ', '.join(str(p) for p in paths if not p.is_file()))
    provenance = json.loads(paths[4].read_text(encoding='utf-8'))
    if provenance.get('partition') != 'official_training' or provenance.get('used_test') is not False:
        raise ValueError('Invalid training-only caption provenance')
    expected = provenance.get('source_sha256', {})
    for p in (paths[1], paths[2]):
        if expected.get(p.name) != digest(p):
            raise ValueError('Declared original source hash mismatch: ' + p.name)
    # Official THINGS-EEG2 image metadata is a trusted user-owned numpy object dictionary.
    meta = np.load(paths[0], allow_pickle=True).item()
    files = list(meta['train_img_files'])
    concepts = list(meta['train_img_concepts'])
    labels = canonical_concepts(concepts, files)
    cached_meta = torch.load(paths[1], map_location='cpu', weights_only=True)
    if list(cached_meta['train_files']) != files:
        raise ValueError('meta_v6 train_files differ from official image_metadata')
    if len(cached_meta['train_rel']) != 16540:
        raise ValueError('meta_v6 image relative paths truncated')
    if any(Path(rel).name != name for rel, name in zip(cached_meta['train_rel'], files)):
        raise ValueError('meta_v6 relative paths disagree with filenames')
    original = json.loads(paths[2].read_text(encoding='utf-8'))['train']
    copied = json.loads(paths[3].read_text(encoding='utf-8'))
    if len(original) != 16540 or copied != original:
        raise ValueError('BLIP caption IDs do not exactly match original train captions')
    return {'status': 'SOURCE_ORDER_AND_TEXT_VERIFIED',
            'train_rows': 16540, 'concepts': 1654,
            'blip_text_equals_caption_ids_all_rows': True,
            'meta_v6_train_files_equals_original_metadata': True,
            'eeg_stimulus_identity': 'NOT_INDEPENDENTLY_VERIFIED',
            'visual_extraction_row_identity': 'NOT_INDEPENDENTLY_VERIFIED',
            'legacy_clip_revision': provenance.get('encoder_revision', 'UNKNOWN'),
            'source_sha256': {p.name: digest(p) for p in paths}}, labels


def literal_name_audit(labels, captions):
    """Literal lexical evidence only: not an estimate of semantic correctness."""
    mention = []
    for i, cap in enumerate(captions):
        normalized = str(cap).lower().replace('-', ' ')
        name = labels[i // 10].lower()
        mention.append(bool(re.search(r'(?<!\w)' + re.escape(name) + r's?(?!\w)', normalized)))
    return {'literal_name_mention_rows': sum(mention),
            'total_rows': len(mention),
            'concepts_without_literal_mention': sum(not any(mention[k*10:(k+1)*10])
                                                    for k in range(1654)),
            'note': 'Synonyms, inflections and valid descriptions can omit a literal name; this is NOT a ground-truth caption-error rate'}


def encode_concept_prompts(labels):
    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError('Install the CPU text encoder: pip install open_clip_torch') from e
    # Explicitly matched to the DECLARED legacy architecture/weights; historical revision unpinned.
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai',
                                                         device='cpu')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    prompts = ['a photo of a ' + label for label in labels]
    with torch.no_grad():
        output = torch.cat([model.encode_text(tokenizer(prompts[i:i+128])).float()
                            for i in range(0, len(prompts), 128)])
    if output.shape != (1654, 512) or not torch.isfinite(output).all():
        raise ValueError('Wrong OpenCLIP text embedding shape or nonfinite features')
    weights_hash = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        weights_hash.update(key.encode('utf-8'))
        weights_hash.update(value.detach().contiguous().cpu().numpy().tobytes())
    return F.normalize(output, dim=-1), {
        'model': 'open_clip ViT-B-32 pretrained=openai',
        'open_clip_version': getattr(open_clip, '__version__', 'UNKNOWN'),
        'model_weights_sha256': weights_hash.hexdigest(),
        'prompt_template': 'a photo of a {concept}',
        'legacy_encoder_exact_weights_match': 'NOT_VERIFIABLE_ORIGINAL_REVISION_UNPINNED'}


def compare_embeddings(eeg_all, caption_all, caption_ids, labels, fit, val, concept_text):
    from diagnostics.eeg_bridge_transfer_gate import bridge_scores_from_embeddings
    from diagnostics.image_caption_bridge_gate import (fit_bridge, caption_hits, bootstrap_ci,
                                                        CONTROL_SEED)
    from phase1.joint_data import derangement
    if (eeg_all.shape != (16540, 1024) or caption_all.shape != (16540, 512)
            or concept_text.shape != (1654, 512) or len(caption_ids) != 16540):
        raise ValueError('Unexpected EEG/caption/concept embedding shapes')
    if not torch.isfinite(eeg_all).all() or not torch.isfinite(concept_text).all():
        raise ValueError('Nonfinite embeddings')
    if not torch.equal(fit, torch.arange(16540)[torch.arange(16540) % 10 != 9]):
        raise ValueError('Unexpected fit rows')
    if not torch.equal(val, torch.arange(9, 16540, 10)):
        raise ValueError('Unexpected development rows')
    repeated = concept_text.repeat_interleave(10, dim=0)
    mean = caption_all[fit].mean(0, keepdim=True)
    std = caption_all[fit].std(0, keepdim=True) + 1e-4
    caption_standardized = (caption_all - mean) / std
    # Fixed fit-only Ridge, identical ridge strength/score/tie semantics.
    concept_map = fit_bridge(eeg_all, repeated, fit)
    caption_map = fit_bridge(eeg_all, caption_standardized, fit)
    permutation = derangement((fit // 10).tolist(), CONTROL_SEED)
    concept_null_map = fit_bridge(eeg_all, repeated, fit, permutation)
    concept_scores = bridge_scores_from_embeddings(eeg_all[val], concept_text, concept_map)
    null_scores = bridge_scores_from_embeddings(eeg_all[val], concept_text, concept_null_map)
    blip_scores = bridge_scores_from_embeddings(eeg_all[val], caption_standardized[val], caption_map)
    # Distinct labels for all 1,654 concept queries, versus duplicate-aware exact-caption positives.
    ids = [str(label) for label in labels]
    concept1, concept5 = caption_hits(concept_scores, ids, ids, val.tolist())
    null1, null5 = caption_hits(null_scores, ids, ids, val.tolist())
    blip_ids = [caption_ids[int(i)] for i in val]
    blip1, blip5 = caption_hits(blip_scores, blip_ids, blip_ids, val.tolist())
    def pack(a, b):
        return {'top1_pct': round(100 * a.mean().item(), 5),
                'top5_pct': round(100 * b.mean().item(), 5),
                'hits_top1': int(a.sum().item()), 'hits_top5': int(b.sum().item())}
    ci_null = bootstrap_ci(concept1, null1, SEED)
    ci_blip = bootstrap_ci(concept1, blip1, SEED + 1)
    return {'concept_text': pack(concept1, concept5),
            'shuffled_concept_training_pairs': pack(null1, null5),
            'blip_caption_reference': pack(blip1, blip5),
            'concept_minus_shuffle_top1_ci95_pp': ci_null,
            'concept_minus_blip_top1_ci95_pp': ci_blip,
            'source_hypothesis_supported_in_known_concept_readout': bool(ci_null[0] > 0 and ci_blip[0] > 0),
            'gpu_decision': 'NO_GO_PENDING_INDEPENDENT_STIMULUS_ID_PROVENANCE_AND_CONTROLLED_TRAINING',
            'interpretation': 'Concept-label retrieval on known backbone concepts is easier than image-specific caption retrieval; task accuracies are NOT interchangeable; no tri-modal interaction trained'}


def run(root=ROOT, cache=CACHE, metadata_path=METADATA):
    from diagnostics.lima_feasibility_gate import (inspect_inputs, validate_baseline_record,
                                                   validate_split)
    from phase1.historical_06b import (FEATURE_WIDTH, POST17, ROUTE_INIT, ROUTE_LAYERS,
                                      _load_training, preflight)
    from phase1.models import Aligner
    from phase1.trimodal_06b import _load_text, preflight_text_inputs
    started = time.monotonic()
    root, cache, metadata_path = map(Path, (root, cache, metadata_path))
    paths, availability = inspect_inputs(root, cache)
    missing = [str(p) for p in (metadata_path, cache/'meta_v6.pt',
                               cache/'captions_blip_large.json') if not p.is_file()]
    if availability['missing_inputs'] or missing:
        return {'status': 'NOT_MEASURED', 'reason': 'Missing private input files',
                'missing_inputs': availability['missing_inputs'] + missing,
                'gpu_decision': 'NO_GO'}
    audit, labels = source_audit(metadata_path, cache/'meta_v6.pt',
                                 cache/'captions_blip_large.json', paths['caption_ids'],
                                 paths['caption_provenance'])
    if not preflight_text_inputs(paths['caption_features'], paths['caption_ids'],
                                 paths['caption_provenance']).get('train_only'):
        raise ValueError('Caption preflight failed')
    validate_baseline_record(json.loads(paths['baseline_record'].read_text()),
                             paths['manifest'], paths['checkpoint'])
    preflight(paths['manifest'])
    _, eeg, layers, fit, val, _ = _load_training(paths['manifest'], torch.device('cpu'))
    validate_split(fit, val)
    record = json.loads(paths['baseline_record'].read_text())
    if record.get('fit_rows') != fit.tolist() or record.get('validation_rows') != val.tolist():
        raise ValueError('Historical checkpoint split mismatch')
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        model = Aligner(len(POST17)*eeg.shape[2], [FEATURE_WIDTH]*len(ROUTE_LAYERS),
                        list(ROUTE_INIT))
        model.load_state_dict(torch.load(paths['checkpoint'], map_location='cpu',
                                         weights_only=True), strict=True)
        model.eval()
        eeg_all = torch.cat([model.eeg(eeg[rows].float()).float()
                             for rows in torch.arange(16540).split(256)])
    del model, eeg, layers
    captions, caption_ids = _load_text(paths['caption_features'], paths['caption_ids'],
                                       torch.device('cpu'))
    original = json.loads((cache/'captions_blip_large.json').read_text(encoding='utf-8'))['train']
    audit['literal_caption_name_check'] = literal_name_audit(labels, original)
    concept_text, encoder = encode_concept_prompts(labels)
    metrics = compare_embeddings(eeg_all, captions, caption_ids, labels, fit.cpu(), val.cpu(),
                                 concept_text)
    return {'status': 'MEASURED_CPU_ONLY', 'official_test_opened': False,
            'gpu_training_executed': False, 'scope': 'sub-01 seed-0, 1654-way TRAIN-PARTITION development',
            'baseline_checkpoint_sha256': digest(paths['checkpoint']),
            'baseline_manifest_sha256': digest(paths['manifest']),
            'audited_provenance': audit, 'concept_text_encoder': encoder,
            'measured_readouts': metrics,
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'cpu_wall_seconds': round(time.monotonic()-started, 2)},
            'limitations': ['Train concepts are seen by frozen EEG backbone',
                            'Concept retrieval is NOT image-specific EEG-to-caption retrieval',
                            'Historical image normalizer includes development examples',
                            'EEG stimulus-row IDs and visual extraction row IDs remain independently unverified',
                            'Original OpenCLIP revision is unpinned; exact parity cannot be guaranteed']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--metadata', type=Path, default=METADATA)
    p.add_argument('--output', type=Path, help='Write an exclusive-new JSON report outside research inputs')
    args = p.parse_args()
    if args.output:
        target = args.output.resolve()
        if target.exists() or any(target.is_relative_to(x.resolve()) for x in (args.root, args.cache)):
            raise ValueError('Report must be new and outside source/checkpoint directories')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    report = run(args.root, args.cache, args.metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
