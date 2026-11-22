"""Read-only, single-configuration 256-way shared OpenCLIP image/text teacher preflight.

TRAIN-PARTITION-only: 256 unique concepts, one *optimizer-fit* image per
concept (row 10*k, not development row 10*k+9). No EEG checkpoint, test
partition, training, gradient, prompt search, or architecture modifications.
Passing this teacher-level diagnostic NEVER establishes EEG->Caption.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.image_caption_bridge_gate import bootstrap_ci, caption_hits

DEFAULT_ROOT = Path('/content/drive/MyDrive/THINGS-EEG2')
DEFAULT_CACHE = Path('/content/drive/MyDrive/eeg_results/cache')
SEED = 20261122
N_CONCEPTS = 1654
IMAGES_PER_CONCEPT = 10
SAMPLE_SIZE = 256
EXPECTED_OPENCLIP_WEIGHT_SHA = 'a3ce3c4a2245ed2a572d2eb864a62f0a4ca62b3123b83f8a52a523c3b1dd32a4'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_training_rows(meta, image_meta):
    """Fixed 256 training-only rows from distinct concept blocks."""
    source_files = list(image_meta['train_img_files'])
    official_concepts = list(image_meta['train_img_concepts'])
    paths = list(meta['train_rel'])
    if (len(source_files) != 16540 or len(official_concepts) != 16540
            or list(meta['train_files']) != source_files or len(paths) != 16540):
        raise ValueError('Official training image catalog / meta_v6 mismatch')
    for concept in range(N_CONCEPTS):
        start = 10 * concept
        names = official_concepts[start:start+10]
        if len(set(names)) != 1 or not str(names[0]).startswith(f'{concept+1:05d}_'):
            raise ValueError('Invalid official ten-image concept block')
    if any(Path(rel).name != str(name) or
           not Path(rel).as_posix().startswith('images_set/training_images/')
           or '..' in Path(rel).parts for rel, name in zip(paths, source_files)):
        raise ValueError('Refusing unexpected image path/order (test or traversal)')
    selected = sorted(np.random.default_rng(SEED).choice(N_CONCEPTS, SAMPLE_SIZE,
                                                           replace=False).tolist())
    rows = [10 * concept for concept in selected]
    names = [str(official_concepts[row]).split('_', 1)[1].replace('_', ' ')
             for row in rows]
    if len(set(names)) != SAMPLE_SIZE or any(row % 10 == 9 for row in rows):
        raise ValueError('Development/test or duplicate concept selected')
    return rows, names, [paths[row] for row in rows]


def normalized_scores(image_vectors, target_vectors):
    if (image_vectors.ndim != 2 or target_vectors.ndim != 2
            or image_vectors.shape != target_vectors.shape
            or image_vectors.shape[0] != SAMPLE_SIZE
            or not torch.isfinite(image_vectors).all()
            or not torch.isfinite(target_vectors).all()):
        raise ValueError('Expected matching finite image/text matrices')
    return F.normalize(image_vectors.float(), dim=1) @ F.normalize(target_vectors.float(), dim=1).T


def score_conditions(image_vectors, blip_vectors, concept_vectors, blip_strings, concept_names, rows):
    """One fixed teacher evaluation with fixed off-by-one, paired controls."""
    if not (len(blip_strings) == len(concept_names) == len(rows) == SAMPLE_SIZE):
        raise ValueError('Wrong score/identity lengths')
    if len(set(rows)) != SAMPLE_SIZE or len(set(concept_names)) != SAMPLE_SIZE:
        raise ValueError('Nonunique image rows or concept identities')
    # Permuting the *vectors*, not the labels, breaks authentic image-caption association.
    # This is a negative control, NOT a training condition.
    permutation = np.roll(np.arange(SAMPLE_SIZE, dtype=np.int64), 1)
    def hits(embeds, ids):
        real = normalized_scores(image_vectors, embeds)
        wrong = normalized_scores(image_vectors, embeds[permutation])
        r1, r5 = caption_hits(real, ids, ids, rows)
        n1, n5 = caption_hits(wrong, ids, ids, rows)
        ci = bootstrap_ci(r1, n1, SEED)
        def pack(a, b):
            return {'top1_pct': round(float(100*a.mean()), 5),
                    'top5_pct': round(float(100*b.mean()), 5),
                    'top1_hits': int(a.sum()), 'top5_hits': int(b.sum())}
        return {'real': pack(r1, r5), 'shuffled_vectors': pack(n1, n5),
                'real_minus_shuffle_top1_ci95_pp': ci}
    caption = hits(blip_vectors, blip_strings)
    concept = hits(concept_vectors, concept_names)
    # Pragmatic fixed thresholds, not posterior tuning or established publication metrics.
    passed = (caption['real']['top1_pct'] >= 25.0
              and concept['real']['top1_pct'] >= 35.0
              and caption['real_minus_shuffle_top1_ci95_pp'][0] > 0
              and concept['real_minus_shuffle_top1_ci95_pp'][0] > 0)
    return {'image_to_existing_blip_caption': caption,
            'image_to_canonical_concept_text': concept,
            'teacher_preflight': ('PASS_SHARED_TEACHER_ONLY_NOT_GPU_APPROVAL'
                                  if passed else 'NO_GO_SHARED_TEACHER'),
            'gpu_training_authorized': False,
            'interpretation': '256-way one-image-per-concept train-fit diagnostic: shared image/text model only; not EEG->Caption, no new EEG model, not the official 200-way test'}


def load_train_catalog(root, cache):
    """Independent original filenames and pinned source JSON. Never touch test keys."""
    root, cache = Path(root).resolve(), Path(cache).resolve()
    metadata = root/'images_set/image_metadata.npy'
    meta_cache = cache/'meta_v6.pt'
    caption_path = cache/'captions_blip_large.json'
    ids_path = cache/'caption_ids_train_only.json'
    provenance_path = cache/'caption_provenance_train_only.json'
    required = [metadata, meta_cache, caption_path, ids_path, provenance_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required training-only source missing: ' + repr(missing))
    prov = json.loads(provenance_path.read_text(encoding='utf-8'))
    if (prov.get('partition') != 'official_training' or prov.get('used_test') is not False
            or prov.get('source_image_partition') != 'official_training'):
        raise ValueError('Caption source partition disallowed')
    for source in (meta_cache, caption_path):
        if prov.get('source_sha256', {}).get(source.name) != sha256(source):
            raise ValueError('Pinned caption source hash mismatch: '+source.name)
    official = np.load(metadata, allow_pickle=True).item()  # trusted official metadata
    meta = torch.load(meta_cache, map_location='cpu', weights_only=True)
    originals = json.loads(caption_path.read_text(encoding='utf-8'))['train']
    caption_ids = json.loads(ids_path.read_text(encoding='utf-8'))
    if originals != caption_ids or len(originals) != 16540:
        raise ValueError('Caption identities disagree with pinned BLIP train source')
    rows, names, selected_paths = select_training_rows(meta, official)
    resolved = [(root/rel).resolve() for rel in selected_paths]
    if any(not p.is_relative_to(root) or not p.is_file() for p in resolved):
        raise FileNotFoundError('Some selected original TRAIN images are absent or outside root')
    return {'image_paths': resolved,
            'captions': [originals[i] for i in rows],
            'names': names, 'rows': rows,
            'source_sha256': {p.name: sha256(p) for p in required},
            'selected_image_sha256': {str(row): sha256(path)
                                      for row, path in zip(rows, resolved)}}


def clip_encode_triplets(image_paths, captions, names):
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device='cpu')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    weight_hash = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        weight_hash.update(key.encode('utf-8'))
        weight_hash.update(value.detach().contiguous().cpu().numpy().tobytes())
    current_sha = weight_hash.hexdigest()
    if current_sha != EXPECTED_OPENCLIP_WEIGHT_SHA:
        raise ValueError('OpenCLIP pretrained weight SHA differs from prior pinned CPU report')
    if not (len(image_paths) == len(captions) == len(names) == SAMPLE_SIZE):
        raise ValueError('Unexpected number of image/text pairs')
    image_vectors = []
    with torch.no_grad():
        for start in range(0, SAMPLE_SIZE, 8):
            batch = []
            for path in image_paths[start:start+8]:
                with Image.open(path) as img:
                    batch.append(preprocess(img.convert('RGB')))
            image_vectors.append(model.encode_image(torch.stack(batch)).float().cpu())
        images = torch.cat(image_vectors)
        def text_embed(strings):
            pieces = []
            for start in range(0, len(strings), 64):
                pieces.append(model.encode_text(tokenizer(strings[start:start+64])).float().cpu())
            return torch.cat(pieces)
        blip = text_embed(captions)
        concept = text_embed(['a photo of a ' + name for name in names])
    return images, blip, concept, {
        'encoder': 'open_clip ViT-B-32 pretrained=openai',
        'open_clip_version': getattr(open_clip, '__version__', 'NOT_REPORTED'),
        'weights_sha256': current_sha,
        'concept_template': 'a photo of a {concept}',
        'image_and_text_encoded_using_same_loaded_checkpoint': True}


def run(root=DEFAULT_ROOT, cache=DEFAULT_CACHE):
    start = time.monotonic()
    source = load_train_catalog(root, cache)
    image, caption, concept, model = clip_encode_triplets(
        source['image_paths'], source['captions'], source['names'])
    scores = score_conditions(image, caption, concept, source['captions'],
                              source['names'], source['rows'])
    # Output does not include the full caption text, which is supplied by original cache.
    return {'status': 'MEASURED_CPU_ONLY', 'official_test_opened': False,
            'gpu_training_executed': False, 'subject': 'NOT_USED',
            'partition': 'optimizer_fit_only (one image per selected known concept)',
            'gallery': SAMPLE_SIZE, 'unique_concepts': SAMPLE_SIZE,
            'selected_training_rows': source['rows'],
            'source_sha256': source['source_sha256'],
            'selected_image_sha256': source['selected_image_sha256'],
            'clip_model': model, 'results': scores,
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'cpu_wall_seconds': round(time.monotonic()-start, 2)},
            'limitations': ['This is image-to-text teacher feasibility, EEG not evaluated',
                            'BLIP captions can be noisy; concept-name retrieval is a separate, easier task',
                            'No official test, no real EEG training, no 200-way SOTA comparison']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, required=True,
                        help='Exclusive new JSON report outside the source directories')
    args = parser.parse_args()
    path = args.output.resolve()
    if (path.exists() or path.is_relative_to(args.root.resolve())
            or path.is_relative_to(args.cache.resolve())):
        raise ValueError('Report must be exclusive-new and outside original input folders')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(args.root, args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'report': str(path), 'status': result['status'],
                      'results': result['results'], 'runtime': result['runtime']},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
