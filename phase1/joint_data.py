"""Train-partition-only data contract. No pickle or official evaluation loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .core import validate_catalog, concept_split


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_training(path, subjects):
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding='utf8'))
    if m.get('partition') != 'official_training':
        raise ValueError('Only an explicitly declared official_training manifest is accepted')
    if m.get('eeg_normalization') != 'not_split_zscored':
        raise ValueError('Need EEG before historical split-specific z-scoring')
    if not m.get('preprocessing_provenance') or not m.get('text_provenance'):
        raise ValueError('Document upstream EEG preprocessing and frozen text provenance')
    if m.get('text_unit') != 'image_caption':
        raise ValueError('This experiment requires image-specific captions, not repeated concept labels')
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Need unique source subjects')
    c = m['catalog']
    validate_catalog(c['image_ids'], c['concept_ids'], c['content_hashes'])
    n = len(c['image_ids'])
    if len(c['caption_ids']) != n or len(set(c['caption_ids'])) != n:
        raise ValueError('Need distinct caption identities; repeated captions require another protocol')
    hashes = {}
    def array(entry):
        if entry.get('partition') != 'official_training':
            raise ValueError('Every loaded artifact must declare official_training')
        p = (path.parent / entry['path']).resolve()
        # A filename cannot establish provenance, but catches common accidental paths.
        if any(word in p.name.lower() for word in ('test', 'eval')):
            raise ValueError('Protected-looking array path refused')
        actual = digest(p)
        if actual != entry['sha256']:
            raise ValueError('Artifact SHA256 mismatch: ' + str(p))
        a = np.load(p, allow_pickle=False, mmap_mode='r')
        if len(a) != n or not np.issubdtype(a.dtype, np.floating):
            raise ValueError('Wrong row count or dtype: ' + str(p))
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite array: ' + str(p))
        hashes[str(p)] = actual
        return torch.from_numpy(np.array(a, dtype=np.float32))
    if m.get('visual_layers') != [20, 24, 28, 32, 36] or len(m['visuals']) != 5:
        raise ValueError('Expected five route_c layers in documented order')
    visual = [array(e) for e in m['visuals']]
    text = array(m['text'])
    eeg = {s: array(m['subjects'][s]) for s in subjects}
    if any(v.ndim != 2 for v in visual) or text.ndim != 2:
        raise ValueError('Visual/text arrays must be rows x features')
    if any(x.ndim != 3 or x.shape != next(iter(eeg.values())).shape for x in eeg.values()):
        raise ValueError('EEG must be aligned rows x channels x time for every source')
    if len(torch.unique(text, dim=0)) != n:
        raise ValueError('Identical caption embeddings: image-specific retrieval would be ambiguous')
    return m, visual, text, eeg, hashes


def split_three(concepts, seed=20261117):
    """Fit/tune/locked development holdout, approximately 80/10/10 by concept."""
    rest, hold = concept_split(concepts, seed, .1)
    f, t = concept_split([concepts[i] for i in rest], seed + 1, 1 / 9)
    fit, tune = [rest[i] for i in f], [rest[i] for i in t]
    if min(map(len, (fit, tune, hold))) < 2:
        raise ValueError('Need at least two images per partition')
    return fit, tune, hold


def derangement(concepts, seed):
    """Fixed cross-concept permutation; never move text across split boundaries."""
    rng = np.random.default_rng(seed)
    groups = {c: np.flatnonzero(np.asarray(concepts) == c) for c in set(concepts)}
    largest = max(map(len, groups.values()))
    if largest * 2 > len(concepts):
        raise ValueError('No cross-concept permutation exists: one concept is a majority')
    keys = sorted(groups); rng.shuffle(keys)
    order = np.concatenate([rng.permutation(groups[c]) for c in keys])
    shifted = np.roll(order, largest)
    p = np.empty(len(order), dtype=np.int64); p[order] = shifted
    if any(concepts[i] == concepts[j] for i, j in enumerate(p)):
        raise RuntimeError('Internal cross-concept permutation error')
    return torch.as_tensor(p)
