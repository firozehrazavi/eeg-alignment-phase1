"""Identity-aware objectives and evaluation, with explicit data-access contracts."""
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F


def positive_mask(query_ids, candidate_ids, device=None):
    identities = {v: i for i, v in enumerate(dict.fromkeys([*query_ids, *candidate_ids]))}
    q = torch.tensor([identities[v] for v in query_ids], device=device)
    c = torch.tensor([identities[v] for v in candidate_ids], device=device)
    return q[:, None] == c[None, :]


def clip_loss(ze, zi, scale):
    """Unchanged 06b symmetric diagonal objective (historical comparator)."""
    ze, zi = F.normalize(ze, dim=-1), F.normalize(zi, dim=-1)
    logits = scale * ze @ zi.T
    y = torch.arange(len(ze), device=ze.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def multi_positive_loss(ze, zi, scale, eeg_ids, image_ids):
    """Symmetric negative log probability of the matching identity set.

    Positives are exact stimulus identities, never subject IDs or concept IDs.
    Supports rectangular EEG/image batches and reordered image galleries.
    """
    if len(ze) != len(eeg_ids) or len(zi) != len(image_ids):
        raise ValueError('ID/embedding length mismatch')
    logits = scale * F.normalize(ze.float(), dim=-1) @ F.normalize(zi.float(), dim=-1).T
    mask = positive_mask(eeg_ids, image_ids, logits.device)
    if not mask.any(1).all() or not mask.any(0).all():
        raise ValueError('Every anchor in both directions needs a positive')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite contrastive logits')
    pos = logits.masked_fill(~mask, -torch.inf)
    return ((logits.logsumexp(1) - pos.logsumexp(1)).mean()
            + (logits.logsumexp(0) - pos.logsumexp(0)).mean()) / 2


def retrieval_metrics(scores, query_ids, gallery_ids, expected_candidates=200):
    """Top-k percentages; ties break by lexicographic string gallery ID.

    Explicit identity lookup makes row/column permutations safe. Query repeats
    are allowed; candidate duplicates are not (200-way means 200 identities).
    """
    if scores.ndim != 2 or scores.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError('Score/identity shape mismatch')
    if not query_ids or len(gallery_ids) != expected_candidates:
        raise ValueError('Empty queries or incorrect gallery size')
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Duplicate gallery identities')
    if not set(query_ids) <= set(gallery_ids) or not torch.isfinite(scores).all():
        raise ValueError('Missing positive or nonfinite score')
    order = sorted(range(len(gallery_ids)), key=lambda j: str(gallery_ids[j]))
    ids = [gallery_ids[j] for j in order]
    ranked = scores[:, order].argsort(dim=1, descending=True, stable=True)
    truth = torch.tensor([ids.index(q) for q in query_ids], device=scores.device)
    return {f'top{k}': 100 * (ranked[:, :min(k, len(ids))] == truth[:, None]).any(1).double().mean().item()
            for k in (1, 5)}


def concept_split(concept_ids, seed=0, fraction=0.1):
    concepts = sorted(set(concept_ids))
    if len(concepts) < 2 or not 0 < fraction < 1:
        raise ValueError('Need at least two concepts and a fraction in (0, 1)')
    random.Random(seed).shuffle(concepts)
    n = min(len(concepts) - 1, max(1, round(len(concepts) * fraction)))
    held = set(concepts[:n])
    return ([i for i, c in enumerate(concept_ids) if c not in held],
            [i for i, c in enumerate(concept_ids) if c in held])


def validate_catalog(image_ids, concept_ids, content_hashes):
    if not (len(image_ids) == len(concept_ids) == len(content_hashes)):
        raise ValueError('Catalog length mismatch')
    if len(set(image_ids)) != len(image_ids) or len(set(content_hashes)) != len(content_hashes):
        raise ValueError('Duplicate image ID or content; canonicalize before splitting')
    if not all(image_ids) or not all(concept_ids) or not all(content_hashes):
        raise ValueError('Missing catalog metadata')


def validate_disjoint_catalogs(train, test):
    """Catalogs are dicts containing image_ids, concept_ids, content_hashes."""
    for catalog in (train, test):
        validate_catalog(**catalog)
    for key in ('image_ids', 'concept_ids', 'content_hashes'):
        if set(train[key]) & set(test[key]):
            raise ValueError(f'Train/test overlap in {key}')


@dataclass(frozen=True)
class AccessPolicy:
    setting: str
    calibration_k: int = 0
    test_query_statistics: bool = False
    target_session_statistics: bool = False

    def validate(self):
        settings = {'within_subject', 'loso_inductive', 'loso_transductive',
                    'labeled_calibration', 'labeled_calibration_transductive'}
        if self.setting not in settings or self.calibration_k < 0:
            raise ValueError('Invalid evaluation setting')
        if self.target_session_statistics:
            raise ValueError('Full target-session fitting is outside these protocols')
        if self.test_query_statistics and self.setting not in {
                'loso_transductive', 'labeled_calibration_transductive'}:
            raise ValueError('Test-query statistics require a transductive setting')
        labeled = self.setting.startswith('labeled_calibration')
        if labeled != (self.calibration_k > 0):
            raise ValueError('Calibration labels and declared setting disagree')


def select_source_config(records, sources, target):
    """Select mean source-validation score; require complete, unique source rows.

    records contain config, subject, score, role='source_validation'. Caller
    must generate these records in inner folds, never from outer target tests.
    """
    sources = set(sources)
    if not sources or target in sources:
        raise ValueError('Invalid source/target partition')
    grouped = {}
    for r in records:
        if r['subject'] not in sources or r['role'] != 'source_validation':
            raise ValueError('Non-source validation in hyperparameter selection')
        g = grouped.setdefault(r['config'], {})
        if r['subject'] in g or not torch.isfinite(torch.tensor(r['score'])):
            raise ValueError('Duplicate or nonfinite validation record')
        g[r['subject']] = r['score']
    if not grouped or any(set(g) != sources for g in grouped.values()):
        raise ValueError('Incomplete source validation')
    return min(grouped, key=lambda c: (-sum(grouped[c].values()) / len(sources), c))


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, fit_data, channelwise=False):
        if len(fit_data) < 2 or not torch.isfinite(fit_data).all():
            raise ValueError('Need finite fit data with at least two rows')
        if channelwise and fit_data.ndim != 3:
            raise ValueError('Channelwise EEG must be rows x channels x time')
        axes = (0, 2) if channelwise else (0,)
        x = fit_data.float()
        return cls(x.mean(axes, keepdim=True), x.std(axes, correction=0, keepdim=True).clamp_min(1e-6))

    def transform(self, x):
        return (x.float() - self.mean) / self.std


def csls(scores, policy, k=10):
    policy.validate()
    if not policy.test_query_statistics or k < 1:
        raise ValueError('CSLS must declare test-query statistics and positive k')
    return (2 * scores - scores.topk(min(k, scores.shape[1]), 1).values.mean(1, keepdim=True)
            - scores.topk(min(k, scores.shape[0]), 0).values.mean(0, keepdim=True))
