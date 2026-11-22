"""Dormant LIMA candidate: train-image marginal teacher, gallery-free student.

This module does not authorize training or claim feasibility. Its frozen image
bank and image-to-caption conditional must be built using optimizer rows only.
The feasibility gate is separate; no minibatch/prototype approximation is made
here. Existing 06b and B/D training code and checkpoint signatures are unchanged.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .core import positive_mask
from .joint_model import _matrix, symmetric_identity_nce
from .models import Aligner, EMB


def _supported_logsumexp(values, dim):
    # An all-excluded block has zero mass. Avoid the undefined derivative of
    # logsumexp(-inf, ...); masked rows get a constant -inf with zero gradient.
    supported = torch.isfinite(values).any(dim)
    safe = values.masked_fill(~supported.unsqueeze(dim), 0.)
    return safe.logsumexp(dim).masked_fill(~supported, -torch.inf)


def marginal_log_probs(image_logits, conditional_logits, *,
                       allowed_mediators=None, mediator_block_size=128):
    """Return log sum_j pi(j|e) q(k|j), with a frozen conditional q.

    Normalized matrix multiplication avoids a query x mediator x caption
    tensor. For extreme logits with underflow, exact log-space block reductions
    are a numerical fallback (not a sampled/prototype approximation). The fixed
    cosine/.07 model takes the matrix path. Optional exclusions precede pi
    normalization; each query needs a mediator. Only image_logits gets gradients.
    """
    _matrix(image_logits, name='EEG/image logits')
    _matrix(conditional_logits, name='image/caption logits')
    if (image_logits.shape[1] != conditional_logits.shape[0]
            or image_logits.device != conditional_logits.device):
        raise ValueError('Mediator rows or devices disagree')
    if (isinstance(mediator_block_size, bool)
            or not isinstance(mediator_block_size, int) or mediator_block_size < 1):
        raise ValueError('Mediator block size must be a positive integer')
    logits = image_logits.float()
    if allowed_mediators is not None:
        if (allowed_mediators.dtype != torch.bool
                or allowed_mediators.shape != logits.shape
                or allowed_mediators.device != logits.device
                or not allowed_mediators.any(1).all()):
            raise ValueError('Each query needs an allowed mediator')
        logits = logits.masked_fill(~allowed_mediators, -torch.inf)
    log_pi = logits.log_softmax(dim=1)
    log_q = conditional_logits.detach().float().log_softmax(dim=1)
    mass = log_pi.exp() @ log_q.exp()
    # No probability floor: clipping would change the objective and gradients.
    # Fixed bounded cosine temperatures have comfortably positive mass. The
    # slow fallback is for extreme arbitrary logits, not the planned workload.
    if (mass > torch.finfo(mass.dtype).tiny).all():
        return mass.log()
    result = None
    for start in range(0, len(log_q), mediator_block_size):
        stop = start + mediator_block_size
        term = _supported_logsumexp(
            log_pi[:, start:stop, None] + log_q[None, start:stop, :], 1)
        result = term if result is None else _supported_logsumexp(torch.stack((result, term)), 0)
    return result


def caption_mass_nll(log_probs, query_caption_ids, gallery_caption_ids):
    """Negative log probability mass of all duplicate-equivalent captions."""
    _matrix(log_probs, name='caption log probabilities')
    if log_probs.shape != (len(query_caption_ids), len(gallery_caption_ids)):
        raise ValueError('Caption identity/logit shape mismatch')
    positives = positive_mask(query_caption_ids, gallery_caption_ids, log_probs.device)
    if not positives.any(1).all():
        raise ValueError('Every caption query needs a positive')
    return -log_probs.masked_fill(~positives, -torch.inf).logsumexp(1).mean()


def duplicate_caption_kl(student_logits, teacher_log_probs, gallery_caption_ids):
    """KL on caption identities, with an entirely detached teacher target."""
    _matrix(student_logits, name='student logits')
    _matrix(teacher_log_probs, name='teacher log probabilities')
    if (student_logits.shape != teacher_log_probs.shape
            or student_logits.shape[1] != len(gallery_caption_ids)):
        raise ValueError('Student/teacher caption gallery mismatch')
    groups = {}
    for index, cid in enumerate(gallery_caption_ids):
        groups.setdefault(cid, []).append(index)
    student = student_logits.float().log_softmax(1)
    target = teacher_log_probs.detach().float()
    s = torch.stack([student[:, rows].logsumexp(1) for rows in groups.values()], 1)
    t = torch.stack([target[:, rows].logsumexp(1) for rows in groups.values()], 1)
    return F.kl_div(s, t, log_target=True, reduction='batchmean')


class LimaAligner(nn.Module):
    """Unchanged 06b backbone and the existing form of direct text readout.

    Only the marginal caption loss sends semantic gradients into EEG. The KL
    student sees detached EEG and a detached teacher, isolating this pathway.
    The train-image bank and its conditional are frozen external artifacts;
    no validation images or labels may enter that bank or its fit.
    """
    def __init__(self, n_in, dims, text_dim, rank=64, init_logits=None,
                 posterior_temperature=.07):
        super().__init__()
        dims = list(dims)
        if n_in < 1 or not dims or min(dims) < 1 or text_dim < 1 or rank < 1:
            raise ValueError('Model dimensions must be positive')
        if (not math.isfinite(posterior_temperature) or posterior_temperature <= 0
                or (init_logits is not None and len(init_logits) != len(dims))):
            raise ValueError('Invalid temperature or layer mixture')
        self.n_in, self.dims, self.text_dim = n_in, dims, text_dim
        self.posterior_temperature = float(posterior_temperature)
        # First construction preserves historical initialization for a fixed RNG.
        self.backbone = Aligner(n_in, dims, init_logits)
        self.readout_eeg = nn.Linear(EMB, rank)
        self.readout_text = nn.Linear(text_dim, rank)
        self.text_log_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def load_06b_backbone_state(self, state):
        expected = self.backbone.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('Expected a plain compatible 06b state_dict')
        for name, value in state.items():
            if (not torch.is_tensor(value) or value.shape != expected[name].shape
                    or not torch.isfinite(value).all()):
                raise ValueError('Invalid historical tensor: ' + name)
        self.backbone.load_state_dict(state, strict=True)

    def encode(self, eeg, visuals):
        if eeg.ndim < 2 or not len(eeg) or eeg[0].numel() != self.n_in:
            raise ValueError('EEG input shape mismatch')
        if not torch.isfinite(eeg).all() or len(visuals) != len(self.dims):
            raise ValueError('Nonfinite EEG or visual layer count mismatch')
        for x, width in zip(visuals, self.dims):
            _matrix(x, width, 'visual feature')
            if len(x) != len(eeg):
                raise ValueError('EEG/image row mismatch')
        return self.backbone.eeg(eeg), self.backbone.img([x.detach() for x in visuals])

    def image_scores(self, ze, zi):
        _matrix(ze, EMB, 'EEG embedding'); _matrix(zi, EMB, 'image embedding')
        return self.backbone.scale() * F.normalize(ze.float(), dim=-1) @ F.normalize(zi.float(), dim=-1).T

    def text_scores(self, ze, text):
        """Direct inference accepts only EEG embeddings and caption gallery."""
        _matrix(ze, EMB, 'EEG embedding'); _matrix(text, self.text_dim, 'text')
        e = F.normalize(self.readout_eeg(ze).float(), dim=-1)
        t = F.normalize(self.readout_text(text.detach()).float(), dim=-1)
        return self.text_log_scale.exp().clamp(max=100) * e @ t.T

    def caption_teacher(self, ze, train_images, conditional_logits,
                        query_concept_ids, mediator_concept_ids):
        """Training-only teacher excludes every same-concept image mediator.

        This prevents an optimizer example from passing its own image/caption
        through the mediator as a near one-hot target. Use the full declared
        train bank in both arms, not a query-dependent sampled approximation.
        """
        _matrix(ze, EMB, 'EEG embedding'); _matrix(train_images, EMB, 'train image bank')
        if (len(query_concept_ids) != len(ze)
                or len(mediator_concept_ids) != len(train_images)):
            raise ValueError('Query/mediator concept identity mismatch')
        mask = ~positive_mask(query_concept_ids, mediator_concept_ids, ze.device)
        logits = (F.normalize(ze.float(), dim=-1)
                  @ F.normalize(train_images.detach().float(), dim=-1).T) / self.posterior_temperature
        return marginal_log_probs(logits, conditional_logits, allowed_mediators=mask)

    def loss_terms(self, ze, zi, text, train_images, conditional_logits, *,
                   image_ids, query_caption_ids, gallery_caption_ids,
                   query_concept_ids, mediator_concept_ids,
                   marginal_weight=.1, distill_weight=.1):
        if any(not math.isfinite(w) or w < 0 for w in (marginal_weight, distill_weight)):
            raise ValueError('Loss weights must be finite and nonnegative')
        base = symmetric_identity_nce(self.image_scores(ze, zi), image_ids)
        teacher = self.caption_teacher(ze, train_images, conditional_logits,
                                       query_concept_ids, mediator_concept_ids)
        marginal = caption_mass_nll(teacher, query_caption_ids, gallery_caption_ids)
        student = self.text_scores(ze.detach(), text)
        distill = duplicate_caption_kl(student, teacher, gallery_caption_ids)
        total = base + marginal_weight * marginal + distill_weight * distill
        if not torch.isfinite(total):
            raise ValueError('Nonfinite LIMA loss')
        return {'total': total, 'base': base, 'marginal': marginal, 'distill': distill}
