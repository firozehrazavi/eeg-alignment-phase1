"""Experimental triadic supervision with image-free EEG-to-text inference.

The CP/tensor-product interaction is established machinery, not a novelty claim.
The experiment tests whether text unexplained by a fixed, fit-only linear visual
predictor supplies useful *joint* supervision to the preserved 06b backbone.
Image retrieval uses the original EEG/image cosine score. Text retrieval uses a
separate EEG/text student, distilled from the triadic teacher during training.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .core import positive_mask
from .models import Aligner, EMB


def _matrix(x, width=None, name='input'):
    if x.ndim != 2 or not len(x) or (width is not None and x.shape[1] != width):
        raise ValueError(f'Invalid {name} shape')
    if not torch.isfinite(x).all():
        raise ValueError(f'Nonfinite {name}')


def symmetric_identity_nce(logits, query_ids, candidate_ids=None):
    """Symmetric negative log mass of all exact-identity positives."""
    candidate_ids = query_ids if candidate_ids is None else candidate_ids
    _matrix(logits, name='logits')
    if logits.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError('Identity/logit shape mismatch')
    positives = positive_mask(query_ids, candidate_ids, logits.device)
    if not positives.any(0).all() or not positives.any(1).all():
        raise ValueError('Every anchor in both directions needs a positive')
    pos = logits.masked_fill(~positives, -torch.inf)
    return ((logits.logsumexp(1) - pos.logsumexp(1)).mean()
            + (logits.logsumexp(0) - pos.logsumexp(0)).mean()) / 2


class JointAligner(nn.Module):
    """06b backbone plus a train-time rank-r three-way energy and text student.

    For candidate j, H_ij = <tanh(U e_i), tanh(V v_j)*tanh(W r_j)> / sqrt(r).
    A single loss on B + alpha*H couples all three modalities. The base loss
    remains active, and no text features are required by image_scores.
    """
    def __init__(self, n_in, dims, text_dim, rank=64, init_logits=None, alpha=1.,
                 temperature=2., semantic_correction=False):
        super().__init__()
        dims = list(dims)
        if n_in < 1 or not dims or min(dims) < 1 or text_dim < 1 or rank < 1:
            raise ValueError('Model dimensions must be positive')
        if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid joint scale or temperature')
        if init_logits is not None and len(init_logits) != len(dims):
            raise ValueError('Layer mixture shape mismatch')
        self.n_in, self.dims, self.text_dim, self.rank = n_in, dims, text_dim, rank
        if not isinstance(semantic_correction, bool):
            raise ValueError('semantic_correction must be a bool')
        self.alpha, self.temperature = float(alpha), float(temperature)
        self.semantic_correction = semantic_correction
        # Construct this first: a matched RNG seed yields precisely 06b weights.
        self.backbone = Aligner(n_in, dims, init_logits)
        self.joint_eeg = nn.Linear(EMB, rank, bias=False)
        self.joint_image = nn.Linear(EMB, rank, bias=False)
        self.joint_text = nn.Linear(text_dim, rank, bias=False)
        self.readout_eeg = nn.Linear(EMB, rank)
        self.readout_text = nn.Linear(text_dim, rank)
        self.text_log_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def encode(self, eeg, visuals):
        if eeg.ndim < 2 or not len(eeg) or eeg[0].numel() != self.n_in:
            raise ValueError('EEG input shape mismatch')
        if not torch.isfinite(eeg).all() or len(visuals) != len(self.dims):
            raise ValueError('Nonfinite EEG or wrong number of visual layers')
        for x, width in zip(visuals, self.dims):
            _matrix(x, width, 'visual feature')
        if any(len(x) != len(eeg) for x in visuals):
            raise ValueError('EEG and visual rows are not aligned')
        return self.backbone.eeg(eeg), self.backbone.img([x.detach() for x in visuals])

    def load_06b_backbone_state(self, state):
        """Load the plain Aligner.state_dict saved by the historical 06b code."""
        expected = self.backbone.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('Checkpoint is not a plain, compatible 06b Aligner state_dict')
        for key, value in state.items():
            if (not torch.is_tensor(value) or value.shape != expected[key].shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid 06b backbone tensor: {key}')
        self.backbone.load_state_dict(state, strict=True)

    def image_scores(self, ze, zi):
        _matrix(ze, EMB, 'EEG embedding'); _matrix(zi, EMB, 'image embedding')
        return self.backbone.scale() * F.normalize(ze.float(), dim=-1) @ F.normalize(zi.float(), dim=-1).T

    def text_scores(self, ze, text):
        """Direct retrieval: no image features, image gallery, or residualizer."""
        _matrix(ze, EMB, 'EEG embedding'); _matrix(text, self.text_dim, 'text')
        e = F.normalize(self.readout_eeg(ze).float(), dim=-1)
        t = F.normalize(self.readout_text(text.detach()).float(), dim=-1)
        return self.text_log_scale.exp().clamp(max=100) * e @ t.T

    def joint_scores(self, ze, zi, residual_text):
        _matrix(residual_text, self.text_dim, 'residual text')
        if len(zi) != len(residual_text):
            raise ValueError('Image and residual-text candidates must align')
        base = self.image_scores(ze, zi)
        e = self.joint_eeg(ze).tanh()
        v = self.joint_image(zi).tanh()
        t = self.joint_text(residual_text.detach()).tanh()
        # Opt-in B'/D' experiment: use the frozen image score as a context,
        # while allowing this loss to update the encoder only via the
        # genuinely triadic interaction. The base image loss remains live
        # separately in loss_terms; the legacy A/B/C/D path is unchanged.
        teacher_base = base.detach() if self.semantic_correction else base
        return teacher_base + self.alpha * (e @ (v * t).T) / math.sqrt(self.rank)

    def loss_terms(self, ze, zi, text, residual_text, ids, joint_weight=.1,
                   distill_weight=.1, image_ids=None, caption_ids=None):
        if any(not math.isfinite(w) or w < 0 for w in (joint_weight, distill_weight)):
            raise ValueError('Loss weights must be finite and nonnegative')
        image_ids = ids if image_ids is None else image_ids
        if len(text) != len(zi):
            raise ValueError('Text and image candidates must align')
        base_logits = self.image_scores(ze, zi)
        base = symmetric_identity_nce(base_logits, ids, image_ids)
        teacher = self.joint_scores(ze, zi, residual_text)
        joint = symmetric_identity_nce(teacher, ids, image_ids)
        # Neither teacher gradients nor student gradients enter the backbone.
        # Image effects can consequently be attributed to the joint objective.
        student = self.text_scores(ze.detach(), text)
        temp = self.temperature
        target_probs = F.softmax(teacher.detach() / temp, dim=1)
        if caption_ids is not None:
            if len(caption_ids) != len(zi):
                raise ValueError('Caption IDs must match candidate rows')
            # Image teacher cannot demand distinguishing two identical
            # caption candidates; aggregate its mass by exact text identity.
            groups = {}
            for j, cid in enumerate(caption_ids):
                groups.setdefault(cid, []).append(j)
            target_probs = target_probs.clone()
            for positions in groups.values():
                if len(positions) > 1:
                    target_probs[:, positions] = (target_probs[:, positions]
                            .sum(dim=1, keepdim=True) / len(positions))
        distill = F.kl_div(F.log_softmax(student / temp, dim=1),
                          target_probs, reduction='batchmean') * temp**2
        total = base + joint_weight * joint + distill_weight * distill
        if not torch.isfinite(total):
            raise ValueError('Nonfinite joint loss')
        return {'total': total, 'base': base, 'joint': joint, 'distill': distill}

    def loss(self, ze, zi, text, residual_text, ids, joint_weight=.1,
             distill_weight=.1, image_ids=None, caption_ids=None):
        return self.loss_terms(ze, zi, text, residual_text, ids, joint_weight,
                               distill_weight, image_ids, caption_ids)['total']


class Residualizer(nn.Module):
    """Frozen ridge prediction of text from a fixed random visual sketch.

    Given already fit-normalized frozen visuals X, S=X R, fit only on I:
      A=(S_c.T S_c + |I|*ridge*Identity)^(-1) S_c.T T_c.
    Residuals are T - ((S-mean_S)A+mean_T). This removes only the fitted linear
    component; it does NOT establish information absent from the visual input.
    No normalization or fitting occurs in transform. Never refit on evaluation.
    """
    def __init__(self, dims, text_dim, rank):
        super().__init__()
        if not dims or min(dims) < 1 or text_dim < 1 or rank < 1:
            raise ValueError('Residualizer dimensions must be positive')
        self.register_buffer('widths', torch.tensor(dims, dtype=torch.long))
        self.register_buffer('projection', torch.zeros(sum(dims), rank))
        self.register_buffer('coefficients', torch.zeros(rank, text_dim))
        self.register_buffer('sketch_mean', torch.zeros(rank))
        self.register_buffer('text_mean', torch.zeros(text_dim))
        self.register_buffer('fit_indices', torch.empty(0, dtype=torch.long))

    @staticmethod
    def _sketch(visuals, projection, widths, rows=None):
        result, offset, n = None, 0, None
        if len(visuals) != len(widths):
            raise ValueError('Residualizer visual layer count mismatch')
        for value, width in zip(visuals, widths):
            x = torch.as_tensor(value)
            if x.ndim != 2 or x.shape[1] != width or (n is not None and len(x) != n):
                raise ValueError('Residualizer visual shape mismatch')
            n = len(x)
            # Select fitting rows before examining their values: held-out values
            # cannot affect either the fit or its validation/acceptance decision.
            if rows is not None:
                x = x[rows.to(x.device)]
            x = x.detach().to(device=projection.device, dtype=projection.dtype)
            _matrix(x, width, 'residualizer visual')
            part = x @ projection[offset:offset + width]
            result = part if result is None else result + part
            offset += width
        return result

    @classmethod
    @torch.no_grad()
    def fit(cls, visuals, text, fit_indices, rank=64, ridge=1., seed=0):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('Ridge must be finite and positive')
        text = torch.as_tensor(text)
        if text.ndim != 2 or not len(text) or not visuals:
            raise ValueError('Residualizer needs aligned matrix inputs')
        rows = torch.as_tensor(fit_indices)
        if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError('Fitting indices must be a one-dimensional integer list')
        rows = rows.long().cpu()
        if len(rows) < 2 or len(rows.unique()) != len(rows) or rows.min() < 0 or rows.max() >= len(text):
            raise ValueError('Invalid or duplicated fitting indices')
        dims = []
        for x in visuals:
            if len(x.shape) != 2 or len(x) != len(text):
                raise ValueError('Residualizer inputs must have aligned rows')
            dims.append(x.shape[1])
        fitted = cls(dims, text.shape[1], rank)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        projection = torch.randn(sum(dims), rank, dtype=torch.float64,
                                 generator=generator) / math.sqrt(sum(dims))
        sketch = cls._sketch(visuals, projection, dims, rows)
        target = text[rows.to(text.device)].detach().to(device='cpu', dtype=torch.float64)
        _matrix(target, text.shape[1], 'fitting text')
        sm, tm = sketch.mean(0), target.mean(0)
        centered, target_centered = sketch - sm, target - tm
        gram = centered.T @ centered + len(rows) * ridge * torch.eye(rank, dtype=torch.float64)
        coefficients = torch.linalg.solve(gram, centered.T @ target_centered)
        fitted.projection.copy_(projection.float())
        fitted.coefficients.copy_(coefficients.float())
        fitted.sketch_mean.copy_(sm.float())
        fitted.text_mean.copy_(tm.float())
        fitted.fit_indices = rows.clone()
        return fitted

    @torch.no_grad()
    def transform(self, visuals, text):
        if len(self.fit_indices) < 2:
            raise ValueError('Residualizer has not been fitted')
        sketch = self._sketch(visuals, self.projection, self.widths.tolist())
        text = torch.as_tensor(text).detach().to(self.projection)
        _matrix(text, self.coefficients.shape[1], 'text')
        if len(text) != len(sketch):
            raise ValueError('Residualizer text/visual row mismatch')
        return text - ((sketch - self.sketch_mean) @ self.coefficients + self.text_mean)

    @classmethod
    def from_state_dict(cls, state):
        """Reconstruct a fitted object from tensor-only checkpoint state."""
        obj = cls(state['widths'].tolist(), state['coefficients'].shape[1],
                  state['coefficients'].shape[0])
        obj.fit_indices = torch.empty_like(state['fit_indices'])
        obj.load_state_dict(state)
        return obj
