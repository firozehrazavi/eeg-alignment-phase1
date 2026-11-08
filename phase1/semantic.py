"""Optional auxiliary semantic path. Frozen features enter; visual path stays intact."""
import torch
from torch import nn
import torch.nn.functional as F


class SemanticHead(nn.Module):
    def __init__(self, eeg_dim, text_dim, visual_dim, dim=256):
        super().__init__()
        self.eeg = nn.Linear(eeg_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.visual = nn.Linear(visual_dim, dim)

    def forward(self, eeg, text, visual, positive, ambiguous, weight=0.1,
                adaptive=True, grounding=0.5):
        """Image-grounded paired targets; ambiguous nonpositives are excluded.

        positive/ambiguous: rows x semantic candidates, curated from identities
        and metadata, never target EEG or test labels. No related image becomes
        an image-retrieval positive. Gate is detached; no gradient shortcut via
        suppressing the gate. Visual/text grounding is trained on source pairs.
        """
        if weight < 0 or not 0 <= grounding <= 1:
            raise ValueError('Invalid semantic weights')
        e = F.normalize(self.eeg(eeg), dim=-1)
        t = F.normalize(self.text(text.detach()), dim=-1)
        v = F.normalize(self.visual(visual.detach()), dim=-1)
        if positive.shape != (len(e), len(t)) or ambiguous.shape != positive.shape:
            raise ValueError('Semantic mask shape mismatch')
        if positive.dtype != torch.bool or ambiguous.dtype != torch.bool:
            raise ValueError('Semantic masks must be boolean')
        if not positive.any(1).all():
            raise ValueError('Every EEG needs a semantic positive')
        prototype = F.normalize(grounding * v + (1 - grounding) * t, dim=-1)
        logits = e @ prototype.T / 0.07
        allowed = ~ambiguous | positive
        row_loss = (logits.masked_fill(~allowed, -torch.inf).logsumexp(1)
                    - logits.masked_fill(~positive, -torch.inf).logsumexp(1))
        agreement = ((t * v).sum(1).detach().clamp(min=0))
        gate = (positive.float() @ agreement) / positive.sum(1)
        if not adaptive:
            gate = torch.ones_like(gate)
        # Alignment term prevents the two source projection heads drifting apart.
        return weight * ((gate * row_loss).mean() + (1 - (t * v).sum(1)).mean())
