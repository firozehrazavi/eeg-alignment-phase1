"""RPA-inspired InternViT adapter; not an official RPA reproduction."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .models import EEGProject, ImageSide


class ResidualPatchAdapter(nn.Module):
    """Spatial residual bottleneck, GAP, projection (RPA appendix architecture).

    Input B x H*W x D excludes CLS. Layer/backbone are explicit cache metadata.
    No attention or token pooling precedes the spatial residual branch.
    """
    def __init__(self, width, grid, embedding=1024, bottleneck=256, groups=32):
        super().__init__()
        if width % groups or bottleneck % groups or min(grid) < 1:
            raise ValueError('Widths must be divisible by GroupNorm groups')
        self.width, self.grid = width, tuple(grid)
        self.branch = nn.Sequential(
            nn.Conv2d(width, bottleneck, 1), nn.GroupNorm(groups, bottleneck), nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1),
            nn.GroupNorm(groups, bottleneck), nn.GELU(),
            nn.Conv2d(bottleneck, width, 1), nn.GroupNorm(groups, width), nn.GELU())
        self.projection = nn.Linear(width, embedding)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (math.prod(self.grid), self.width):
            raise ValueError('Expected CLS-free tokens B x (H*W) x D')
        x = tokens.transpose(1, 2).reshape(len(tokens), self.width, *self.grid)
        return self.projection((x + self.branch(x)).mean((2, 3)))


class PatchAligner(nn.Module):
    def __init__(self, n_in, pooled_dims, width, grid, arm, embedding=1024,
                 bottleneck=256, groups=32):
        super().__init__()
        self.arm = arm
        self.eeg = EEGProject(n_in, d=embedding)
        if arm == 'baseline':
            if embedding != 1024:
                raise ValueError('Historical baseline architecture requires embedding=1024')
            self.img = ImageSide(pooled_dims, [-2., -1., 0., -1., -2.] if len(pooled_dims) == 5 else None)
        elif arm == 'mean':
            self.img = nn.Linear(width, embedding)
        elif arm == 'patch':
            self.img = ResidualPatchAdapter(width, grid, embedding, bottleneck, groups)
        else:
            raise ValueError('Unknown arm')
        self.log_scale = nn.Parameter(torch.tensor(math.log(1/.07)))

    def scale(self):
        return self.log_scale.exp().clamp(max=100)

    def visual(self, pooled=None, tokens=None, chunk=16):
        if self.arm == 'baseline':
            return self.img(pooled)
        if self.arm == 'mean':
            return self.img(tokens.mean(1).to(self.log_scale.device))
        # GroupNorm is sample-local, so chunking preserves the objective/batch.
        # Keep the large input on CPU; each checkpoint transfers its own chunk.
        def project(t):
            return self.img(t.to(self.log_scale.device))
        return torch.cat([checkpoint(project, t, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else project(t)
                          for t in tokens.split(chunk)])
