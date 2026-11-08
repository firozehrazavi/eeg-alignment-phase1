"""Architecture copied unchanged from 06b cell 5 (base bd1046d2)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
EMB = 1024

class EEGProject(nn.Module):
    def __init__(self, n_in, d=EMB, drop=0.3):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.res = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.Dropout(drop))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = self.inp(x.flatten(1)); return self.norm(h + self.res(h))

class ImageSide(nn.Module):
    def __init__(self, dims, init_logits=None):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, EMB) for d in dims])
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float) if init_logits is not None else torch.zeros(len(dims)),
                                   requires_grad=len(dims) > 1)
    def weights(self): return F.softmax(self.logits, 0)
    def forward(self, feats):                   # feats: list of (N, d_l), same rows
        w = self.weights()
        return sum(w[i] * p(f) for i, (p, f) in enumerate(zip(self.proj, feats)))

class Aligner(nn.Module):
    def __init__(self, n_in, dims, init_logits=None):
        super().__init__()
        self.eeg, self.img = EEGProject(n_in), ImageSide(dims, init_logits)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def scale(self): return self.log_scale.exp().clamp(max=100)
