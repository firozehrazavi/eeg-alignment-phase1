"""Isolated v7-inspired ideas. No automatic text, MMD or subject routing."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .models import EEGProject
from .core import positive_mask, retrieval_metrics


class ConvEEG(nn.Module):
    """Temporal/spatial/separable conv + attention, with sample-local norms.

    Inspired by the supplied v7; odd kernels preserve length and GroupNorm
    avoids fitting running batch statistics. It is not a reproduction claim.
    """
    def __init__(self, channels, samples, embedding=1024, dropout=.3):
        super().__init__()
        if channels<1 or samples<4:raise ValueError('Need channels and at least four samples')
        self.shape=(channels,samples)
        self.features=nn.Sequential(
            nn.Conv2d(1,16,(1,63),padding=(0,31),bias=False),nn.GroupNorm(4,16),nn.GELU(),
            nn.Conv2d(16,32,(channels,1),groups=16,bias=False),nn.GroupNorm(8,32),nn.GELU(),
            nn.AvgPool2d((1,4)),
            nn.Conv2d(32,32,(1,15),padding=(0,7),groups=32,bias=False),
            nn.Conv2d(32,32,1,bias=False),nn.GroupNorm(8,32),nn.GELU())
        self.attention=nn.MultiheadAttention(32,4,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(32)
        position=torch.arange(samples//4).float()[:,None]
        frequency=torch.exp(torch.arange(0,32,2).float()*(-math.log(10000.)/32))
        pe=torch.zeros(samples//4,32);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position',pe[None])
        self.pool=nn.AdaptiveAvgPool1d(8)
        self.dropout=nn.Dropout(dropout)
        self.head=EEGProject(32*8,d=embedding,drop=dropout)

    def forward(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=self.shape:raise ValueError('EEG shape mismatch')
        h=self.features(x.unsqueeze(1)).squeeze(2).transpose(1,2)
        h=h+self.position.to(h.dtype)
        a,_=self.attention(h,h,h,need_weights=False)
        h=self.pool(self.norm(h+a).transpose(1,2)).flatten(1)
        return self.head(self.dropout(h))


def source_training_rows(fit_images, validation_images, n_images, sources, target=None):
    """Flatten subject-major rows only after proving image split disjointness."""
    if not sources or len(sources)!=len(set(sources)) or target in sources:
        raise ValueError('Invalid source partition')
    fit=list(fit_images);val=list(validation_images)
    if (not fit or not val or len(fit)!=len(set(fit)) or len(val)!=len(set(val))
            or set(fit)&set(val) or any(i<0 or i>=n_images for i in fit+val)):
        raise ValueError('Invalid training/validation image rows')
    return np.array([s*n_images+i for s in range(len(sources)) for i in fit],dtype=np.int64)


def stratified_rows(allowed_rows, subject_ids, batch_size, rng):
    """Return GLOBAL row IDs, not positions in subject_ids[allowed_rows]."""
    rows=np.asarray(allowed_rows,dtype=np.int64);ids=np.asarray(subject_ids)
    if not len(rows) or len(set(rows.tolist()))!=len(rows) or rows.min()<0 or rows.max()>=len(ids):
        raise ValueError('Invalid allowed rows')
    subjects=np.unique(ids[rows])
    if batch_size<len(subjects):raise ValueError('Batch must fit at least one row per subject')
    queues=[rng.permutation(rows[ids[rows]==s]).tolist() for s in subjects]
    batches=[]
    while any(queues):
        batch=[]
        while len(batch)<batch_size and any(queues):
            for s in rng.permutation(len(queues)):
                if queues[s] and len(batch)<batch_size:batch.append(queues[s].pop())
        batches.append(np.asarray(batch,dtype=np.int64))
    return batches


def augment_eeg(x, noise_std=0., cut_fraction=0., channel_drop=0.):
    if noise_std<0 or not 0<=cut_fraction<1 or not 0<=channel_drop<1:
        raise ValueError('Invalid augmentation settings')
    if not any((noise_std,cut_fraction,channel_drop)):return x
    y=x.clone()
    if noise_std:y=y+noise_std*torch.randn_like(y)
    if cut_fraction:
        length=max(1,int(x.shape[-1]*cut_fraction))
        starts=torch.randint(0,x.shape[-1]-length+1,(len(x),1,1),device=x.device)
        time=torch.arange(x.shape[-1],device=x.device)[None,None]
        y=y*~((time>=starts)&(time<starts+length))
    if channel_drop:y=y*(torch.rand(len(x),x.shape[1],1,device=x.device)>=channel_drop)
    return y


def margin_multi_positive_loss(ze,zi,scale,eeg_ids,image_ids,margin=0.):
    """Apply angular-cosine additive margin to ALL exact-identity positives."""
    if margin<0 or len(ze)!=len(eeg_ids) or len(zi)!=len(image_ids):raise ValueError('Invalid margin/IDs')
    mask=positive_mask(eeg_ids,image_ids,ze.device)
    if not mask.any(0).all() or not mask.any(1).all():raise ValueError('Missing positive')
    logits=scale*(F.normalize(ze.float(),dim=-1)@F.normalize(zi.float(),dim=-1).T-margin*mask)
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
    positive=logits.masked_fill(~mask,-torch.inf)
    return ((logits.logsumexp(1)-positive.logsumexp(1)).mean()+
            (logits.logsumexp(0)-positive.logsumexp(0)).mean())/2


def margin_at_epoch(config,epoch):
    stage=config.get('margin_stage')
    if not stage:return 0.
    if config['objective']!='multi_positive' or not 1<=stage['start_epoch']<=config['epochs'] or stage['margin']<0:
        raise ValueError('Margin stages require identity-aware loss and a valid fixed schedule')
    return stage['margin'] if epoch>=stage['start_epoch'] else 0.


def align_score_bundle(bundle,query_ids,gallery_ids):
    q,g=bundle['query_ids'],bundle['gallery_ids']
    if len(q)!=len(set(q)) or len(g)!=len(set(g)) or set(q)!=set(query_ids) or set(g)!=set(gallery_ids):
        raise ValueError('Ensemble query/gallery identities disagree')
    scores=bundle['scores']
    if scores.shape!=(len(q),len(g)) or not torch.isfinite(scores).all():raise ValueError('Invalid score matrix')
    return scores[[q.index(i) for i in query_ids]][:,[g.index(i) for i in gallery_ids]]


def select_score_ensemble(validation,weight_candidates,sources,target=None):
    """Fit weights on declared source validation only; test bundles are rejected.

    validation: model -> subject -> {role,score_type,query_ids,gallery_ids,scores}.
    Candidate weights are predeclared; missing seeds/models are errors.
    """
    if not sources or len(sources)!=len(set(sources)) or target in sources or not validation:
        raise ValueError('Invalid source-only ensemble selection')
    for model,subjects in validation.items():
        if set(subjects)!=set(sources):raise ValueError('Incomplete source validation')
        for b in subjects.values():
            if b['role']!='source_validation' or b['score_type']!='cosine':
                raise ValueError('Only source-validation cosine scores may select weights')
    best=None
    for candidate in weight_candidates:
        if set(candidate)!=set(validation) or any(not math.isfinite(w) or w<0 for w in candidate.values()) or sum(candidate.values())<=0:
            raise ValueError('Supply finite weights for every declared model')
        weights={k:v/sum(candidate.values()) for k,v in candidate.items()}
        vals=[]
        for s in sources:
            first=next(iter(validation.values()))[s]
            q=sorted(first['query_ids']);g=sorted(first['gallery_ids'])
            scores=sum(w*align_score_bundle(validation[k][s],q,g) for k,w in weights.items())
            vals.append(retrieval_metrics(scores,q,g,len(g))['top1'])
        score=float(np.mean(vals))
        if best is None or score>best['source_val_top1']:best=dict(weights=weights,source_val_top1=score,sources=list(sources))
    if best is None:raise ValueError('No predeclared weight candidates')
    return best


def apply_score_ensemble(selection,bundles):
    if set(selection['weights'])!=set(bundles):raise ValueError('Missing ensemble members')
    first=next(iter(bundles.values()));q=sorted(first['query_ids']);g=sorted(first['gallery_ids'])
    if any(b['score_type']!='cosine' for b in bundles.values()):raise ValueError('Mixed scoring protocols')
    scores=sum(w*align_score_bundle(bundles[k],q,g) for k,w in selection['weights'].items())
    return scores,q,g
