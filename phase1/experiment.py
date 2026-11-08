"""Paired visual-objective experiment over explicit, prevalidated tensors.

No implicit cache loading. Dataset preprocessing provenance remains the caller's
responsibility; use only pre-z-score EEG with documented upstream preprocessing.
"""
import copy
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from .core import clip_loss, multi_positive_loss, retrieval_metrics, Standardizer, concept_split
from .models import Aligner


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def paired_train(eeg, image_ids, concepts, targets, *, seed=0, split_seed=0,
                 epochs=60, batch=1024, lr=1e-4, wd=1e-4, eval_every=5,
                 init_logits=None, model_factory=None):
    """eeg: source subjects x images x channels x time; targets: layer list.

    All source subjects validate on the same held-out concepts. This selects
    checkpoints, NOT outer-fold architecture hyperparameters: those must be
    fixed by inner subject folds before this call. No target EEG is accepted.
    The only difference between arms is the contrastive objective.
    """
    if eeg.ndim != 4 or eeg.shape[1] != len(image_ids) or len(concepts) != len(image_ids):
        raise ValueError('Expected source x image x channel x time and aligned metadata')
    if len(set(image_ids)) != len(image_ids) or any(len(t) != len(image_ids) for t in targets):
        raise ValueError('Image catalog must be unique and aligned')
    fit, val = concept_split(concepts, split_seed)
    source_count, n = eeg.shape[:2]
    eeg_norm = Standardizer.fit(eeg[:, fit].flatten(0, 1), channelwise=True)
    x = eeg_norm.transform(eeg.flatten(0, 1))
    norms = [Standardizer.fit(t[fit]) for t in targets]
    ts = [norm.transform(t) for norm, t in zip(norms, targets)]
    train_rows = torch.tensor([s*n+i for s in range(source_count) for i in fit])
    if len(train_rows) < 2 or batch < 2 or epochs < 1:
        raise ValueError('Insufficient training rows or invalid training budget')
    make = model_factory or (lambda: Aligner(x[0].numel(), [t.shape[1] for t in ts], init_logits))
    seed_all(seed)
    initial = copy.deepcopy(make().to(x.device).state_dict())
    results = {}
    for objective in ('diagonal', 'multi_positive'):
        seed_all(seed)
        model = make().to(x.device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        steps_per_epoch = sum(len(train_rows[i:i+batch]) >= 2 for i in range(0, len(train_rows), batch))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                        total_steps=epochs*steps_per_epoch, pct_start=0.05)
        best, best_state, history = -1, None, []
        for epoch in range(1, epochs+1):
            model.train()
            perm = train_rows[torch.randperm(len(train_rows))]
            losses = []
            for start in range(0, len(perm), batch):
                rows = perm[start:start+batch]
                if len(rows) < 2:
                    continue
                images = rows % n
                ze, zi = model.eeg(x[rows]), model.img([t[images] for t in ts])
                ids = [image_ids[i] for i in images.tolist()]
                loss = (clip_loss(ze, zi, model.scale()) if objective == 'diagonal' else
                        multi_positive_loss(ze, zi, model.scale(), ids, ids))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            if epoch % eval_every == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    gallery = F.normalize(model.img([t[val] for t in ts]), dim=-1)
                    scores = []
                    for s in range(source_count):
                        q = F.normalize(model.eeg(x[s*n + torch.tensor(val)]), dim=-1)
                        ids = [image_ids[i] for i in val]
                        scores.append(retrieval_metrics(q @ gallery.T, ids, ids, len(ids))['top1'])
                score = sum(scores)/len(scores)
                history.append({'epoch': epoch, 'source_val_top1': score, 'train_loss': sum(losses)/len(losses)})
                if score > best:
                    best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        model.eval()
        results[objective] = {'model': model, 'history': history, 'best_epoch': best_epoch}
    return results, eeg_norm, norms, {'fit': fit, 'validation': val}


@torch.no_grad()
def evaluate(model, eeg, targets, eeg_norm, target_norms, query_ids, gallery_ids):
    model.eval()
    z = F.normalize(model.eeg(eeg_norm.transform(eeg)), dim=-1)
    t = F.normalize(model.img([s.transform(x) for s, x in zip(target_norms, targets)]), dim=-1)
    return retrieval_metrics(z @ t.T, query_ids, gallery_ids)
