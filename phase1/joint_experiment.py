"""Bounded A/B/C/D training on official TRAINING partition only.

python -m phase1.joint_experiment --manifest train.json --config configs/joint_pilot.json --output NEW_DIR
"""
import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core import Standardizer, multi_positive_loss, retrieval_metrics
from .experiment import seed_all
from .joint_data import load_training, split_three, derangement, digest
from .joint_model import JointAligner, Residualizer, symmetric_identity_nce


@torch.no_grad()
def evaluate(model, eeg, visual, text, rows, image_ids, caption_ids, device, batch):
    model.eval()
    ze, zi = [], []
    for start in range(0, len(rows), batch):
        ix = rows[start:start + batch]
        e, v = model.encode(eeg[ix].to(device), [x[ix].to(device) for x in visual])
        ze.append(e); zi.append(v)
    ze, zi = torch.cat(ze), torch.cat(zi)
    si = model.image_scores(ze, zi).cpu()
    st = model.text_scores(ze, text[rows].to(device)).cpu()
    ids, tids = [image_ids[i] for i in rows], [caption_ids[i] for i in rows]
    im = retrieval_metrics(si, ids, ids, len(rows))
    tx = retrieval_metrics(st, tids, tids, len(rows))
    return {**{'image_' + k: v for k, v in im.items()}, **{'text_' + k: v for k, v in tx.items()}}, dict(image_scores=si, text_scores=st, image_ids=ids, caption_ids=tids)


def run(manifest_path, config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    required = {'subjects', 'seeds', 'epochs', 'batch', 'lr', 'weight_decay', 'device', 'max_seconds'}
    if not required <= cfg.keys() or cfg.get('setting', 'within_subject') not in ('within_subject', 'loso'):
        raise ValueError('Incomplete configuration or unsupported setting')
    if cfg['epochs'] < 1 or cfg['batch'] < 2 or cfg['max_seconds'] <= 0:
        raise ValueError('Invalid experiment budget')
    if len(set(cfg['subjects'])) != len(cfg['subjects']) or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Duplicate subject or model seed')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but this Python environment has no available GPU')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    code = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    provenance = dict(config=cfg, manifest_sha256=digest(manifest_path), code_sha256=code,
                      python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      device=str(device), official_test_opened=False, status='running',
                      benchmark_use='development only; all arrays from official training partition')
    def save_status(status):
        provenance['status'] = status
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    save_status('running')
    try:
        for target in cfg.get('targets', cfg['subjects']):
            if target not in cfg['subjects']:
                raise ValueError('Unknown target')
            sources = [target] if cfg.get('setting', 'within_subject') == 'within_subject' else [s for s in cfg['subjects'] if s != target]
            m, raw_v, raw_t, raw_e, hashes = load_training(manifest_path, sources)
            cat = m['catalog']; ids = cat['image_ids']; cids = cat['caption_ids']
            fit, tune, hold = split_three(cat['concept_ids'], cfg.get('split_seed', 20261117))
            en = Standardizer.fit(torch.cat([x[fit] for x in raw_e.values()]), channelwise=True)
            vn = [Standardizer.fit(x[fit]) for x in raw_v]
            tn = Standardizer.fit(raw_t[fit])
            visual = [norm.transform(x) for norm, x in zip(vn, raw_v)]
            text = tn.transform(raw_t)
            eeg = {s: en.transform(x) for s, x in raw_e.items()}
            residualizer = Residualizer.fit(visual, text, fit, rank=cfg.get('rank', 64), ridge=cfg.get('ridge', 1.), seed=cfg.get('split_seed', 20261117))
            residual = residualizer.transform(visual, text)
            provenance[target] = dict(sources=sources, input_sha256=hashes, fit=fit, tune=tune, holdout=hold,
                residual_to_text_fit_energy=float(residual[fit].square().sum() / text[fit].square().sum().clamp_min(1e-9)))
            save_status('running')
            train_rows = [(s, i) for s in sources for i in fit]
            for seed in cfg['seeds']:
                seed_all(seed)
                initial = JointAligner(next(iter(eeg.values()))[0].numel(), [x.shape[1] for x in visual], text.shape[1], rank=cfg.get('rank', 64))
                state0 = copy.deepcopy(initial.state_dict())
                # Every arm uses fixed semantics across epochs, with permutations confined to fit.
                p = derangement([cat['concept_ids'][i] for i in fit], seed + 7000)
                shuffled = residual.clone(); shuffled[fit] = residual[torch.tensor(fit)[p]]
                g = torch.Generator().manual_seed(seed + 8000)
                pseudo = torch.randn(residual.shape, generator=g) * residual[fit].std(0, correction=0).clamp_min(1e-6)
                for arm in cfg.get('arms', ['A', 'B', 'C', 'D']):
                    if arm not in 'ABCD' or len(arm) != 1:
                        raise ValueError('Unknown arm')
                    model = copy.deepcopy(initial).to(device); model.load_state_dict(state0)
                    seed_all(seed + 10000)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
                    rng = np.random.default_rng(seed)
                    best, best_state, best_epoch = -float('inf'), None, None
                    run_name = f'{target}_s{seed}_{arm}'
                    train_semantic = {'A': residual, 'B': residual, 'C': pseudo, 'D': shuffled}[arm]
                    for epoch in range(1, cfg['epochs'] + 1):
                        model.train(); losses = []; epoch_start = time.monotonic()
                        perm = rng.permutation(len(train_rows))
                        for start in range(0, len(perm), cfg['batch']):
                            if time.monotonic() - started > cfg['max_seconds']:
                                raise TimeoutError('Configured total experiment wall-time budget exhausted')
                            items = [train_rows[j] for j in perm[start:start + cfg['batch']]]
                            if len(items) < 2: continue
                            ix = [i for _, i in items]
                            x = torch.stack([eeg[s][i] for s, i in items]).to(device)
                            ze, zi = model.encode(x, [v[ix].to(device) for v in visual])
                            identity = [ids[i] for i in ix]
                            if arm == 'A':
                                loss = multi_positive_loss(ze, zi, model.backbone.scale(), identity, identity)
                            else:
                                loss = model.loss(ze, zi, text[ix].to(device), train_semantic[ix].to(device), identity,
                                    joint_weight=cfg.get('joint_weight', .1), distill_weight=cfg.get('distill_weight', .1))
                            # Identical supervised readout in all arms, detached from backbone.
                            loss = loss + cfg.get('readout_weight', .1) * symmetric_identity_nce(model.text_scores(ze.detach(), text[ix].to(device)), identity, identity)
                            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
                            optimizer.zero_grad(set_to_none=True); loss.backward()
                            # Independent clipping prevents student gradients from rescaling backbone steps.
                            main = [p for name, p in model.named_parameters() if not name.startswith(('readout_', 'text_log_scale'))]
                            student = [p for name, p in model.named_parameters() if name.startswith(('readout_', 'text_log_scale'))]
                            torch.nn.utils.clip_grad_norm_(main, 1., error_if_nonfinite=True)
                            torch.nn.utils.clip_grad_norm_(student, 1., error_if_nonfinite=True)
                            optimizer.step(); losses.append(float(loss.detach()))
                        scheduler.step()
                        vals = [evaluate(model, eeg[s], visual, text, tune, ids, cids, device, cfg['batch'])[0] for s in sources]
                        score = float(np.mean([v['image_top1'] for v in vals]))
                        log = dict(run=run_name, epoch=epoch, loss=float(np.mean(losses)), tune_image_top1=score, seconds=time.monotonic()-epoch_start)
                        with (out / 'epochs.jsonl').open('a') as f: f.write(json.dumps(log) + '\n')
                        print(json.dumps(log), flush=True)
                        if score > best:
                            best, best_epoch = score, epoch
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch), out / (run_name + '_latest.pt'))
                    model.load_state_dict(best_state); model.eval()
                    torch.save(dict(model=best_state, config=cfg, eeg_mean=en.mean, eeg_std=en.std,
                        visual_mean=[n.mean for n in vn], visual_std=[n.std for n in vn], text_mean=tn.mean,
                        text_std=tn.std, residualizer=residualizer.state_dict(), sources=sources,
                        selected_epoch=best_epoch, model_dimensions=dict(n_in=next(iter(eeg.values()))[0].numel(), dims=[v.shape[1] for v in visual], text_dim=text.shape[1], rank=cfg.get('rank',64))), out / (run_name + '_best.pt'))
                    if cfg.get('setting', 'within_subject') == 'loso':
                        # Held-out subject opened ONLY after source-only checkpoint selection.
                        _, _, _, target_eeg, _ = load_training(manifest_path, [target])
                        evaluation_eeg = en.transform(target_eeg[target])
                    else:
                        evaluation_eeg = eeg[target]
                    metrics, bundle = evaluate(model, evaluation_eeg, visual, text, hold, ids, cids, device, cfg['batch'])
                    torch.save(bundle, out / (run_name + '_holdout_scores.pt'))
                    row = dict(subject=target, seed=seed, arm=arm, setting=cfg.get('setting','within_subject'), role='locked_development_holdout',
                        candidates=len(hold), selected_epoch=best_epoch, parameters=sum(p.numel() for p in model.parameters()),
                        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()), **metrics)
                    records.append(row)
                    with (out / 'results.csv').open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(records)
        save_status('complete_development_only')
    except Exception as error:
        provenance['error'] = type(error).__name__ + ': ' + str(error)
        save_status('incomplete'); raise
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.manifest, args.config, args.output)
