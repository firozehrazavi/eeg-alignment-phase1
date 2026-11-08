"""Matched pooled/mean/patch experiment. No target training arrays in LOSO.

python -m phase1.patch_experiment --dataset DATASET.json --config CONFIG.json --output NEW_DIR
All settings are fixed before outer tests. Validation selects checkpoints only.
"""
import argparse
import copy
import csv
import json
import os
import platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core import (Standardizer, concept_split, clip_loss, multi_positive_loss,
                   retrieval_metrics, csls, AccessPolicy)
from .experiment import seed_all
from .patch_cache import TokenCache, digest, read_catalog
from .patch_adapter import PatchAligner


def schedule(config, subjects):
    if config['setting'] not in ('within_subject','loso'):
        raise ValueError('Expected within_subject or loso')
    if len(subjects) != len(set(subjects)):
        raise ValueError('Duplicate subjects')
    targets = config.get('targets', subjects)
    if not set(targets) <= set(subjects):
        raise ValueError('Unknown target subject')
    if config['setting']=='loso' and len(subjects)<2:
        raise ValueError('LOSO needs source subjects')
    return [(t, [t] if config['setting']=='within_subject' else [s for s in subjects if s!=t], seed)
            for t in targets for seed in config['seeds']]


def enforce_provenance(manifest, config):
    p = manifest['preprocessing']
    if config['regime'] == 'strict':
        if (p.get('regime') != 'strict' or p.get('upstream_target_independent') is not True
                or not p.get('audit_evidence')):
            raise ValueError('Strict evaluation requires audited upstream target-independent preprocessing')
    elif config['regime'] != 'benchmark_preprocessed':
        raise ValueError('Unknown preprocessing regime')
    if config['objective'] not in ('diagonal','multi_positive'):
        raise ValueError('Unknown objective')
    if config.get('hyperparameters','fixed_a_priori') != 'fixed_a_priori':
        raise ValueError('This runner accepts pre-fixed settings only; no outer-test tuning')


def fit_eeg_norm(arrays, rows):
    # Float64 streaming moments over source training images and time only.
    total = square = None
    count = 0
    for x in arrays:
        for start in range(0,len(rows),128):
            b = np.asarray(x[rows[start:start+128]],dtype=np.float64)
            if not np.isfinite(b).all():
                raise ValueError('Nonfinite source EEG')
            m, q = b.sum((0,2)), np.square(b).sum((0,2))
            total = m if total is None else total+m
            square = q if square is None else square+q
            count += b.shape[0]*b.shape[2]
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean**2,0)).clip(1e-6)
    return Standardizer(torch.tensor(mean[None,:,None],dtype=torch.float32),
                        torch.tensor(std[None,:,None],dtype=torch.float32))


def run(dataset_path, config_path, output, resume=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cfg = json.loads(Path(config_path).read_text())
    root = Path(dataset_path).resolve().parent
    manifest = json.loads(Path(dataset_path).read_text())
    enforce_provenance(manifest,cfg)
    catalog_path = root/manifest['catalog']
    cat = read_catalog(catalog_path)
    plan = schedule(cfg, sorted(manifest['subjects']))
    tokens = TokenCache(root/manifest['tokens'],catalog_path)
    if tokens.meta['layer'] != cfg['patch_layer']:
        raise ValueError('Configured layer does not match tokens')
    pooled = {s:[np.load(root/p,mmap_mode='r') for p in manifest['pooled'][s]] for s in ('train','test')}
    if len(pooled['train']) != 5 or any(x.shape[0]!=len(cat[s]) for s in pooled for x in pooled[s]):
        raise ValueError('Expected five aligned route_c layers [20,24,28,32,36]')
    if any(x.shape[1] != tokens.meta['width'] for s in pooled for x in pooled[s]):
        raise ValueError('Pooled and token widths disagree')
    # Catch wrong hook layer, ordering or preprocessing before costly training.
    # FP16 pooling differs slightly from mean-then-FP16 historical features.
    for s in ('train','test'):
        for i in np.linspace(0,len(cat[s])-1,min(8,len(cat[s])),dtype=int):
            a = tokens.get(s,[int(i)]).mean(1)[0].numpy()
            b = pooled[s][2][i]
            if np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-8) > .02:
                raise ValueError(f'Patch/pooled L28 mismatch at {s} row {i}; verify extraction provenance')
    out = Path(output)
    code_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    if out.exists() and not resume:
        raise FileExistsError('Choose a new output directory, or explicitly resume this experiment')
    previous = None
    if out.exists():
        previous = json.loads((out/'provenance.json').read_text())
        if (previous['config'] != cfg or previous['dataset_sha256'] != digest(dataset_path)
                or previous['catalog_sha256'] != digest(catalog_path) or previous['token_manifest'] != tokens.meta):
            raise ValueError('Resume provenance mismatch; choose a new output directory')
        if previous.get('code_hashes') != code_hashes:
            raise ValueError('Code changed since the interrupted run')
        if any(digest(root/p) != h for p,h in previous['input_hashes'].items()):
            raise ValueError('An input changed since the interrupted run')
    else:
        out.mkdir(parents=True)
    device = torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
    fit, val = concept_split([r['concept'] for r in cat['train']],cfg['split_seed'],cfg.get('val_fraction',.1))
    norms = [Standardizer.fit(torch.from_numpy(np.array(x[fit]))) for x in pooled['train']]
    visual_norm = norms[2]
    # Reuse exactly the pooled L28 affine transformation on every token.
    # Thus normalization and mean pooling commute for the mean/patch controls.
    provenance = dict(config=cfg,dataset=manifest,dataset_sha256=digest(dataset_path),code_hashes=code_hashes,
        catalog_sha256=digest(catalog_path),token_manifest=tokens.meta,
        split={'fit':fit,'validation':val},python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        input_hashes={p:digest(root/p) for s in manifest['pooled'].values() for p in s})
    if previous:
        provenance['input_hashes'].update(previous['input_hashes'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    ids = {s:[r['id'] for r in cat[s]] for s in cat}
    records=json.loads((out/'results.json').read_text()) if previous and (out/'results.json').exists() else []
    batch=cfg['batch']; chunk=cfg.get('adapter_chunk',8)
    if batch<2 or chunk<1 or cfg['epochs']<1 or len(fit)<2:
        raise ValueError('Invalid training budget')

    for target,sources,seed in plan:
        # Crucially: do not even open the held-out subject's TRAIN array in LOSO.
        arrays = [np.load(root/manifest['subjects'][s]['train'],mmap_mode='r') for s in sources]
        if any(x.ndim!=3 or len(x)!=len(cat['train']) or x.shape[1:]!=arrays[0].shape[1:] for x in arrays):
            raise ValueError('Source EEG shape/row order mismatch')
        en = fit_eeg_norm(arrays,fit)
        for s in sources:
            p = manifest['subjects'][s]['train']
            provenance['input_hashes'][p] = digest(root/p)
        n = len(cat['train'])
        train_rows = np.array([s*n+i for s in range(len(sources)) for i in fit])
        def eeg_batch(rows):
            x = np.stack([arrays[int(r)//n][int(r)%n] for r in rows])
            return en.transform(torch.from_numpy(x)).to(device)
        def visual(model,split,rows):
            if model.arm=='baseline':
                return model.visual(pooled=[norm.transform(torch.from_numpy(np.array(x[rows]))).to(device)
                                           for norm,x in zip(norms,pooled[split])])
            t = tokens.get(split,rows)
            t = (t-visual_norm.mean[:,None,:])/visual_norm.std[:,None,:]
            return model.visual(tokens=t.to(device),chunk=chunk)
        @torch.no_grad()
        def gallery(model,split,rows):
            return torch.cat([F.normalize(visual(model,split,rows[k:k+batch]),dim=-1)
                              for k in range(0,len(rows),batch)])
        @torch.no_grad()
        def validation(model):
            g = gallery(model,'train',val)
            vals=[]
            for s in range(len(sources)):
                q = torch.cat([F.normalize(model.eeg(eeg_batch([s*n+i for i in val[k:k+batch]])),dim=-1)
                               for k in range(0,len(val),batch)])
                vals.append(retrieval_metrics(q@g.T,[ids['train'][i] for i in val],
                                             [ids['train'][i] for i in val],len(val))['top1'])
            return float(np.mean(vals))
        for arm in cfg['arms']:
            completed=[r for r in records if r['subject']==target and r['seed']==seed and r['arm']==arm]
            if len(completed)==(2 if cfg.get('report_csls',False) else 1):
                continue
            seed_all(seed)
            model = PatchAligner(int(np.prod(arrays[0].shape[1:])),[x.shape[1] for x in pooled['train']],
                tokens.meta['width'],tokens.meta['grid'],arm,
                bottleneck=cfg.get('bottleneck',256),groups=cfg.get('groups',32)).to(device)
            # EEG initialization is identical across arms (created first).
            # Reset training RNG so architecture initialization cannot shift dropout/batches.
            seed_all(seed+10000)
            optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
            rng = np.random.default_rng(seed)
            best=-1.; history=[]; state=None
            for epoch in range(1,cfg['epochs']+1):
                model.train(); losses=[]
                for rows in np.array_split(rng.permutation(train_rows),int(np.ceil(len(train_rows)/batch))):
                    if len(rows)<2: continue
                    images=(rows%n).tolist()
                    z=model.eeg(eeg_batch(rows)); v=visual(model,'train',images)
                    identity=[ids['train'][i] for i in images]
                    loss=(clip_loss(z,v,model.scale()) if cfg['objective']=='diagonal' else
                          multi_positive_loss(z,v,model.scale(),identity,identity))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss; no accuracy result written')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('grad_clip',1.))
                    optimizer.step(); losses.append(float(loss.detach()))
                scheduler.step(); model.eval()
                score=validation(model)
                history.append(dict(epoch=epoch,loss=float(np.mean(losses)),source_val_top1=score))
                if score>best:
                    best=score; best_epoch=epoch
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                print(target,seed,arm,epoch,'source validation',score,flush=True)
            model.load_state_dict(state); model.eval()
            name=f'{cfg["setting"]}_{target}_s{seed}_{arm}_{cfg["objective"]}'
            torch.save(dict(model=state,eeg_norm=en,visual_norms=norms,config=cfg,
                            best_epoch=best_epoch,sources=sources,target=target),out/f'{name}.pt')
            # Only after checkpoint selection is final, open target test EEG.
            p=manifest['subjects'][target]['test']; provenance['input_hashes'][p]=digest(root/p)
            test=np.load(root/p,mmap_mode='r')
            if test.shape!=(200,*arrays[0].shape[1:]) or not np.isfinite(test).all():
                raise ValueError('Expected finite target EEG aligned with 200 test stimuli')
            with torch.no_grad():
                q=torch.cat([F.normalize(model.eeg(en.transform(torch.from_numpy(np.array(test[k:k+batch]))).to(device)),dim=-1)
                             for k in range(0,200,batch)])
                g=gallery(model,'test',list(range(200)))
                scores=q@g.T
                setting='within_subject' if cfg['setting']=='within_subject' else 'loso_inductive'
                label=setting if cfg['regime']=='strict' else setting.replace('loso_inductive','loso_zero_calibration')+'_benchmark_preprocessed'
                base=dict(subject=target,seed=seed,arm=arm,objective=cfg['objective'],
                          best_epoch=best_epoch,source_val_top1=best,regime=cfg['regime'])
                records.append(dict(**base,setting=label,**retrieval_metrics(scores,ids['test'],ids['test'])))
                if cfg.get('report_csls',False):
                    policy=AccessPolicy('loso_transductive',test_query_statistics=True)
                    records.append(dict(**base,setting=('loso_transductive_csls' if cfg['setting']=='loso' else 'within_transductive_csls')+'_'+cfg['regime'],
                                        **retrieval_metrics(csls(scores,policy),ids['test'],ids['test'])))
            np.save(out/f'{name}_scores.npy',scores.cpu().numpy())
            (out/f'{name}_history.json').write_text(json.dumps(history,indent=2))
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
            (out/'results.tmp').write_text(json.dumps(records,indent=2))
            os.replace(out/'results.tmp',out/'results.json')
            with open(out/'results.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            del model,optimizer,state
        del arrays
    provenance['peak_gpu_allocated_bytes']=max(torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                                              (previous or {}).get('peak_gpu_allocated_bytes',0))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (out/'COMPLETE').write_text('All configured folds, seeds and arms completed.\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true',help='Skip completed arms only if provenance matches')
    a=p.parse_args();run(a.dataset,a.config,a.output,a.resume)
