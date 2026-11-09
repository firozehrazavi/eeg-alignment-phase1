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
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core import (Standardizer, concept_split, clip_loss, multi_positive_loss,
                   retrieval_metrics, csls, AccessPolicy)
from .experiment import seed_all
from .patch_cache import TokenCache, digest, read_catalog
from .patch_adapter import PatchAligner
from .patch_resources import BudgetMeter, runtime_profile, retry_oom
from .v7_controls import (ConvEEG, source_training_rows, stratified_rows, augment_eeg,
                          margin_multi_positive_loss, margin_at_epoch)


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
    if config.get('text_weight',0)!=0:
        raise ValueError('Text supervision is a separate experiment, never automatically enabled')
    if config.get('encoder','mlp') not in ('mlp','conv'):
        raise ValueError('Unknown EEG encoder')
    if config.get('sampler','random') not in ('random','subject_stratified'):
        raise ValueError('Unknown row sampler')
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


def run(dataset_path, config_path, output, resume=False, profile_one_epoch=False, train_only=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cfg = json.loads(Path(config_path).read_text())
    margin_at_epoch(cfg,1)  # Validate explicit stage settings before touching data.
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
    profile=runtime_profile()
    precision=cfg.get('precision','auto')
    if precision=='auto':precision=profile['precision'] if device.type=='cuda' else 'fp32'
    if precision not in ('fp32','fp16','bf16'):
        raise ValueError('Unknown mixed precision mode')
    amp_enabled=device.type=='cuda' and precision!='fp32'
    amp_dtype=torch.bfloat16 if precision=='bf16' else torch.float16
    meter=BudgetMeter();meter.check('training_start',force=True)
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
        split={'fit':fit,'validation':val},image_ids={s:[r['id'] for r in cat[s]] for s in cat},python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,precision=precision,runtime=profile,
        input_hashes={p:digest(root/p) for s in manifest['pooled'].values() for p in s})
    if previous:
        if previous.get('precision') != precision:
            raise ValueError('Runtime precision changed: choose a new matched run instead of resuming')
        provenance['input_hashes'].update(previous['input_hashes'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    ids = {s:[r['id'] for r in cat[s]] for s in cat}
    records=json.loads((out/'results.json').read_text()) if previous and (out/'results.json').exists() else []
    batch=cfg['batch']; chunk=min(cfg.get('adapter_chunk',8),profile['adapter_chunk'])
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
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        n = len(cat['train'])
        train_rows = source_training_rows(fit,val,n,sources,target if cfg['setting']=='loso' else None)
        def eeg_batch(rows):
            x = np.stack([arrays[int(r)//n][int(r)%n] for r in rows])
            return en.transform(torch.from_numpy(x)).to(device)
        def encode_eeg(model,x):
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                return model.eeg(x)
        def visual(model,split,rows,use_chunk):
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                if model.arm=='baseline':
                    return model.visual(pooled=[norm.transform(torch.from_numpy(np.array(x[rows]))).to(device)
                                               for norm,x in zip(norms,pooled[split])])
                t = tokens.get(split,rows)
                t = (t-visual_norm.mean[:,None,:])/visual_norm.std[:,None,:]
                return model.visual(tokens=t,chunk=use_chunk)
        @torch.no_grad()
        def gallery(model,split,rows):
            nonlocal chunk
            outputs=[]
            for k in range(0,len(rows),batch):
                value,chunk=retry_oom(lambda c:visual(model,split,rows[k:k+batch],c),chunk)
                outputs.append(F.normalize(value.float(),dim=-1))
            return torch.cat(outputs)
        @torch.no_grad()
        def validation(model,return_bundles=False):
            g = gallery(model,'train',val)
            vals=[];bundles={}
            for s in range(len(sources)):
                q = torch.cat([F.normalize(encode_eeg(model,eeg_batch([s*n+i for i in val[k:k+batch]])).float(),dim=-1)
                               for k in range(0,len(val),batch)])
                identity=[ids['train'][i] for i in val];scores=q@g.T
                vals.append(retrieval_metrics(scores,identity,identity,len(val))['top1'])
                if return_bundles:
                    bundles[sources[s]]=dict(role='source_validation',score_type='cosine',
                                             query_ids=identity,gallery_ids=identity,scores=scores.cpu())
            score=float(np.mean(vals))
            return (score,bundles) if return_bundles else score
        for arm in cfg['arms']:
            completed=[r for r in records if r['subject']==target and r['seed']==seed and r['arm']==arm]
            if len(completed)==(2 if cfg.get('report_csls',False) else 1):
                continue
            seed_all(seed)
            model = PatchAligner(int(np.prod(arrays[0].shape[1:])),[x.shape[1] for x in pooled['train']],
                tokens.meta['width'],tokens.meta['grid'],arm,
                bottleneck=cfg.get('bottleneck',256),groups=cfg.get('groups',32)).to(device)
            if cfg.get('encoder','mlp')=='conv':
                # Preserve image-head initialization; initialize EEG independently of visual arm.
                torch.manual_seed(seed)
                model.eeg=ConvEEG(*arrays[0].shape[1:]).to(device)
            # EEG initialization is identical across arms (created first).
            # Reset training RNG so architecture initialization cannot shift dropout/batches.
            seed_all(seed+10000)
            optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
            scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled and precision=='fp16')
            rng = np.random.default_rng(seed)
            best=-1.; history=[]; state=None
            name=f'{cfg["setting"]}_{target}_s{seed}_{arm}_{cfg["objective"]}'
            latest=out/f'{name}_latest.pt'
            first_epoch=1
            if resume and latest.exists():
                saved=torch.load(latest,map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
                rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
                if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
                first_epoch=saved['epoch']+1;best=saved['best'];best_epoch=saved['best_epoch']
                state=saved['best_state'];history=saved['history'];chunk=min(chunk,saved['chunk'])
            last_epoch=1 if profile_one_epoch else cfg['epochs']
            for epoch in range(first_epoch,last_epoch+1):
                epoch_start=time.time()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                model.train(); losses=[]
                batches=(stratified_rows(train_rows,np.repeat(np.arange(len(sources)),n),batch,rng)
                         if cfg.get('sampler','random')=='subject_stratified' else
                         np.array_split(rng.permutation(train_rows),int(np.ceil(len(train_rows)/batch))))
                for rows in batches:
                    if len(rows)<2: continue
                    meter.check(f'{name}:epoch{epoch}')
                    images=(rows%n).tolist()
                    identity=[ids['train'][i] for i in images]
                    cpu_rng=torch.get_rng_state()
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
                    def step(use_chunk):
                        optimizer.zero_grad(set_to_none=True)
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                        x=augment_eeg(eeg_batch(rows),**cfg.get('augmentation',{}))
                        z=encode_eeg(model,x);v=visual(model,'train',images,use_chunk)
                        margin=margin_at_epoch(cfg,epoch)
                        loss=(margin_multi_positive_loss(z,v,model.scale(),identity,identity,margin) if margin else
                              clip_loss(z.float(),v.float(),model.scale()) if cfg['objective']=='diagonal' else
                              multi_positive_loss(z,v,model.scale(),identity,identity))
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                        scaler.scale(loss).backward()
                        return float(loss.detach())
                    loss,chunk=retry_oom(step,chunk)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('grad_clip',1.))
                    scaler.step(optimizer);scaler.update();losses.append(loss)
                scheduler.step(); model.eval()
                score=validation(model)
                epoch_seconds=time.time()-epoch_start
                epoch_peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0
                history.append(dict(epoch=epoch,loss=float(np.mean(losses)),source_val_top1=score,
                                    seconds=epoch_seconds,peak_gpu_allocated_bytes=epoch_peak))
                if score>best:
                    best=score; best_epoch=epoch
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],epoch=epoch,
                    best=best,best_epoch=best_epoch,best_state=state,history=history,chunk=chunk)
                torch.save(saved,out/f'{name}_latest.tmp');os.replace(out/f'{name}_latest.tmp',latest)
                with open(out/'epoch_timing.jsonl','a') as f:
                    f.write(json.dumps(dict(run=name,epoch=epoch,seconds=epoch_seconds,chunk=chunk,loss=float(np.mean(losses)),source_val_top1=score,peak_gpu_allocated_bytes=epoch_peak))+'\n')
                print(json.dumps(dict(run=name,**history[-1])),flush=True)
                meter.check(f'{name}:epoch{epoch}:saved',force=True)
            if profile_one_epoch or train_only:
                del model,optimizer,state,saved
                continue
            model.load_state_dict(state); model.eval()
            if cfg.get('save_validation_scores',False):
                _,bundles=validation(model,return_bundles=True)
                torch.save(bundles,out/f'{name}_source_validation.pt')
            torch.save(dict(model=state,eeg_norm=en,visual_norms=norms,config=cfg,
                            best_epoch=best_epoch,sources=sources,target=target),out/f'{name}.pt')
            # Only after checkpoint selection is final, open target test EEG.
            p=manifest['subjects'][target]['test']; provenance['input_hashes'][p]=digest(root/p)
            test=np.load(root/p,mmap_mode='r')
            if test.shape!=(200,*arrays[0].shape[1:]) or not np.isfinite(test).all():
                raise ValueError('Expected finite target EEG aligned with 200 test stimuli')
            with torch.no_grad():
                q=torch.cat([F.normalize(encode_eeg(model,en.transform(torch.from_numpy(np.array(test[k:k+batch]))).to(device)).float(),dim=-1)
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
            if cfg.get('save_validation_scores',False):
                torch.save(dict(role='test',score_type='cosine',query_ids=ids['test'],gallery_ids=ids['test'],
                                scores=scores.cpu()),out/f'{name}_test_bundle.pt')
            (out/f'{name}_history.json').write_text(json.dumps(history,indent=2))
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
            (out/'results.tmp').write_text(json.dumps(records,indent=2))
            os.replace(out/'results.tmp',out/'results.json')
            with open(out/'results.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            del model,optimizer,state,saved
        del arrays
    provenance['peak_gpu_allocated_bytes']=max(torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                                              (previous or {}).get('peak_gpu_allocated_bytes',0))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    marker='PILOT_PROFILE_COMPLETE' if profile_one_epoch else 'TRAINING_COMPLETE' if train_only else 'COMPLETE'
    (out/marker).write_text('Training only, no test evaluation.\n' if (profile_one_epoch or train_only) else
                          'All configured folds, seeds and arms completed.\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true',help='Skip completed arms only if provenance matches')
    p.add_argument('--profile-one-epoch',action='store_true',help='Checkpoint one epoch per arm without test evaluation')
    p.add_argument('--train-only',action='store_true',help='Complete training checkpoints without test evaluation')
    a=p.parse_args();run(a.dataset,a.config,a.output,a.resume,a.profile_one_epoch,a.train_only)
