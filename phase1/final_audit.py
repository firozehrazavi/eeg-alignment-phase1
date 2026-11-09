"""Read-only training audit and separately approved frozen inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .patch_cache import digest, read_catalog, TokenCache
from .patch_adapter import PatchAligner
from .patch_experiment import fit_eeg_norm
from .core import Standardizer, concept_split, retrieval_metrics
from .patch_resources import BudgetMeter, retry_oom

ARMS=('baseline','mean','patch')

def selected(history):
    if [h['epoch'] for h in history]!=list(range(1,6)):
        raise ValueError('Expected exactly epochs 1..5 in checkpoint history')
    if not all(np.isfinite(h['loss']) and np.isfinite(h['source_val_top1']) for h in history):
        raise ValueError('Nonfinite training metrics')
    return max(history,key=lambda h:h['source_val_top1'])  # first epoch wins ties


def audit(training):
    root=Path(training)
    if not (root/'TRAINING_COMPLETE').is_file():raise ValueError('Training completion marker missing')
    p=json.loads((root/'provenance.json').read_text());cfg=p['config']
    if cfg['setting']!='within_subject' or cfg['seeds']!=[0] or cfg['epochs']!=5 or cfg['arms']!=list(ARMS):
        raise ValueError('Expected fixed sub-01 three-arm five-epoch pilot')
    if cfg.get('encoder','mlp')!='mlp' or cfg.get('report_csls',False):raise ValueError('Expected original MLP cosine-only pilot')
    fit=p['split']['fit'];val=p['split']['validation']
    if set(fit)&set(val) or not val:raise ValueError('Invalid validation split')
    result={'validation_candidates':len(val),'validation_top1_units':'percent',
            'chance_top1_percent':100/len(val),'arms':{},
            'checkpoint_retention':'Three latest checkpoints with five-epoch histories and embedded best_state; not fifteen separate epoch snapshots.'}
    for arm in ARMS:
        name=f'within_subject_sub-01_s0_{arm}_{cfg["objective"]}'
        path=root/f'{name}_latest.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        best=selected(ck['history'])
        if ck['epoch']!=5 or ck['best_epoch']!=best['epoch'] or ck['best']!=best['source_val_top1'] or not ck['best_state']:
            raise ValueError(f'Checkpoint selection/history mismatch: {arm}')
        result['arms'][arm]=dict(history=ck['history'],best_epoch=best['epoch'],
            best_validation_top1=best['source_val_top1'],checkpoint=str(path),sha256=digest(path))
    timing=root/'epoch_timing.jsonl'
    if timing.is_file():
        rows=[json.loads(line) for line in timing.read_text().splitlines() if line.strip()]
        pairs=[(r['run'],r['epoch']) for r in rows]
        if len(rows)!=15 or len(set(pairs))!=15:raise ValueError('Timing log has missing or duplicated epoch records')
        expected={(f'within_subject_sub-01_s0_{arm}_{cfg["objective"]}',epoch) for arm in ARMS for epoch in range(1,6)}
        if set(pairs)!=expected:raise ValueError('Timing epoch identities mismatch')
        result['timing_records_verified']=15
    else:result['timing_records_verified']=False
    result['verified_epoch_records']=sum(len(a['history']) for a in result['arms'].values())
    return result


def preflight(dataset,training):
    # Existence checks only: do not deserialize catalog/provenance or read test labels.
    root=Path(dataset).parent;training=Path(training)
    for path in [Path(dataset),root/'preparation_state.json',training/'provenance.json',training/'TRAINING_COMPLETE']:
        if not path.is_file():raise FileNotFoundError(path)
    paths=list(training.glob('*_latest.pt'))
    if len(paths)!=3:raise ValueError('Expected three latest checkpoint files')
    return {'status':'preapproval_existence_checks_only','checkpoint_bytes':sum(p.stat().st_size for p in paths),
            'test_labels_read':False,'next':'Approval required before semantic/leakage checks and inference'}


def metrics(scores,ids):
    result=retrieval_metrics(scores,ids,ids)
    order=sorted(range(len(ids)),key=lambda i:str(ids[i]));sorted_ids=[ids[i] for i in order]
    ranks=scores[:,order].argsort(dim=1,descending=True,stable=True)
    truth=torch.tensor([sorted_ids.index(i) for i in ids],device=scores.device)
    result['mrr']=(1/(ranks==truth[:,None]).nonzero()[:,1].double().add(1)).mean().item()
    return result


def evaluate(dataset,training,output,approve=False,device='cpu'):
    if not approve:raise PermissionError('Explicit final-test approval required before reading test identities')
    summary=audit(training);root=Path(dataset).resolve().parent;out=Path(output)
    p=json.loads((Path(training)/'provenance.json').read_text());cfg=p['config']
    if digest(dataset)!=p['dataset_sha256']:raise ValueError('Dataset changed since training')
    m=json.loads(Path(dataset).read_text());cat=read_catalog(root/m['catalog'])
    if digest(root/m['catalog'])!=p['catalog_sha256']:raise ValueError('Catalog changed')
    fit,val=concept_split([r['concept'] for r in cat['train']],cfg['split_seed'],cfg['val_fraction'])
    if fit!=p['split']['fit'] or val!=p['split']['validation']:raise ValueError('Validation split changed')
    if set(cat['train'][i]['concept'] for i in fit)&set(cat['train'][i]['concept'] for i in val):raise ValueError('Concept leakage')
    for name,h in p['input_hashes'].items():
        if digest(root/name)!=h:raise ValueError(f'Training input changed: {name}')
    tokens=TokenCache(root/m['tokens'],root/m['catalog'])
    if tokens.meta!=p['token_manifest']:raise ValueError('Token provenance changed')
    # Hash test payloads only after approval; bind idempotent output to exact data.
    identity={'evaluator_sha256':digest(__file__),'dataset_sha256':digest(dataset),'training_audit':summary,'device':device,
              'precision':'fp32','test_hashes':{name:digest(root/name) for name in
              [m['subjects']['sub-01']['test'],*m['pooled']['test'],str(Path(m['tokens'])/'test.npy')]}}
    if out.exists():
        if not (out/'COMPLETE').is_file():raise FileExistsError('Incomplete evaluation directory preserved; choose a new output path')
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Existing evaluation identity differs')
        return json.loads((out/'results.json').read_text())
    pooled={s:[np.load(root/name,mmap_mode='r') for name in m['pooled'][s]] for s in ('train','test')}
    train=np.load(root/m['subjects']['sub-01']['train'],mmap_mode='r')
    norm=fit_eeg_norm([train],fit)
    norms=[Standardizer.fit(torch.from_numpy(np.array(x[fit]))) for x in pooled['train']]
    test=np.load(root/m['subjects']['sub-01']['test'],mmap_mode='r')
    if test.shape!=(200,17,250) or not np.isfinite(test).all():raise ValueError('Invalid test EEG')
    dev=torch.device(device);meter=BudgetMeter();results={}
    out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps(identity,indent=2))
    with torch.inference_mode():
        for arm in ARMS:
            ck=torch.load(summary['arms'][arm]['checkpoint'],map_location='cpu',weights_only=False)
            model=PatchAligner(17*250,[x.shape[1] for x in pooled['train']],tokens.meta['width'],tokens.meta['grid'],arm,
                bottleneck=cfg['bottleneck'],groups=cfg['groups']).to(dev)
            model.load_state_dict(ck['best_state']);model.requires_grad_(False);model.eval()
            q=[];g=[]
            for start in range(0,200,8):
                meter.check(f'evaluation:{arm}:{start}',force=True);rows=list(range(start,min(start+8,200)))
                q.append(F.normalize(model.eeg(norm.transform(torch.from_numpy(np.array(test[rows]))).to(dev)).float(),dim=-1))
                if arm=='baseline':v=model.visual(pooled=[n.transform(torch.from_numpy(np.array(x[rows]))).to(dev) for n,x in zip(norms,pooled['test'])])
                else:
                    t=tokens.get('test',rows);t=(t-norms[2].mean[:,None,:])/norms[2].std[:,None,:]
                    v,_=retry_oom(lambda chunk:model.visual(tokens=t,chunk=chunk),1)
                g.append(F.normalize(v.float(),dim=-1))
            scores=torch.cat(q)@torch.cat(g).T
            results[arm]=dict(**metrics(scores,[r['id'] for r in cat['test']]),best_epoch=summary['arms'][arm]['best_epoch'],
                candidates=200,precision='fp32',regime='within_subject_benchmark_preprocessed')
            np.save(out/f'{arm}_scores.npy',scores.cpu().numpy())
            del model,ck,q,g,scores
            if dev.type=='cuda':torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(results,indent=2));(out/'COMPLETE').write_text('Frozen inference only')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--training',required=True);ap.add_argument('--dataset');ap.add_argument('--output')
    ap.add_argument('--audit',action='store_true');ap.add_argument('--approve-final-test',action='store_true');ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=ap.parse_args()
    value=audit(a.training) if a.audit else evaluate(a.dataset,a.training,a.output,True,a.device) if a.approve_final_test else preflight(a.dataset,a.training)
    print(json.dumps(value,indent=2))
