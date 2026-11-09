"""Read-only CPU diagnostics restricted to training/validation artifacts.

Never call TokenCache/read_catalog: those helpers deserialize final-test metadata.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core import concept_split, retrieval_metrics, Standardizer, validate_catalog
from .patch_adapter import PatchAligner


def prefix_field(path, keys):
    """Read a requested JSON prefix field and stop immediately after its value.

    Only use on the repository's insertion-ordered JSON artifacts. Fail closed if
    a test/image_ids field precedes the requested field. No full json.load.
    """
    with open(path,encoding='utf-8') as f:
        def nonspace():
            ch=f.read(1)
            while ch and ch.isspace():ch=f.read(1)
            if not ch:raise ValueError('Unexpected end of JSON')
            return ch
        def value(first):
            chars=[first];depth=int(first in '[{');quoted=first=='"';escape=False
            if not depth and not quoted:raise ValueError('Expected object, array or string prefix')
            while depth or quoted:
                ch=f.read(1)
                if not ch:raise ValueError('Incomplete JSON prefix')
                chars.append(ch)
                if quoted:
                    if escape:escape=False
                    elif ch=='\\':escape=True
                    elif ch=='"':quoted=False
                elif ch=='"':quoted=True
                elif ch in '[{':depth+=1
                elif ch in ']}':depth-=1
            return ''.join(chars)
        def descend(wanted):
            if nonspace()!='{':raise ValueError('Expected JSON object')
            while True:
                if (ch:=nonspace())!='"':raise ValueError('Requested prefix key absent')
                key=json.loads(value(ch))
                if nonspace()!=':':raise ValueError('Invalid JSON')
                if key in {'test','te','image_ids'}:raise ValueError('Refusing to read a test identity field')
                if key==wanted[0]:
                    return descend(wanted[1:]) if len(wanted)>1 else json.loads(value(nonspace()))
                # Known earlier fields contain configuration/paths, never test identities.
                if key not in {'catalog','tokens','subjects','pooled','preprocessing','selection','source_hashes','config'}:
                    raise ValueError(f'Unapproved prefix field: {key}')
                value(nonspace())
                if nonspace()!=',':raise ValueError('Requested prefix key absent')
        return descend(list(keys))


def wilson(k,n,z=1.959963984540054):
    if not 0<=k<=n or n<1:raise ValueError('Invalid count')
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-h),100*(c+h)]


def describe_pool(records,fit,val):
    validate_catalog([r['id'] for r in records],[r['concept'] for r in records],[r['sha256'] for r in records])
    if sorted(fit+val)!=list(range(len(records))) or set(fit)&set(val):raise ValueError('Invalid partition')
    if {records[i]['concept'] for i in fit}&{records[i]['concept'] for i in val}:raise ValueError('Concept overlap')
    counts=Counter(records[i]['concept'] for i in val)
    return dict(fit_images=len(fit),fit_concepts=len({records[i]['concept'] for i in fit}),
        validation_images=len(val),validation_concepts=len(counts),images_per_validation_concept=sorted(counts.values()),
        exact_image_chance_percent=100/len(val),same_concept_top_image_chance_percent=100*sum(v*v for v in counts.values())/len(val)**2)


def norm_eeg(x,fit):
    total=np.zeros(x.shape[1]);square=total.copy();count=0
    for start in range(0,len(fit),64):
        b=np.asarray(x[fit[start:start+64]],dtype=np.float64)
        if not np.isfinite(b).all():raise ValueError('Nonfinite fit EEG')
        total+=b.sum((0,2));square+=(b*b).sum((0,2));count+=b.shape[0]*b.shape[2]
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean**2,0)).clip(1e-6)
    return Standardizer(torch.tensor(mean[None,:,None],dtype=torch.float32),torch.tensor(std[None,:,None],dtype=torch.float32))


def diagnose(prepared,training,validation_inference=False):
    root=Path(prepared);training=Path(training)
    cfg=prefix_field(training/'provenance.json',['config'])
    if cfg['setting']!='within_subject' or cfg['seeds']!=[0] or cfg.get('encoder','mlp')!='mlp':raise ValueError('Only fixed sub-01 pilot supported')
    records=prefix_field(root/'catalog.json',['train'])
    original_rows=prefix_field(root/'dataset.json',['selection','train'])
    if len(original_rows)!=len(records) or original_rows!=sorted(set(original_rows)) or min(original_rows)<0 or max(original_rows)>=16540:
        raise ValueError('Invalid original training row mapping')
    fit,val=concept_split([r['concept'] for r in records],seed=cfg['split_seed'],fraction=cfg['val_fraction'])
    report={'scope':'training/validation only; no optimizer or final-test access','pool':describe_pool(records,fit,val),
            'split_check':'Reconstructed from saved config; full provenance not loaded because it includes test identities',
            'mapping_first_rows':[dict(prepared_row=i,original_training_row=original_rows[i],image_id=records[i]['id'],concept=records[i]['concept']) for i in range(min(3,len(records)))],
            'original_EEG_identity_verified':False,'arms':{}}
    x=np.load(root/'sub-01_train_prez.npy',mmap_mode='r')
    if x.shape!=(len(records),17,250) or x.dtype!=np.float32:raise ValueError('Invalid prepared training EEG')
    for start in range(0,len(x),64):
        if not np.isfinite(x[start:start+64]).all():raise ValueError('Nonfinite training EEG')
    en=norm_eeg(x,fit)
    report['eeg_channel_std_fit']=en.std.flatten().tolist()
    report['eeg_normalized_validation_rms']=float(en.transform(torch.from_numpy(np.array(x[val]))).square().mean().sqrt())
    pooled=[np.load(root/f'train_L{layer}_mean.npy',mmap_mode='r') for layer in [20,24,28,32,36]]
    if any(t.shape!=(len(records),3200) or t.dtype!=np.float32 for t in pooled):raise ValueError('Invalid pooled shapes')
    norms=[Standardizer.fit(torch.from_numpy(np.array(t[fit]))) for t in pooled]
    report['visual_std_floor_counts']=[int((n.std<=1e-6).sum()) for n in norms]
    tokens=np.load(root/'tokens_L28/train.npy',mmap_mode='r');done=np.load(root/'tokens_L28/train_done.npy')
    if tokens.shape!=(len(records),1024,3200) or tokens.dtype!=np.float16 or done.dtype!=bool or done.shape!=(len(records),) or not done.all():
        raise ValueError('Incomplete/misaligned training tokens')
    errors=[];ratios=[]
    for i in np.linspace(0,len(records)-1,8,dtype=int):
        t=np.array(tokens[i],dtype=np.float32);p=np.array(pooled[2][i]);mean=t.mean(0)
        errors.append(float(np.linalg.norm(mean-p)/max(np.linalg.norm(p),1e-8)))
        zn=(t-norms[2].mean.numpy())/norms[2].std.numpy()
        ratios.append(float(np.sqrt(np.mean(zn**2))/max(np.sqrt(np.mean(zn.mean(0)**2)),1e-8)))
    report['sampled_token_mean_relative_errors']=errors
    report['sampled_token_to_mean_normalized_rms_ratio']=ratios
    if max(errors)>.02:raise ValueError('Training token/pooled means mismatch; do not relax tolerance')
    for arm in ['baseline','mean','patch']:
        ck=torch.load(training/f'within_subject_sub-01_s0_{arm}_{cfg["objective"]}_latest.pt',map_location='cpu',weights_only=False)
        h=ck['history']
        if ck['epoch']!=5 or [r['epoch'] for r in h]!=list(range(1,6)):raise ValueError('Expected five completed epochs')
        best=max(h,key=lambda a:a['source_val_top1'])
        if ck['best_epoch']!=best['epoch'] or ck['best']!=best['source_val_top1']:raise ValueError('Best checkpoint selection mismatch')
        scale=float(ck['best_state']['log_scale'].exp().clamp(max=100))
        result=dict(history=h,best_epoch=best['epoch'],saved_validation_top1_percent=best['source_val_top1'],
            learned_inverse_temperature=scale,best_log_scale=float(ck['best_state']['log_scale']),
            temperature_at_cap=scale>=100,naive_wilson95_percent=wilson(round(best['source_val_top1']*len(val)/100),len(val)),
            interval_warning='Descriptive only: ignores concept clustering and best-of-five selection')
        report['arms'][arm]=result
        if not validation_inference:continue
        model=PatchAligner(4250,[3200]*5,3200,[32,32],arm,bottleneck=cfg['bottleneck'],groups=cfg['groups'])
        model.load_state_dict(ck['best_state']);model.requires_grad_(False);model.eval()
        q=[];g=[]
        with torch.inference_mode():
            for start in range(0,len(val),4):
                rows=val[start:start+4]
                q.append(F.normalize(model.eeg(en.transform(torch.from_numpy(np.array(x[rows])))).float(),dim=-1))
                if arm=='baseline':v=model.visual(pooled=[n.transform(torch.from_numpy(np.array(t[rows]))) for n,t in zip(norms,pooled)])
                else:
                    t=torch.from_numpy(np.array(tokens[rows],dtype=np.float32));t=(t-norms[2].mean[:,None,:])/norms[2].std[:,None,:]
                    v=model.visual(tokens=t,chunk=1)
                g.append(F.normalize(v.float(),dim=-1))
            scores=torch.cat(q)@torch.cat(g).T
            ids=[records[i]['id'] for i in val];order=sorted(range(len(ids)),key=lambda i:str(ids[i]))
            winners=torch.tensor(order)[scores[:,order].argmax(1)].tolist()
            result['cpu_fp32_validation']=retrieval_metrics(scores,ids,ids,len(ids))
            result['same_concept_top_image_percent']=100*sum(records[val[i]]['concept']==records[val[j]]['concept'] for i,j in enumerate(winners))/len(val)
            result['correct_query_rows']=[val[i] for i,j in enumerate(winners) if i==j]
            result['predicted_gallery_rows']=[val[j] for j in winners]
            result['precision_note']='CPU FP32 re-score; original GPU autocast ranking can differ. Do not replace historical metric.'
        del model,ck,q,g
    if validation_inference:
        report['paired_validation_diagnostics']={}
        concepts=[records[i]['concept'] for i in val];clusters=sorted(set(concepts))
        for a,b in [('patch','mean'),('patch','baseline')]:
            ca=set(report['arms'][a]['correct_query_rows']);cb=set(report['arms'][b]['correct_query_rows'])
            differences=np.array([float(i in ca)-float(i in cb) for i in val])
            sums=np.array([sum(d for d,c in zip(differences,concepts) if c==g) for g in clusters])
            counts=np.array([concepts.count(g) for g in clusters])
            draws=np.random.default_rng(0).integers(0,len(clusters),(5000,len(clusters)))
            bootstrap=100*sums[draws].sum(1)/counts[draws].sum(1)
            report['paired_validation_diagnostics'][a+' minus '+b]=dict(
                difference_percentage_points=float(100*differences.mean()),
                a_only_correct=len(ca-cb),b_only_correct=len(cb-ca),
                concept_cluster_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                warning='Conditional on selected checkpoints/this subject; no best-epoch or seed-selection adjustment')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--training',required=True)
    p.add_argument('--validation-inference',action='store_true')
    a=p.parse_args();print(json.dumps(diagnose(a.prepared,a.training,a.validation_inference),indent=2))
