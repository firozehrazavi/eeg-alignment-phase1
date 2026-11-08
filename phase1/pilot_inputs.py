"""CPU-only inventory/download/verification for exactly two sub-01 EEG files.

No snapshot_download, image download, feature extraction or training is invoked.
Hugging Face's persistent local_dir cache resumes .incomplete transfers on Drive.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

REPO='LidongYang/EEG_Image_decode'
FILES={f'Preprocessed_data_250Hz/sub-01/preprocessed_eeg_{s}.npy':shape for s,shape in
       [('training',(16540,4,63,250)),('test',(200,80,63,250))]}
CHANNELS=['Fp1','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','F2','F4','F6','F8',
          'FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T7','C5','C3','C1',
          'Cz','C2','C4','C6','T8','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
          'TP8','TP10','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)


def inventory(root,cache,image_roots=()):
    """Read-only cache/image check, before ANY Hub metadata/data requests."""
    import torch
    root,cache=Path(root),Path(cache)
    report={'eeg':{p:(root/p).is_file() for p in FILES},'cache':{},'image_roots':[],
            'image_root':None,'missing':[],'ready_images_and_features':False}
    meta_path=cache/'meta_v6.pt';feature_path=cache/'feat6_internvit6b.pt'
    meta=None
    for p in [meta_path,feature_path]:
        report['cache'][str(p)]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None}
        if not p.is_file():report['missing'].append(str(p))
    if meta_path.is_file():
        meta=torch.load(meta_path,map_location='cpu',weights_only=False)
    # Cached paths are relative to the dataset parent, not images_set itself.
    roots=list(dict.fromkeys(str(Path(p).parent if Path(p).name=='images_set' else Path(p))
                            for p in [root,*image_roots]))
    for candidate in roots:
        directory=Path(candidate);mp=directory/'images_set/image_metadata.npy'
        item={'root':candidate,'metadata_path':str(mp),'metadata_exists':mp.is_file(),
              'missing_images':[], 'metadata_matches':False,'reasons':[], 'splits':{}}
        if not mp.is_file():item['reasons'].append('image_metadata_missing')
        if meta is None:item['reasons'].append('cache_metadata_missing')
        if mp.is_file() and meta is not None:
            raw=load_dictionary(mp)
            item['metadata_matches']=True
            for s in ['train','test']:
                names=list(raw.get(f'{s}_img_files',[]));cached=list(meta.get(f'{s}_files',[]))
                rels=list(meta.get(f'{s}_rel',[]));expected=16540 if s=='train' else 200
                aligned=len(names)==len(cached)==len(rels)==expected and names==cached
                item['metadata_matches'] &= aligned
                if not aligned:item['reasons'].append(f'{s}_metadata_count_or_order_mismatch')
                if len(set(map(str,rels)))!=len(rels):item['reasons'].append(f'{s}_duplicate_paths')
                checked=0;missing=0
                # Exhaustive identity-based checks, never bounded directory scans.
                for rel in rels:
                    path=directory/rel
                    checked+=1
                    if not path.is_file():
                        missing+=1
                        if len(item['missing_images'])<10:item['missing_images'].append(str(path))
                item['splits'][s]={'expected':expected,'metadata_count':len(names),
                                  'cached_count':len(cached),'paths_checked':checked,'missing_count':missing}
                if missing:item['reasons'].append(f'{s}_referenced_images_missing')
            if not item['reasons'] and report['image_root'] is None:
                report['image_root']=candidate
        report['image_roots'].append(item)
    if report['image_root'] is None:
        report['missing'].append(str(root/'images_set/image_metadata.npy')+' and/or its referenced original images (see image_roots)')
    valid_features=False
    if feature_path.is_file():
        try:features=torch.load(feature_path,map_location='cpu',weights_only=False,mmap=True)
        except RuntimeError as error:
            if 'mmap' not in str(error):raise
            features=torch.load(feature_path,map_location='cpu',weights_only=False)
        shapes={};valid_features=True
        for split,count in [('tr',16540),('te',200)]:
            for layer in [20,24,28,32,36]:
                x=features.get(split,{}).get((layer,'mean'))
                shape=list(x.shape) if x is not None else None;shapes[f'{split}_L{layer}']=shape
                valid_features &= shape==[count,3200]
        report['cache'][str(feature_path)].update(shapes=shapes,expected_shapes_verified=bool(valid_features))
        if not valid_features:report['missing'].append(str(feature_path)+' has missing or unexpected pooled feature shapes')
        del features;gc.collect()
    report['ready_images_and_features']=bool(report['image_root'] and valid_features)
    return report


def verify_bytes(path,expected):
    path=Path(path)
    if path.stat().st_size!=expected['size']:
        raise ValueError(f'Size mismatch, preserved without overwrite: {path}')
    actual=sha256(path)
    if expected.get('sha256') and actual!=expected['sha256']:
        raise ValueError(f'SHA256 mismatch, preserved without overwrite: {path}')
    return dict(bytes=path.stat().st_size,sha256=actual,remote_sha256_verified=bool(expected.get('sha256')))


def load_dictionary(path):
    """Accept trusted direct-pickle dictionaries and scalar NPY object wrappers."""
    obj=np.load(path,allow_pickle=True)
    if isinstance(obj,np.ndarray) and obj.shape==() and obj.dtype==object:
        obj=obj.item()
    if not isinstance(obj,dict):
        raise ValueError(f'Expected a dictionary or scalar object array containing a dictionary: {path}')
    return obj


def validate_eeg_times(values,n_samples):
    """Recognize aligned metadata or the documented 50-sample EEG-only crop.

    EEG_Image_decode/EEG-preprocessing/preprocessing_utils.py epoching saves
    epochs.times unchanged, but returns sorted_data[..., 50:]. Never crop EEG again.
    """
    times=np.asarray(values)
    if (times.ndim!=1 or times.size==0 or times.dtype.kind not in 'fiu'
            or not np.isfinite(times).all()
            or not np.allclose(np.diff(times),1/250,atol=1e-6,rtol=0)):
        raise ValueError('Invalid time metadata: expected finite 1-D values spaced at 0.004 seconds')
    offset=0;convention='sample_aligned'
    if len(times)!=n_samples:
        if (n_samples==250 and len(times)==300
                and np.allclose(times,np.arange(300)/250-.2,atol=1e-6,rtol=0)):
            offset=50;convention='legacy_uncropped_epoch_metadata'
        else:
            raise ValueError(f'Time metadata length/range mismatch: {len(times)} values for {n_samples} EEG samples; '
                             'only the documented 300-point -0.2..0.996 axis may accompany 250 samples')
    effective=times[offset:]
    return dict(time_metadata_verified=True,time_convention=convention,
                time_metadata_samples=len(times),time_metadata_start_seconds=float(times[0]),
                time_metadata_end_seconds=float(times[-1]),time_metadata_offset_samples=offset,
                time_start_seconds=float(effective[0]),time_end_seconds=float(effective[-1]),
                time_samples=len(effective),sampling_rate_hz=250)


def inspect_eeg(path,shape,check_ram=True):
    """These trusted NPY dictionaries are pickled; load one at a time, never GPU."""
    if check_ram:
        import psutil
        required=int(Path(path).stat().st_size*2.2)
        if psutil.virtual_memory().available<required:
            raise MemoryError(f'Use a CPU High-RAM runtime: about {required/2**30:.1f} GiB free RAM required to verify {path}')
    data=load_dictionary(path);x=data['preprocessed_eeg_data']
    if tuple(x.shape)!=tuple(shape) or x.dtype.kind!='f':
        raise ValueError(f'Unexpected EEG shape/dtype at {path}: {x.shape}, {x.dtype}; expected {shape}')
    for start in range(0,len(x),8):
        if not np.isfinite(x[start:start+8]).all():raise ValueError(f'Nonfinite EEG in {path}')
    result=dict(shape=list(x.shape),dtype=str(x.dtype),finite=True,keys=sorted(data),
                channels_metadata_verified=False,time_metadata_verified=False)
    if 'ch_names' in data:
        names=list(map(str,data['ch_names']))
        if names!=CHANNELS:raise ValueError('Channel order differs from the existing posterior17 convention')
        result.update(channel_names=names,channels_metadata_verified=True)
    if 'times' in data:
        result.update(validate_eeg_times(data['times'],shape[-1]))
    del x,data;gc.collect()
    return result


def fetch_files(root,plan,download,inspect=inspect_eeg,allow_download=True):
    """Injected downloader supports offline tests; only the fixed sub-01 allowlist."""
    root=Path(root)
    if set(plan['files'])!=set(FILES):raise ValueError('Only sub-01 training and test are authorized')
    reports={}
    staging=root/'.pilot_downloads'/plan['revision']
    for name,expected in plan['files'].items():
        target=root/name
        if target.exists():
            details=verify_bytes(target,expected)
        else:
            if not allow_download:
                raise FileNotFoundError(f'Existing-only verification: missing {target}; no download attempted')
            staging.mkdir(parents=True,exist_ok=True)
            present=(staging/name).stat().st_size if (staging/name).is_file() else max(
                [p.stat().st_size for p in staging.rglob(target.name+'*.incomplete')],default=0)
            needed=max(256*1024*1024,(expected['size']-present)*1.05)
            if shutil.disk_usage(staging).free<needed:
                raise OSError('Insufficient Drive space for this EEG file')
            source=Path(download(repo_id=REPO,repo_type='dataset',filename=name,revision=plan['revision'],
                                 local_dir=str(staging),force_download=False))
            if not source.resolve().is_relative_to(staging.resolve()):raise ValueError('Unexpected staging path')
            details=verify_bytes(source,expected)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError(f'Will not overwrite {target}')
            # Same-Drive rename after byte verification. Caller holds the operation lock.
            source.rename(target)
        details.update(inspect(target,FILES[name]))
        reports[name]=details
        print(json.dumps({'verified_file':name,**details}),flush=True)
    return reports


def prepare_inputs(root,cache,image_roots=(),download=False,verify_existing=False):
    # CPU only, including metadata torch.load below. This never requests a GPU.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(root/'.pilot_inputs.lock')):
        report=inventory(root,cache,image_roots)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(time.time_ns())
        output=root/'pilot_input_reports'/f'{stamp}.json'
        write_json(output,{'inventory':report,'status':'inventory_only'})
        print(json.dumps({'inventory_report':str(output),'images_root':report['image_root'],
                          'ready_images_and_features':report['ready_images_and_features'],'missing':report['missing'],
                          'image_roots':report['image_roots']},indent=2))
        if not download and not verify_existing:return output
        from huggingface_hub import HfApi,hf_hub_download
        plan_path=root/'pilot_sub01_download_plan.json'
        if plan_path.exists():plan=json.loads(plan_path.read_text())
        elif verify_existing:
            raise FileNotFoundError('Existing-only verification needs the saved pilot_sub01_download_plan.json; no download attempted')
        else:
            info=HfApi().dataset_info(REPO,files_metadata=True)
            available={f.rfilename:f for f in info.siblings}
            plan={'repo_id':REPO,'revision':info.sha,'files':{}}
            for name in FILES:
                f=available[name];lfs=f.lfs
                sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
                plan['files'][name]={'size':f.size,'sha256':sha}
            write_json(plan_path,plan)
        if plan.get('repo_id')!=REPO or len(plan['revision'])!=40:raise ValueError('Invalid pinned download plan')
        provenance={'repo_id':REPO,'revision':plan['revision'],
            'dataset_card':f'https://huggingface.co/datasets/{REPO}/blob/{plan["revision"]}/README.md',
            'regime':'benchmark_preprocessed','upstream_target_independent':False,
            'description':'Distributed THINGS-EEG2 250 Hz mirror; no additional normalization during download.',
            'upstream_whitening_and_split_fit_scope':'not established by mirror metadata; strict LOSO remains blocked',
            'original_source':'https://osf.io/3jk45/','declared_license':'CC-BY-4.0'}
        state={'inventory':report,'provenance':provenance,'status':'verifying','ready_for_preparation':False}
        write_json(output,state)
        try:
            state['verified_eeg']=fetch_files(root,plan,hf_hub_download,allow_download=not verify_existing)
            state.update(status='eeg_verified',ready_for_preparation=report['ready_images_and_features'])
        except Exception as error:
            state.update(status='incomplete',error=str(error));write_json(output,state);raise
        write_json(output,state)
        # This pointer is informational; previous timestamped reports are preserved.
        write_json(root/'pilot_inputs_latest.json',{'report':str(output)})
        print('CPU input verification complete. No GPU work or other-subject download was started.')
        return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/content/drive/MyDrive/THINGS-EEG2')
    p.add_argument('--cache',default='/content/drive/MyDrive/eeg_results/cache')
    p.add_argument('--image-roots',nargs='*',default=['/content/EEG_Data','/content/drive/MyDrive/eeg_results'])
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--download-sub01',action='store_true',help='Download only the two authorized pilot files')
    mode.add_argument('--verify-existing-sub01',action='store_true',help='Verify existing files only; never download missing files')
    a=p.parse_args();prepare_inputs(a.root,a.cache,a.image_roots,a.download_sub01,a.verify_existing_sub01)
