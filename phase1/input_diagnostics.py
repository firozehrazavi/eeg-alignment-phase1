"""Read-only, bounded diagnostics. No download, array dump, or verification bypass."""
import gc
import json
import os
from pathlib import Path

import numpy as np


def eeg_summary(path, available_bytes):
    path = Path(path)
    out = {'path': str(path), 'exists': path.is_file()}
    if not path.is_file():
        return out
    out['file_bytes'] = path.stat().st_size
    # Pickled dictionaries cannot be memory-mapped. Load one file at a time.
    if available_bytes < 2.2 * out['file_bytes']:
        out['status'] = 'skipped: select CPU High-RAM; insufficient free RAM'
        return out
    obj = np.load(path, allow_pickle=True)  # Only the user's trusted EEG files.
    out['container'] = type(obj).__name__
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if not isinstance(obj, dict):
        out['status'] = 'unexpected container'
        return out
    eeg = obj.get('preprocessed_eeg_data')
    out['eeg_shape'] = list(eeg.shape) if isinstance(eeg, np.ndarray) else None
    out['eeg_dtype'] = str(eeg.dtype) if isinstance(eeg, np.ndarray) else None
    out['channel_count'] = len(obj['ch_names']) if 'ch_names' in obj else None
    times = np.asarray(obj.get('times', []))
    out['time_shape'] = list(times.shape)
    out['time_dtype'] = str(times.dtype)
    if times.ndim == 1 and times.size and times.dtype.kind in 'fiu':
        finite = bool(np.isfinite(times).all())
        out['time_finite'] = finite
        if finite:
            t = times.astype(np.float64)
            out['time_first_last'] = [float(t[0]), float(t[-1])]
            if len(t) > 1:
                dt = np.diff(t)
                out['interval_min_median_max'] = [float(dt.min()), float(np.median(dt)), float(dt.max())]
                out['strictly_increasing'] = bool((dt > 0).all())
            nearest = int(np.argmin(np.abs(t)))
            out['nearest_zero_index_value'] = [nearest, float(t[nearest])]
            if isinstance(eeg, np.ndarray) and eeg.ndim:
                out['time_count_minus_eeg_samples'] = int(len(t) - eeg.shape[-1])
    del obj, eeg, times
    gc.collect()
    return out


def image_summary(roots, max_dirs=4000, max_depth=4):
    """Inspect bounded directory entries only; never read image pixels."""
    reports = []
    for root in dict.fromkeys(map(str, roots)):
        base = Path(root)
        report = {'root': root, 'exists': base.is_dir(), 'metadata_paths': [],
                  'image_directories': [], 'image_files_seen': 0, 'scan_truncated': False}
        visited = 0
        def onerror(error):
            report.setdefault('unreadable_paths', []).append(str(error.filename))
        for directory, dirs, files in os.walk(base, followlinks=False, onerror=onerror):
            visited += 1
            if visited > max_dirs:
                report['scan_truncated'] = True
                break
            depth = len(Path(directory).relative_to(base).parts)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in
                             {'Preprocessed_data_250Hz', 'cache', 'patch_phase1'})
            if depth >= max_depth:
                if dirs: report['scan_truncated'] = True
                dirs[:] = []
            if 'image_metadata.npy' in files:
                report['metadata_paths'].append(str(Path(directory) / 'image_metadata.npy'))
            count = sum(Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} for f in files)
            report['image_files_seen'] += count
            if count and len(report['image_directories']) < 4:
                report['image_directories'].append({'path': directory, 'files': count})
        report['directories_scanned'] = min(visited, max_dirs)
        reports.append(report)
    return reports


def main():
    # Refuse an attached GPU before loading EEG. No accelerator allocation calls.
    import torch
    if torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > Hardware accelerator: None, then rerun.')
    from google.colab import drive
    import psutil
    drive.mount('/content/drive')
    root = Path('/content/drive/MyDrive/THINGS-EEG2')
    cache = Path('/content/drive/MyDrive/eeg_results/cache')
    # Add the exact earlier extraction directory here if it differs.
    image_roots = [root, Path('/content/drive/MyDrive/eeg_results'), Path('/content/EEG_Data')]
    for split in ('training', 'test'):
        path = root / 'Preprocessed_data_250Hz/sub-01' / f'preprocessed_eeg_{split}.npy'
        try:
            report = eeg_summary(path, psutil.virtual_memory().available)
        except Exception as error:
            report = {'path': str(path), 'error_type': type(error).__name__}
        print(json.dumps({'eeg': report}, allow_nan=False))
    print(json.dumps({'images': image_summary(image_roots)}, allow_nan=False))
    meta_path = cache / 'meta_v6.pt'
    if meta_path.is_file():
        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        for split in ('train', 'test'):
            rels = meta.get(f'{split}_rel', [])
            print(json.dumps({'cached_paths': split, 'count': len(rels),
                              'first_two': [str(p) for p in rels[:2]],
                              'first_two_exist_per_root': {
                                  str(r): [(r / str(p)).is_file() for p in rels[:2]] for r in image_roots}}))
        del meta
    else:
        print(json.dumps({'missing_cache_metadata': str(meta_path)}))
    print('STOP: diagnostic only. No validation bypass, download, preparation, or GPU work performed.')


if __name__ == '__main__':
    main()
