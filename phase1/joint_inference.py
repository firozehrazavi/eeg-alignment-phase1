"""Frozen-checkpoint retrieval scores; never fit or normalize using query statistics.

python -m phase1.joint_inference --checkpoint RUN_best.pt --eeg queries.npy \
    --gallery captions.npy --modality text --output NEW_scores.npz

Image galleries are NPZ files with layer20, layer24, layer28, layer32, layer36.
Text galleries are NPY matrices. EEG is a pre-z-score NPY rows/channels/time
array with the SAME upstream preprocessing and channel/time order as training.
This tool reports scores, not accuracy: no positional identity assumption is made.
The caller must separately authorize any access to a protected benchmark.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .joint_model import JointAligner

LAYERS = (20, 24, 28, 32, 36)


def _finite_float_array(value, ndim, label):
    if (not isinstance(value, np.ndarray) or value.ndim != ndim or not len(value)
            or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()):
        raise ValueError(f'{label} must be a nonempty finite floating-point array with {ndim} dimensions')
    return torch.from_numpy(np.array(value, dtype=np.float32))


def _npy(path, ndim, label):
    if Path(path).suffix.lower() != '.npy':
        raise ValueError(f'{label} requires a .npy file')
    return _finite_float_array(np.load(path, allow_pickle=False), ndim, label)


def _normalize(x, mean, std, label):
    mean, std = mean.cpu(), std.cpu()
    if (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all() or mean.shape != std.shape):
        raise ValueError(f'Invalid saved {label} normalization')
    if x.ndim == 3:
        expected = (1, x.shape[1], 1)
    else:
        expected = (1, x.shape[1])
    if tuple(mean.shape) != expected:
        raise ValueError(f'Saved {label} normalization shape mismatch')
    result = (x - mean) / std
    if not torch.isfinite(result).all():
        raise ValueError(f'Nonfinite normalized {label}')
    return result


@torch.no_grad()
def run(checkpoint, eeg, gallery, modality, output, device='cpu', batch=256):
    """Save a new NPZ containing scores, query_count, gallery_count, modality.

    Text inference intentionally never looks up visual_mean/visual_std, loads
    any image array, invokes the residualizer, or uses an image representation.
    """
    if modality not in ('image', 'text') or batch < 1:
        raise ValueError('Expected modality image|text and a positive batch')
    out = Path(output)
    if out.suffix.lower() != '.npz':
        raise ValueError('Output must be a new .npz file')
    if out.exists():
        raise FileExistsError('Inference output already exists')
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # Tensor-only loading prevents arbitrary pickle objects in a checkpoint.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = JointAligner(**saved['model_dimensions']).to(target).eval()
    model.load_state_dict(saved['model'])
    x = _npy(eeg, 3, 'EEG')
    if x[0].numel() != model.n_in:
        raise ValueError('EEG channel/time dimensions disagree with the checkpoint')
    x = _normalize(x, saved['eeg_mean'], saved['eeg_std'], 'EEG')
    ze = torch.cat([model.backbone.eeg(x[i:i + batch].to(target))
                    for i in range(0, len(x), batch)])
    if modality == 'text':
        # No image input or residual text is needed on this path.
        text = _npy(gallery, 2, 'Text gallery')
        if text.shape[1] != model.text_dim:
            raise ValueError('Text encoder width disagrees with the checkpoint')
        text = _normalize(text, saved['text_mean'], saved['text_std'], 'text')
        scores = torch.cat([model.text_scores(ze[i:i + batch], text.to(target)).cpu()
                            for i in range(0, len(ze), batch)])
        gallery_count = len(text)
    else:
        if Path(gallery).suffix.lower() != '.npz' or len(model.dims) != len(LAYERS):
            raise ValueError('Image inference expects the five-layer route_c NPZ gallery')
        with np.load(gallery, allow_pickle=False) as archive:
            keys = [f'layer{layer}' for layer in LAYERS]
            if set(archive.files) != set(keys):
                raise ValueError('Image gallery must contain exactly layer20/24/28/32/36')
            visual = [_finite_float_array(archive[key], 2, key) for key in keys]
        gallery_count = len(visual[0])
        if any(len(v) != gallery_count or v.shape[1] != width
               for v, width in zip(visual, model.dims)):
            raise ValueError('Image gallery layer shapes or row identities are not aligned')
        if len(saved['visual_mean']) != len(LAYERS) or len(saved['visual_std']) != len(LAYERS):
            raise ValueError('Saved image normalizers do not match route_c')
        visual = [_normalize(v, mean, std, f'layer{layer}') for v, mean, std, layer
                  in zip(visual, saved['visual_mean'], saved['visual_std'], LAYERS)]
        zi = torch.cat([model.backbone.img([v[i:i + batch].to(target) for v in visual])
                        for i in range(0, gallery_count, batch)])
        scores = torch.cat([model.image_scores(ze[i:i + batch], zi).cpu()
                            for i in range(0, len(ze), batch)])
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite retrieval scores')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards a race after the initial existence check.
    with out.open('xb') as handle:
        np.savez_compressed(handle, scores=scores.numpy(), query_count=len(ze),
                            gallery_count=gallery_count, modality=modality)
    return {'output': str(out.resolve()), 'query_count': len(ze),
            'gallery_count': gallery_count, 'modality': modality}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--eeg', required=True)
    parser.add_argument('--gallery', required=True)
    parser.add_argument('--modality', choices=('image', 'text'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    print(run(**vars(args)))
