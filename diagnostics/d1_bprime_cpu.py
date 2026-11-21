"""Read-only, CPU-only D1' on saved epoch-5 B'/D' corrected checkpoints.
Never trains, writes an artifact, opens official test, or invokes CUDA.
"""
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Make phase1 importable when launched as a standalone script from Colab.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from phase1.historical_06b import _load_training, PER_CONCEPT
from phase1.trimodal_06b import _load_text, JOINT_WEIGHT, DISTILL_WEIGHT, READOUT_WEIGHT
from phase1.joint_data import derangement
from phase1.joint_model import JointAligner, Residualizer, symmetric_identity_nce


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def audit():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = Path(os.getenv('EEG_06C_RUN_ROOT',
        '/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec'))
    cache = Path(os.getenv('EEG_CACHE', '/content/drive/MyDrive/eeg_results/cache'))
    dest = root / 'trimodal_semantic_correction_bd_20261121'
    record = json.loads((dest / 'run.json').read_text())
    signature = record['signature']
    if signature.get('semantic_correction') is not True or signature.get('joint_alpha') != 20:
        raise ValueError('Not the alpha=20 corrected B-prime/D-prime run')
    if record.get('official_test_opened') is not False:
        raise ValueError('Official-test provenance not verified')
    prep_file = dest / 'preprocessing.pt'
    if file_hash(prep_file) != record.get('preprocessing_sha256'):
        raise ValueError('Preprocessing digest changed')
    prep = torch.load(prep_file, map_location='cpu', weights_only=True)
    dims = prep['model_dimensions']
    if dims.get('semantic_correction') is not True or dims.get('alpha') != 20:
        raise ValueError('Incorrect architecture in saved preprocessing')
    print("D1_PRIME: CPU only; reading existing train-only artifacts", flush=True)
    _, eeg, visual, fit, _, _ = _load_training(root / 'train_only_manifest.json',
                                               torch.device('cpu'))
    if not torch.equal(fit.cpu(), prep['fit_rows'].cpu()):
        raise ValueError('Training fit indices changed')
    fit = fit.cpu()
    rows = fit[:1024]
    raw_text, captions = _load_text(cache / 'captions_train_only.npy',
                    cache / 'caption_ids_train_only.json', torch.device('cpu'))
    text = (raw_text - prep['text_mean']) / prep['text_std']
    rz = Residualizer.from_state_dict(prep['residualizer'])
    if not torch.equal(rz.fit_indices.cpu(), fit):
        raise ValueError('Residualizer training rows changed')
    perm = derangement((fit // PER_CONCEPT).tolist(), 7000)
    other = fit[perm[:len(rows)]]
    with torch.no_grad():
        residuals = {
            'B': rz.transform([v[rows] for v in visual], text[rows]),
            'D': rz.transform([v[other] for v in visual], text[other]),
        }
    ids = [str(int(i)) for i in rows]
    cids = [captions[int(i)] for i in rows]
    for arm in ('B', 'D'):
        progress = json.loads((dest / f'sub-01_s0_{arm}_progress.json').read_text())
        ck_path = dest / f'sub-01_s0_{arm}_latest.pt'
        if (progress.get('latest_checkpoint') != ck_path.name or
                progress.get('latest_checkpoint_sha256') != file_hash(ck_path)):
            raise ValueError(f'{arm} saved checkpoint hash mismatch')
        saved = torch.load(ck_path, map_location='cpu', weights_only=True)
        if saved['epoch'] != 5 or saved['signature'] != signature:
            raise ValueError(f'{arm} checkpoint is not matching epoch 5')
        model = JointAligner(**dims).cpu()
        model.load_state_dict(saved['model'], strict=True)
        del saved
        gc.collect()
        model.eval()
        residual = residuals[arm]
        with torch.no_grad():
            z0 = model.backbone.eeg(eeg[rows].float())
            zi = model.backbone.img([v[rows] for v in visual])
        z = z0.detach().requires_grad_(True)
        zi = zi.detach()
        base = model.image_scores(z, zi)
        L_base = symmetric_identity_nce(base, ids)
        g_base = torch.autograd.grad(L_base, z, retain_graph=True)[0]
        teacher = model.joint_scores(z, zi, residual)
        H = (teacher - base.detach()).detach()
        L_teacher = symmetric_identity_nce(teacher, ids)
        # Corrected teacher uses base.detach(). Do NOT subtract grad of
        # NCE(base): only joint teacher gradient flows via triadic H.
        g_sem = torch.autograd.grad(JOINT_WEIGHT * L_teacher, z)[0]
        h_ratio = (H.std(1, unbiased=False) /
                   base.detach().std(1, unbiased=False).clamp_min(1e-10)).median().item()
        g_ratio = (g_sem.norm() / g_base.norm().clamp_min(1e-12)).item()
        flip = (teacher.detach().argmax(1) != base.detach().argmax(1)).float().mean().item()
        nce_delta = (L_teacher - L_base).item()
        if not all(math.isfinite(v) for v in (h_ratio, g_ratio, flip, nce_delta)):
            raise FloatingPointError(f'{arm} nonfinite D1-prime result')
        print(f"{arm}_prime epoch=5 H_to_B_std_median={h_ratio:.6f} "
              f"NCE_delta={nce_delta:.6f} G_sem_over_image={g_ratio:.6f} "
              f"argmax_flip_pct={100*flip:.3f}", flush=True)
        del z, z0, zi, base, teacher, H, g_sem, g_base, L_teacher, L_base
        # Read-only backward through one fit batch to determine if clipping
        # would activate; no optimizer or scheduler is instantiated.
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(20261121)
        ze = model.backbone.eeg(eeg[rows].float())
        vi = model.backbone.img([v[rows] for v in visual])
        terms = model.loss_terms(ze.float(), vi.float(), text[rows], residual, ids,
            joint_weight=JOINT_WEIGHT, distill_weight=DISTILL_WEIGHT, caption_ids=cids)
        loss = terms['total'] + READOUT_WEIGHT * symmetric_identity_nce(
            model.text_scores(ze.detach(), text[rows]), cids, cids)
        loss.backward()
        main_sq = sum(float(p.grad.detach().square().sum())
                      for n, p in model.named_parameters()
                      if p.grad is not None and not n.startswith(('readout_', 'text_log_scale')))
        preclip = math.sqrt(main_sq)
        print(f"{arm}_prime training_mode_preclip_main_grad_norm={preclip:.6f} "
              f"clip_active={preclip > 1.0}", flush=True)
        del model, ze, vi, terms, loss
        gc.collect()
    print('D1_PRIME_DONE: read-only; no optimizer update; no checkpoint modified')


if __name__ == '__main__':
    audit()
