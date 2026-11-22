"""Issue #5: one fixed, read-only CPU latent-image marginal feasibility gate.

The teacher is a retrieval-augmented diagnostic, NOT gallery-free inference.
Only optimizer-fit images mediate EEG queries; development caption vectors are
retrieval candidates, never fit targets. No GPU, optimizer, or official test.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.image_caption_bridge_gate import (
    BOOTSTRAP_SEED, CONTROL_SEED, RIDGE_RATIO, bootstrap_ci, caption_hits,
    fit_bridge, sha256)
from phase1.historical_06b import PARTS, TRAIN_IMAGES, _resolve

IMAGE_TEMPERATURE = 0.07
CAPTION_TEMPERATURE = 0.07
MIN_TEACHER_TOP1_PCT = 5.0
DEFAULT_ROOT = Path('/content/drive/MyDrive/eeg_results/v6b_checkpointed_phase1/sub01_seed0_routec')
DEFAULT_CACHE = Path('/content/drive/MyDrive/eeg_results/cache')


def validate_split(fit, query, n_rows=TRAIN_IMAGES):
    """Require the exact 9/1 per-concept historical split, including ordering."""
    all_rows = torch.arange(n_rows)
    if (n_rows != TRAIN_IMAGES or fit.device.type != 'cpu' or query.device.type != 'cpu'
            or fit.dtype != torch.long or query.dtype != torch.long
            or not torch.equal(fit, all_rows[all_rows % 10 != 9])
            or not torch.equal(query, all_rows[all_rows % 10 == 9])):
        raise ValueError('Expected exact historical 14886/1654 optimizer/development rows')


def _cpu_matrix(value, name):
    if (not torch.is_tensor(value) or value.device.type != 'cpu' or value.ndim != 2
            or min(value.shape) < 1 or not torch.isfinite(value).all()):
        raise ValueError(name + ' must be a finite, nonempty CPU matrix')


@torch.no_grad()
def mediated_scores(eeg_queries, train_images, caption_gallery, fitted_bridge,
                    *, query_batch_size=64, allowed_mediators=None):
    """Exact full training-image marginal; query chunks do not change support.

    pi(j|e)=softmax(cos(e,i_j)/.07), q(k|j)=softmax(cos(ridge(i_j),t_k)/.07).
    Every query sees every train mediator and every candidate caption. There is
    no minibatch approximation, top-k truncation, or development image input.
    """
    for name, value in (('EEG', eeg_queries), ('train images', train_images),
                        ('caption gallery', caption_gallery)):
        _cpu_matrix(value, name)
    if eeg_queries.shape[1] != train_images.shape[1]:
        raise ValueError('EEG/image embedding dimensions differ')
    if isinstance(query_batch_size, bool) or not isinstance(query_batch_size, int) or query_batch_size < 1:
        raise ValueError('query_batch_size must be a positive integer')
    weights, mx, my = fitted_bridge
    for name, value in zip(('ridge weights', 'image mean', 'caption mean'), fitted_bridge):
        _cpu_matrix(value, name)
    images = F.normalize(train_images.float(), dim=-1)
    mapped = F.normalize((images - mx) @ weights + my, dim=-1)
    conditional = ((mapped @ F.normalize(caption_gallery.float(), dim=-1).T)
                   / CAPTION_TEMPERATURE).softmax(-1)
    return _marginal_scores(eeg_queries, images, conditional, query_batch_size,
                            allowed_mediators)


def _marginal_scores(eeg_queries, normalized_images, conditional,
                     query_batch_size=64, allowed_mediators=None):
    if allowed_mediators is not None:
        if (allowed_mediators.device.type != 'cpu' or allowed_mediators.dtype != torch.bool
                or allowed_mediators.shape != (len(eeg_queries), len(normalized_images))
                or not allowed_mediators.any(-1).all()):
            raise ValueError('Every query needs at least one permitted training mediator')
    result = []
    for start in range(0, len(eeg_queries), query_batch_size):
        queries = eeg_queries[start:start+query_batch_size]
        logits = (F.normalize(queries.float(), dim=-1) @ normalized_images.T) / IMAGE_TEMPERATURE
        if allowed_mediators is not None:
            logits = logits.masked_fill(~allowed_mediators[start:start+query_batch_size], -torch.inf)
        posterior = logits.softmax(-1)
        result.append(posterior @ conditional)
    scores = torch.cat(result)
    if not torch.isfinite(scores).all() or not torch.allclose(
            scores.sum(-1), torch.ones(len(scores)), atol=2e-6, rtol=2e-6):
        raise ValueError('Marginal probability normalization failed')
    return scores


def gate_decision(real_hits, shuffled_hits):
    """Predeclared numerical screen; does not establish provenance or novelty."""
    if (real_hits.ndim != 1 or real_hits.shape != shuffled_hits.shape or not len(real_hits)
            or not torch.isin(real_hits, torch.tensor([0., 1.])).all()
            or not torch.isin(shuffled_hits, torch.tensor([0., 1.])).all()):
        raise ValueError('Paired finite binary hits are required')
    top1 = float(real_hits.double().mean() * 100)
    ci = bootstrap_ci(real_hits, shuffled_hits, BOOTSTRAP_SEED)
    passed = top1 >= MIN_TEACHER_TOP1_PCT and ci[0] > 0
    return {'teacher_top1_pct': top1,
            'shuffled_teacher_top1_pct': float(shuffled_hits.double().mean() * 100),
            'real_minus_shuffle_top1_pp': float((real_hits.double()-shuffled_hits).mean()*100),
            'paired_delta_ci95_pp': ci, 'bootstrap_replicates': 1000,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'numerical_gate': 'PASS' if passed else 'FAIL',
            'gpu_decision': 'NO_GO',
            'gpu_blockers': ['Independent EEG/image/caption stimulus-row identity is NOT_VERIFIED',
                             'No explicit GPU pilot execution authorization'],
            'criteria': 'teacher Top-1 >= 5% AND paired bootstrap lower bound > 0 pp'}


@torch.no_grad()
def evaluate(eeg_queries, train_images, text_all, caption_ids, fit, query):
    """Inputs structurally exclude development image vectors from the teacher."""
    from phase1.joint_data import derangement
    validate_split(fit, query, len(text_all))
    if (len(eeg_queries) != len(query) or len(train_images) != len(fit)
            or len(caption_ids) != len(text_all)):
        raise ValueError('Embedding/identity rows do not match the fixed split')
    _cpu_matrix(text_all, 'captions')
    _cpu_matrix(eeg_queries, 'EEG queries')
    _cpu_matrix(train_images, 'train images')
    # train_images is compact; fit_bridge is deliberately given no dev rows.
    train_text = text_all[fit]
    local_fit = torch.arange(len(fit))
    permutation = derangement((fit // 10).tolist(), CONTROL_SEED)
    maps = {'real': fit_bridge(train_images, train_text, local_fit),
            'shuffled': fit_bridge(train_images, train_text, local_fit, permutation)}
    identities = [caption_ids[i] for i in query.tolist()]
    hits, metrics, excluded_hits, excluded_metrics = {}, {}, {}, {}
    allowed = query[:, None] // 10 != fit[None, :] // 10
    normalized_images = F.normalize(train_images.float(), dim=-1)
    for arm, bridge in maps.items():
        weights, mx, my = bridge
        mapped = F.normalize((normalized_images-mx) @ weights+my, dim=-1)
        conditional = ((mapped @ F.normalize(text_all[query].float(), dim=-1).T)
                       / CAPTION_TEMPERATURE).softmax(-1)
        scores = _marginal_scores(eeg_queries, normalized_images, conditional)
        a1, a5 = caption_hits(scores, identities, identities, query.tolist())
        hits[arm] = a1
        metrics[arm] = {'hits_top1': int(a1.sum()), 'hits_top5': int(a5.sum()),
                        'top1_pct': float(a1.double().mean()*100),
                        'top5_pct': float(a5.double().mean()*100)}
        excluded = _marginal_scores(eeg_queries, normalized_images, conditional,
                                   allowed_mediators=allowed)
        e1, e5 = caption_hits(excluded, identities, identities, query.tolist())
        excluded_hits[arm] = e1
        excluded_metrics[arm] = {'hits_top1': int(e1.sum()), 'hits_top5': int(e5.sum()),
                                 'top1_pct': float(e1.double().mean()*100),
                                 'top5_pct': float(e5.double().mean()*100)}
    return {'metrics': metrics, 'decision': gate_decision(hits['real'], hits['shuffled']),
            'paired_top1_hits': {name: value.int().tolist() for name, value in hits.items()},
            'same_concept_excluded_sensitivity': {
                'metrics': excluded_metrics,
                'decision': gate_decision(excluded_hits['real'], excluded_hits['shuffled']),
                'paired_top1_hits': {name: value.int().tolist() for name, value in excluded_hits.items()},
                'interpretation': 'Same fitted maps/temperatures/caption gallery, but exclude all train mediators of the query concept; matches candidate training anti-shortcut mask'},
            'queries': len(query), 'caption_gallery': len(query), 'image_mediators': len(fit),
            'unique_caption_candidates': len(set(identities)),
            'control': 'cross-concept train-caption pairing in ridge fit; identical EEG queries and image mediators',
            'control_seed': CONTROL_SEED, 'mediator_rows': fit.tolist(),
            'query_and_caption_gallery_rows': query.tolist()}


def inspect_inputs(root, cache):
    """Metadata/existence-only check: never download or open tensor caches."""
    root, cache = Path(root).resolve(), Path(cache).resolve()
    paths = {'manifest': root/'train_only_manifest.json',
             'baseline_record': root/'baseline/run.json',
             'checkpoint': root/'baseline/within_route_c_sub-01_s0_best.pt',
             'caption_features': cache/'captions_train_only.npy',
             'caption_ids': cache/'caption_ids_train_only.json',
             'caption_provenance': cache/'caption_provenance_train_only.json'}
    if paths['manifest'].is_file():
        manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
        if (manifest.get('partition') != 'official_training'
                or manifest.get('official_test_opened') is not False
                or manifest.get('eeg_file') != 'sub-01_train_z16.npy'
                or manifest.get('feature_parts') != list(PARTS)):
            raise ValueError('Refusing inputs outside the fixed official-training cache')
        # Preserve the signed manifest's historical cache location, without rewriting it.
        historical_cache = Path(manifest.get('cache_root', paths['manifest'].parent)).resolve()
    else:
        historical_cache = cache
    for name in ('sub-01_train_z16.npy', *PARTS):
        paths['historical_cache/' + name] = _resolve(historical_cache, name)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    return paths, {'status': 'NOT_MEASURED', 'mode': 'input_existence_check',
                   'official_test_opened': False, 'gpu_decision': 'NO_GO',
                   'all_required_files_present': not missing, 'missing_inputs': missing,
                   'stimulus_row_identity': 'NOT_VERIFIED',
                   'reason': 'Private artifacts absent' if missing else 'Existence is not validation or a measured gate'}


def validate_baseline_record(record, manifest, checkpoint):
    """Bind checkpoint and loader to the recorded baseline, allowing only CRLF/LF differences."""
    from phase1.historical_06b import _code_hashes
    if (record.get('status') != 'complete_training_only'
            or record.get('official_test_opened') is not False
            or record.get('checkpoint') != checkpoint.name
            or record.get('checkpoint_sha256') != sha256(checkpoint)
            or record.get('signature', {}).get('manifest_sha256') != sha256(manifest)):
        raise ValueError('Historical baseline checkpoint/manifest provenance mismatch')
    actual = _code_hashes()
    expected = record['signature'].get('code_hashes')
    if not isinstance(expected, dict) or expected.keys() != actual.keys():
        raise ValueError('Historical baseline signed source hash set mismatch')
    source_root = Path(__file__).resolve().parents[1]/'phase1'
    for name, raw_hash in actual.items():
        # Baseline ran on Linux; a Windows git checkout may materialize CRLF.
        # No whitespace stripping or any other source transformation is allowed.
        lf_hash = hashlib.sha256((source_root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if expected[name] not in (raw_hash, lf_hash):
            raise ValueError('Historical baseline signed source changed: '+name)


def run(root=DEFAULT_ROOT, cache=DEFAULT_CACHE, *, check_inputs=False):
    from phase1.historical_06b import (
        FEATURE_WIDTH, POST17, ROUTE_INIT, ROUTE_LAYERS, _load_training, preflight)
    from phase1.models import Aligner
    from phase1.trimodal_06b import _load_text, preflight_text_inputs
    started = time.monotonic()
    paths, availability = inspect_inputs(root, cache)
    if check_inputs or availability['missing_inputs']:
        return availability
    manifest = paths['manifest']
    record = json.loads(paths['baseline_record'].read_text(encoding='utf-8'))
    validate_baseline_record(record, manifest, paths['checkpoint'])
    # Full existing preflight checks the exact protocol, channel order, sizes,
    # finiteness and source hashes before the legacy loader touches any arrays.
    preflight(manifest)
    caption_check = preflight_text_inputs(paths['caption_features'], paths['caption_ids'],
                                          paths['caption_provenance'])
    source_root = Path(__file__).resolve().parents[1]
    sources = ['diagnostics/lima_feasibility_gate.py', 'diagnostics/image_caption_bridge_gate.py',
               'phase1/historical_06b.py', 'phase1/models.py', 'phase1/core.py',
               'phase1/trimodal_06b.py', 'phase1/joint_data.py', 'phase1/joint_model.py',
               'phase1/experiment.py', 'phase1/joint_experiment.py']
    paths.update({'code/'+name: source_root/name for name in sources})
    before = {name: sha256(path) for name, path in paths.items()}
    _, eeg, layers, fit, query, _ = _load_training(manifest, torch.device('cpu'))
    validate_split(fit, query)
    if record.get('fit_rows') != fit.tolist() or record.get('validation_rows') != query.tolist():
        raise ValueError('Saved baseline split disagrees with diagnostic split')
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = Aligner(len(POST17)*eeg.shape[2], [FEATURE_WIDTH]*len(ROUTE_LAYERS), list(ROUTE_INIT))
        model.load_state_dict(torch.load(paths['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.eval()
        # Development image vectors are never encoded, passed to, or scored by the mediator.
        images = torch.cat([model.img([v[rows] for v in layers]).float() for rows in fit.split(256)])
        queries = torch.cat([model.eeg(eeg[rows].float()).float() for rows in query.split(256)])
    del model, layers, eeg
    captions, caption_ids = _load_text(paths['caption_features'], paths['caption_ids'], torch.device('cpu'))
    mean, std = captions[fit].mean(0, keepdim=True), captions[fit].std(0, keepdim=True)+1e-4
    text = (captions-mean)/std
    measured = evaluate(queries, images, text, caption_ids, fit, query)
    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError('Input artifact changed during read-only diagnostic')
    return {'status': 'MEASURED_CPU_ONLY', 'device': 'cpu', 'official_test_opened': False,
            'gpu_training_executed': False, 'scope': 'sub-01 seed-0; 1654-way training-partition development',
            'temperatures': {'image': IMAGE_TEMPERATURE, 'caption': CAPTION_TEMPERATURE},
            'fixed_ridge_ratio': RIDGE_RATIO, 'selected_baseline_epoch': record['selected_epoch'],
            'caption_preflight': {'status': caption_check['status'], 'unique_caption_identities': caption_check['unique_caption_identities']},
            'input_sha256_before': before, 'input_sha256_after': after, 'inputs_unchanged': True,
            'source_sha256': {name: sha256(source_root/name) for name in sources},
            'baseline_signed_source_verification': 'recorded raw hash or CRLF-to-LF-only hash; no other normalization',
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'cpu_threads': torch.get_num_threads()},
            'cpu_wall_seconds': time.monotonic()-started, **measured,
            'limitations': ['Teacher uses a training-image gallery: it is a retrieval-augmented proxy, not direct inference',
                            'No evidence yet that a gallery-free student can recover this signal',
                            'Historical image normalization includes development rows and is preserved',
                            'Independent EEG/image/caption stimulus-row identity remains NOT_VERIFIED',
                            'Caption provenance declarations/hashes are checked, not independently established',
                            'No official 200-way/10-subject score or novelty claim']}


def write_report(report, output, root, cache):
    """Exclusive new report only; cannot overwrite an existing research artifact."""
    output, root, cache = (Path(p).resolve() for p in (output, root, cache))
    if output.is_relative_to(root) or output.is_relative_to(cache):
        raise ValueError('Report must be outside existing run and cache directories')
    serialized = json.dumps(report, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        handle.write(serialized+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None:
        # Refuse reuse before any potentially expensive read-only measurement.
        if args.output.exists():
            raise FileExistsError(str(args.output))
        if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.cache)):
            raise ValueError('Report must be outside existing run and cache directories')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    report = run(args.root, args.cache, check_inputs=args.check_inputs)
    if args.output is not None:
        write_report(report, args.output, args.root, args.cache)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
