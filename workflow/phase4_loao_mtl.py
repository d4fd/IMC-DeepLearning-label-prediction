"""One-factor leave-one-aux-out ablation for retained frozen equal-weight MTL."""
import submission_paths as _submission_paths
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import phase4_equal_mtl as base

R = Path(str(_submission_paths.root()))
CONTRACT = R / 'phase4_aux_targets_v1.json'
BASELINE = R / 'phase4_equal_mtl_results.json'
CACHE = _submission_paths.tokens()
OUT = R / 'loao_outputs'
RESULT = R / 'phase4_loao_results.json'
SEEDS = (42, 43, 44)
THRESHOLD = 0.01


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_artifacts(directory, seed, expected_ids):
    history = json.loads((directory / f'equal_mtl_epoch_log_seed{seed}.json').read_text())
    prediction = pd.read_csv(directory / f'equal_mtl_validation_seed{seed}.csv')
    finite_history = all(
        np.isfinite(record['val_pr_auc']) and np.isfinite(record['val_roc_auc']) and
        all(value is None or np.isfinite(value) for values in (record['raw_losses'], record['effective_weighted_losses']) for value in values.values())
        for record in history
    )
    prediction_ok = (len(prediction) == len(expected_ids) and prediction['tissue_id'].nunique() == len(expected_ids) and
                     set(prediction['tissue_id']) == set(expected_ids) and np.isfinite(prediction['probability']).all())
    if not finite_history or not prediction_ok:
        raise ValueError(f'artifact audit failed for {directory.name}, seed {seed}')
    return {'epochs_logged': len(history), 'finite_history': finite_history,
            'prediction_count': len(prediction), 'unique_tissue_ids': int(prediction['tissue_id'].nunique()),
            'finite_predictions': bool(np.isfinite(prediction['probability']).all())}


def main():
    contract = json.loads(CONTRACT.read_text())
    assert len(contract['retained']) == 9
    assert contract['split_artifact_sha256'] == sha(R / 'splits/rcc2025necrosis_pid_split.csv')
    baseline = json.loads(BASELINE.read_text())
    all_auxiliary_mean = baseline['comparison']['equal_mtl_mean_pr_auc']
    train, validation = base.load('train', contract), base.load('validation', contract)
    if not len(train[0]) or not len(validation[0]):
        raise ValueError('Empty development partition')
    tokens = np.concatenate(train[0])
    mean, std = tokens.mean(0).astype(np.float32), (tokens.std(0) + 1e-6).astype(np.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('LOAO must run on the CUDA profile used for the Phase 4 comparator')
    ablations = {}
    for omitted in contract['retained']:
        variant = copy.deepcopy(contract)
        variant['retained'] = [name for name in contract['retained'] if name != omitted]
        directory = OUT / f'without_{omitted}'
        base.OUT = directory
        per_seed = []
        for seed in SEEDS:
            result = base.train_seed(seed, train, validation, mean, std, variant, device, multitask=True)
            result['artifact_audit'] = audit_artifacts(directory, seed, validation[2])
            result['omitted_auxiliary'] = omitted
            result['active_auxiliaries'] = variant['retained']
            per_seed.append(result)
            print('LOAO', omitted, seed, result['val_pr_auc'], result['best_epoch'], flush=True)
        mean_pr_auc = float(np.mean([result['val_pr_auc'] for result in per_seed]))
        delta = mean_pr_auc - all_auxiliary_mean
        decision = 'remove' if delta >= THRESHOLD else 'retain' if delta <= -THRESHOLD else 'inconclusive_retain'
        ablations[omitted] = {'per_seed': per_seed, 'mean_val_pr_auc': mean_pr_auc,
                              'delta_from_all_auxiliary': delta, 'decision': decision,
                              'finite_checks_passed': True}
    selected = [name for name in contract['retained'] if ablations[name]['decision'] != 'remove']
    report = {
        'protocol': {
            'study': 'one-factor LOAO; the omitted auxiliary head/loss is the sole change from retained equal-weight MTL',
            'cores_train': len(train[0]), 'cores_validation': len(validation[0]), 'seeds': list(SEEDS), 'omissions': contract['retained'],
            'architecture': 'mean tile/core pooling then Linear(512,128)+ReLU', 'normalization': 'train tiles only',
            'optimizer': 'Adam lr=1e-3', 'batch_cores': 32, 'max_epochs': 300, 'patience': 30,
            'losses': 'unweighted primary BCE plus unit-weight categorical CE / train-standardized Huber(delta=1)',
            'decision_threshold': THRESHOLD, 'inconclusive_rule': 'retain', 'test_used': False,
            'encoder_trained': False, 'device': device,
        },
        'baseline': {'all_auxiliary_mean_val_pr_auc': all_auxiliary_mean, 'results_sha256': sha(BASELINE)},
        'aux_manifest_sha256': sha(CONTRACT), 'tile_cache_metadata_sha256': sha(CACHE / 'cache_metadata.json'),
        'ablations': ablations, 'selected_auxiliaries': selected,
        'removed_auxiliaries': [name for name in contract['retained'] if name not in selected],
        'all_finite_checks_passed': True,
    }
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + chr(10))
    print('PHASE4_LOAO_OK', json.dumps({'selected_auxiliaries': selected, 'removed_auxiliaries': report['removed_auxiliaries']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
