import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'workflow')]

from demo import run_demo, synthetic_contract
import phase5_finetune_mtl as p5
from phase6_train_final import fit_epoch
from phase6_final08_evaluate import do_evaluate, metrics, clustered_intervals
import phase6_final08_evaluate as evaluation
from submission_features import load_cached_features, tile_features
from virtues_t4_attention import sdpa_varlen_qkvpacked


def test_demo_uses_exported_training_path():
    assert run_demo() == {'cores': 6, 'normalizer_fit_tiles': 10, 'auxiliary_heads': 9, 'device': 'cpu'}


def test_normalizer_only_fits_selected_cores():
    features = [np.zeros((2, 512), np.float32), np.full((1, 512), 2, np.float32),
                np.full((4, 512), 1e6, np.float32)]
    mean, std, count = p5.fit_normalizer(features, [0, 1])
    np.testing.assert_allclose(mean, 2 / 3)
    assert count == 3 and np.isfinite(std).all() and (std > 0).all()


def test_primary_loss_survives_missing_auxiliaries():
    torch.manual_seed(13)
    contract = synthetic_contract()
    model = p5.CoreMTL(contract, 0.0)
    values = torch.randn(4, 2, 512)
    mask = torch.ones(4, 2, dtype=torch.bool)
    y = torch.tensor([0., 1., 0., 1.])
    aux = {name: (torch.full((4,), float('nan')) if spec['type'] == 'continuous'
                  else torch.full((4,), -1, dtype=torch.long))
           for name, spec in contract['targets'].items()}
    expected = F.binary_cross_entropy_with_logits(model(values, mask)[0], y).item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    loss = fit_epoch(model, optimizer, values, mask, y, aux,
                     {'batch_size': 4, 'aux_scale': .25}, contract, torch.device('cpu'), 2, 1)
    assert loss == pytest.approx(expected, abs=1e-6)
    assert all(p.grad is None for p in model.aux.parameters())
    assert model.necrose.weight.grad is not None


def test_unseen_and_missing_category_masked():
    contract = {'retained': ['toy'], 'targets': {'toy': {'type': 'categorical', 'vocabulary': ['known']}}}
    rows = pd.DataFrame({'toy': ['known', 'unseen', None]})
    np.testing.assert_array_equal(p5.encode_aux(rows, contract, {})['toy'], [0, -1, -1])


def test_training_loaders_reject_test_before_reading(tmp_path):
    with pytest.raises(ValueError, match='train/validation'):
        load_cached_features(pd.DataFrame({'tissue_id': ['synthetic']}), 'test', tmp_path)
    with pytest.raises(ValueError, match='train/validation'):
        p5.load_split_rows('test', {'retained': []})


def test_grouped_folds_never_split_patient():
    rows = pd.DataFrame({'patient_id': [f'toy_{i // 2}' for i in range(36)],
                         'necrose': [(i // 2) % 2 for i in range(36)]})
    for fit, valid in p5.check_inner_folds(rows):
        assert not set(rows.iloc[fit].patient_id) & set(rows.iloc[valid].patient_id)


def test_compact_and_full_token_parity(tmp_path):
    rng = np.random.default_rng(3)
    tokens = rng.normal(size=(2, 256, 512)).astype(np.float16)
    coords = np.zeros((2, 2), np.int64)
    full = tmp_path / 'toy_full.npz'
    compact = tmp_path / 'toy_compact.npz'
    np.savez(full, tokens=tokens, coords=coords)
    np.savez(compact, tile_features=tokens.astype(np.float32).mean(1), coords=coords)
    with np.load(full, allow_pickle=False) as a, np.load(compact, allow_pickle=False) as b:
        np.testing.assert_array_equal(tile_features(a), tile_features(b))


@pytest.mark.parametrize('bad', ['dtype', 'nan', 'coords'])
def test_feature_reader_rejects_invalid_arrays(tmp_path, bad):
    features = np.zeros((2, 512), np.float32)
    coords = np.zeros((2, 2), np.int64)
    if bad == 'dtype':
        features = features.astype(np.float64)
    elif bad == 'nan':
        features[0, 0] = np.nan
    else:
        coords = coords[:1]
    file = tmp_path / 'toy.npz'
    np.savez(file, tile_features=features, coords=coords)
    with np.load(file, allow_pickle=False) as cached, pytest.raises(ValueError):
        tile_features(cached)


def test_test_access_requires_explicit_authorization():
    with pytest.raises(SystemExit, match='allow-test-access'):
        do_evaluate({}, allow_test_access=False)


def test_metrics_single_class_and_clustered_interval():
    assert metrics(np.zeros(3), np.array([.1, .2, .3]), .5)['roc_auc'] is None
    toy = pd.DataFrame({'patient_id': ['toy_a', 'toy_a', 'toy_b', 'toy_b'],
                        'necrose': [0, 1, 0, 1], 'probability_ensemble': [.1, .8, .3, .7]})
    intervals = clustered_intervals(toy, .5, replicates=8, seed=2)
    assert all(v['attempted'] == 8 and v['valid'] == 8 for v in intervals.values())


def test_segmented_sdpa_matches_independent_segments():
    torch.manual_seed(5)
    qkv = torch.randn(5, 3, 2, 4)
    bounds = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    actual = sdpa_varlen_qkvpacked(qkv, bounds, 3)
    expected = []
    for start, end in [(0, 2), (2, 5)]:
        q, k, v = [qkv[start:end, i].transpose(0, 1).unsqueeze(0) for i in range(3)]
        expected.append(F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1))
    torch.testing.assert_close(actual, torch.cat(expected))
    with pytest.raises(ValueError, match='eval-mode'):
        sdpa_varlen_qkvpacked(qkv, bounds, 3, dropout_p=.1)


def test_synthetic_lock_evaluation_and_repeat_guard(tmp_path, monkeypatch):
    """Exercise lock/checkpoint/cache binding without study inputs or an encoder."""
    contract = synthetic_contract()
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    tissue_file = dataset / 'tissues.parquet'
    metadata_file = tmp_path / 'metadata.csv'
    contract_file = tmp_path / 'contract.json'
    marker_file = tmp_path / 'markers.csv'
    split_file = tmp_path / 'split.csv'
    manifest_file = tmp_path / 'tokens.json'
    contract_file.write_text(json.dumps(contract))
    marker_file.write_text('synthetic schema fixture only')
    split_file.write_text('synthetic split fixture only')
    manifest_file.write_text(json.dumps({'preprocessing_hash': 'synthetic', 'preprocessing': {'sha256': {}}}))
    heads = tmp_path / 'final_model_08'
    heads.mkdir()
    tokens = tmp_path / 'tokens'
    (tokens / 'validation').mkdir(parents=True)
    cache = tmp_path / 'test_features'
    cache.mkdir()
    monkeypatch.setattr(evaluation, 'R', tmp_path)
    monkeypatch.setattr(evaluation, 'CONTRACT_FILE', contract_file)
    monkeypatch.setattr(evaluation, 'HEAD_DIR', heads)
    monkeypatch.setattr(evaluation, 'LOCK_FILE', tmp_path / 'lock.json')
    monkeypatch.setattr(evaluation, 'RESULT_FILE', tmp_path / 'result.json')
    monkeypatch.setattr(evaluation, 'PRED_FILE', tmp_path / 'predictions.csv')
    monkeypatch.setattr(evaluation, 'TEST_CACHE', cache)
    monkeypatch.setattr(evaluation, 'TISSUES', tissue_file)
    monkeypatch.setattr(evaluation, 'SOURCE_META', metadata_file)
    monkeypatch.setattr(evaluation, 'BOOTSTRAP', {'replicates': 4, 'seed': 2})
    monkeypatch.setattr(p5, 'SPLIT_FILE', split_file)
    monkeypatch.setattr(p5, 'MARKER_MANIFEST', marker_file)
    monkeypatch.setattr(p5, 'TOKEN_MANIFEST', manifest_file)
    monkeypatch.setattr(p5, 'DATA', tissue_file)
    monkeypatch.setattr(p5, 'TOKEN_CACHE', tokens)
    monkeypatch.setattr(evaluation._submission_paths, 'dataset', lambda: dataset)
    rows, metadata = [], []
    rng = np.random.default_rng(9)
    for split in ['validation', 'test']:
        for i in range(4):
            tid = f'toy_{split}_{i}'
            rows.append({'tissue_id': tid, 'patient_id': f'toy_{split}_patient_{i // 2}',
                         'split': split, 'necrose': i % 2, 'original_file_name': tid,
                         **{name: (float(i) if spec['type'] == 'continuous' else 'class_a')
                            for name, spec in contract['targets'].items()}})
            metadata.append({'sample_id': tid, 'TMA': 1})
            features = rng.normal(size=(2, 512)).astype(np.float32)
            if split == 'validation':
                np.savez(tokens / split / f'{tid}.npz', tile_features=features, coords=np.zeros((2, 2)))
            else:
                np.savez(cache / f'{tid}.npz', features=features)
    pd.DataFrame(rows).to_parquet(tissue_file)
    pd.DataFrame(metadata).to_csv(metadata_file, index=False)
    for seed in evaluation.SEEDS:
        torch.manual_seed(seed)
        model = p5.CoreMTL(contract, .8)
        torch.save({'state_dict': model.state_dict(), 'mean': torch.zeros(512), 'std': torch.ones(512)},
                   heads / f'head08_seed{seed}.pt')
    evaluation.do_lock(contract)
    lock = json.loads(evaluation.LOCK_FILE.read_text())
    assert np.isfinite(lock['threshold']) and lock['test_used'] is False
    provenance = {'lock_sha256': evaluation.sha256(evaluation.LOCK_FILE),
                  'feature_hashes': {p.name: evaluation.sha256(p) for p in cache.glob('*.npz')}}
    (cache / 'provenance.json').write_text(json.dumps(provenance))
    # A changed private runtime input is rejected before any test inference.
    original = marker_file.read_bytes()
    marker_file.write_text('changed synthetic fixture')
    with pytest.raises(ValueError, match='changed'):
        evaluation.do_evaluate(contract, True)
    marker_file.write_bytes(original)
    evaluation.do_evaluate(contract, True)
    result = json.loads(evaluation.RESULT_FILE.read_text())
    assert result['n_cores'] == 4 and result['n_pids'] == 2 and result['test_used'] is True
    with pytest.raises(SystemExit, match='already exist'):
        evaluation.do_evaluate(contract, True)
