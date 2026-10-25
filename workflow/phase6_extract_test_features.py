"""Explicitly authorized test-feature extraction, adapted from the original evaluator.

Run only after the final-head/threshold lock; never for model development.
No images, features, locks or patient identifiers are bundled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import submission_paths as paths
import phase6_final08_evaluate as evaluation


def extract(ids):
    from virtues_t4_attention import install_t4_attention
    install_t4_attention()
    from spora_io import MultiplexImagingDataset
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict
    from safetensors.torch import load_file

    root = paths.root()
    embeddings = root / 'marker_embeddings/esm2_t30_150M_UR50D'
    model = MultiplexVirtues(prior_bias_embeddings=load_marker_embeddings(str(embeddings)))
    model.load_state_dict(load_file(str(root / 'checkpoints/virtues-sp32/model.safetensors')), strict=True)
    model = model.cuda().eval()
    marker_index = load_marker_embedding_dict(str(embeddings))
    dataset = MultiplexImagingDataset(name='rcc2025necrosis', path=str(paths.dataset()), modality='imc',
        standardization='quantile_clipping_log1p/uq_0.99_image', resolution='2.678571', tile_size=128,
        disable_quantile_mask=True, filter_list=['gaussian_blur'], use_mean_std=True)
    cache = evaluation.TEST_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    # A partial extraction must not silently reuse potentially stale feature files.
    if any(cache.iterdir()):
        raise FileExistsError('Test-feature directory is not empty; inspect it before any new extraction')
    for i, tissue_id in enumerate(ids):
        tissue = dataset.get_tissue(tissue_id, kind='uniprot_filtered', preprocess=True)
        marker_ids = torch.tensor([marker_index[u] for u in tissue.uniprot_ids], dtype=torch.long, device='cuda')
        features = []
        coords = dataset.tile_coordinates.get(tissue_id, [])
        for k in range(len(coords)):
            tile = dataset.get_tile(tissue_id, k, kind='uniprot_filtered')
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                output = model.encoder.forward_list([tile.image.cuda()], [marker_ids])
            # Preserve the original float16-cache -> float32 patch-mean sequence.
            tokens = output.patch_summary_tokens[0].reshape(-1, 512).float().cpu().numpy().astype(np.float16)
            if tokens.shape != (256, 512) or not np.isfinite(tokens).all():
                raise ValueError('Invalid encoder token array')
            features.append(tokens.astype(np.float32).mean(axis=0))
        if not features:
            raise ValueError('Test core has no valid tiles')
        np.savez_compressed(cache / f'{tissue_id}.npz', features=np.stack(features),
                            coords=np.asarray(coords, dtype=np.int32))
        print('EXTRACT', i + 1, len(ids), flush=True)
    provenance = {'lock_sha256': evaluation.sha256(evaluation.LOCK_FILE),
                  'feature_hashes': {p.name: evaluation.sha256(p) for p in sorted(cache.glob('*.npz'))}}
    (cache / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-test-access', action='store_true')
    args = parser.parse_args()
    if not args.allow_test_access:
        raise SystemExit('Test extraction requires --allow-test-access')
    if evaluation.RESULT_FILE.exists() or evaluation.PRED_FILE.exists():
        raise SystemExit('Test evaluation already exists; do not re-extract for selection')
    lock = json.loads(evaluation.LOCK_FILE.read_text())
    evaluation.verify_lock_inputs(lock)
    if not (paths.root() / 'checkpoints/virtues-sp32/model.safetensors').is_file():
        raise SystemExit('Externally licensed encoder weights are required')
    rows = pd.read_parquet(evaluation.TISSUES, columns=['tissue_id', 'patient_id', 'split'])
    test = rows[rows.split == 'test']
    development = rows[rows.split.isin(['train', 'validation'])]
    if test.empty or test.tissue_id.duplicated().any() or test.patient_id.isna().any():
        raise ValueError('Invalid test identifiers')
    if set(test.patient_id) & set(development.patient_id):
        raise ValueError('Patient leakage into test')
    extract(test.tissue_id.tolist())


if __name__ == '__main__':
    main()
