"""Phase 4 item 2: extract and cache frozen VirTues tile tokens for train and validation cores.

Uses the verified Phase 3 inference path (T4 SDPA adapter, frozen virtues-sp32, spora_io
quantile_clipping_log1p/uq_0.99_image + gaussian_blur, the dataset's 128-px tiles). Caches
per-tile patch-summary tokens per core under the ignored runtime token directory, plus a metadata
record with model/vendor commit, checkpoint hash, marker-manifest hash, preprocessing hash,
source-code commit + script hash, tile configuration, and split-artifact hash.
"""
import submission_paths as _submission_paths
import os, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from safetensors.torch import load_file

from virtues_t4_attention import install_t4_attention
install_t4_attention()
from spora_io import MultiplexImagingDataset
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

ROOT = Path(str(_submission_paths.root()))
OUT = _submission_paths.dataset()
TT = _submission_paths.tokens()
VENDOR = ROOT / 'virtues'
CKPT = ROOT / 'checkpoints/virtues-sp32/model.safetensors'
EMB = ROOT / 'marker_embeddings/esm2_t30_150M_UR50D'
RES = '2.678571'
TILE = 128
STD = 'quantile_clipping_log1p/uq_0.99_image'
FILELIST = [
    'imc/channels.parquet', 'imc/channels_per_tissue.parquet',
    'imc/2_678571mpp/standardization/quantile_clipping_log1p/uq_0.99_image/image_level_upper_quantiles.parquet',
    'imc/2_678571mpp/standardization/quantile_clipping_log1p/uq_0.99_image/global_level_means.parquet',
    'imc/2_678571mpp/standardization/quantile_clipping_log1p/uq_0.99_image/global_level_stds.parquet',
    'tiling/2_678571mpp/default/128_tile_coordinates.parquet',
    'metadata/tissues.parquet',
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def gitsha(repo):
    return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def main():
    assert torch.cuda.is_available()
    emb = load_marker_embeddings(str(EMB))
    midict = load_marker_embedding_dict(str(EMB))
    model = MultiplexVirtues(prior_bias_embeddings=emb)
    missing, unexpected = model.load_state_dict(load_file(str(CKPT), device='cpu'), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model = model.cuda().eval()
    ds = MultiplexImagingDataset(name='rcc2025necrosis', path=str(OUT), modality='imc', standardization=STD,
        resolution=RES, tile_size=TILE, disable_quantile_mask=True, filter_list=['gaussian_blur'], use_mean_std=True)
    tissues = pd.read_parquet(OUT / 'metadata' / 'tissues.parquet')

    for sp in ['train', 'validation']:
        (TT / sp).mkdir(parents=True, exist_ok=True)

    totals = {}
    token_dim = None
    for sp in ['train', 'validation']:
        ids = tissues.loc[tissues['split'] == sp, 'tissue_id'].tolist()
        n_core = n_tiles = 0
        for tid in ids:
            tissue = ds.get_tissue(tid, kind='uniprot_filtered', preprocess=True)
            midxs = torch.tensor([midict[u] for u in tissue.uniprot_ids], dtype=torch.long).cuda()
            coords = ds.tile_coordinates.get(tid, [])
            toks = []
            for k in range(len(coords)):
                tile = ds.get_tile(tid, k, kind='uniprot_filtered')
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model.encoder.forward_list([tile.image.cuda()], [midxs])
                t = rearrange(out.patch_summary_tokens[0], 'h w d -> (h w) d')
                assert bool(torch.isfinite(t).all())
                toks.append(t.float().cpu().numpy().astype(np.float16))
            if not toks:
                continue
            arr = np.stack(toks)
            assert np.isfinite(arr).all()
            token_dim = int(arr.shape[-1])
            np.savez_compressed(TT / sp / (tid + '.npz'), tokens=arr, coords=np.asarray(coords, dtype=np.int32))
            n_core += 1
            n_tiles += len(toks)
            if n_core % 50 == 0:
                print('  %s %d/%d cores' % (sp, n_core, len(ids)), flush=True)
        totals[sp] = {'cores': n_core, 'tiles': n_tiles}
        print('split done', sp, totals[sp], flush=True)

    preproc = {'standardization': STD, 'filter_list': ['gaussian_blur'], 'resolution': RES, 'tile_size': TILE,
               'sha256': {f: sha256(OUT / f) for f in FILELIST}}
    meta = {
        'vendor_commit': gitsha(VENDOR), 'checkpoint_sha256': sha256(CKPT),
        'marker_manifest_sha256': sha256(ROOT / 'phase1_marker_manifest.csv'),
        'preprocessing': preproc,
        'preprocessing_hash': hashlib.sha256(json.dumps(preproc, sort_keys=True).encode()).hexdigest(),
        'source_code_commit': gitsha(_submission_paths.code()),
        'extraction_script_sha256': sha256(_submission_paths.code() / 'phase4_extract_tile_tokens.py'),
        'dataset_manifest_sha256': sha256(ROOT / 'phase2_spora_dataset_manifest.json'),
        'split_artifact_sha256': sha256(ROOT / 'splits/rcc2025necrosis_pid_split.csv'),
        'tile_configuration': {'tile_size': TILE, 'stride': 64, 'tolerance': 0.95, 'coverage_goal': 0.95,
                               'min_gain_ratio': 0.1, 'strategy': 'default', 'resolution_dir': '2_678571mpp'},
        'inference_path': 'virtues.modules.multiplex_virtues.MultiplexVirtues.encoder.forward_list (as spora_bench SporaVirTuesWrapper.embed_tile)',
        'token_shape_per_tile': [256, token_dim], 'dtype': 'float16',
        'totals': totals,
    }
    json.dump(meta, open(TT / 'cache_metadata.json', 'w'), indent=2)
    print('cache_metadata', json.dumps({k: meta[k] for k in ['vendor_commit', 'source_code_commit', 'preprocessing_hash', 'split_artifact_sha256']}, indent=2), flush=True)
    print('totals', totals, flush=True)
    print('PHASE4_TILE_TOKENS_OK', flush=True)


if __name__ == '__main__':
    main()
