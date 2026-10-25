"""Phase 3: VirTues pretrained compatibility smoke test on the RCC Spora dataset.

Runs one training and one validation core through the source-pinned VirTues inference path
(the vendor MultiplexVirtues encoder forward_list, exactly as spora-bench SporaVirTuesWrapper.embed_tile
uses it) with identical spora_io preprocessing. Also exercises the released decoder on one tile.
Saves only non-identifying diagnostics (split, shapes, counts, hashes) plus the checkpoint hash.
"""
import submission_paths as _submission_paths
import os, json, hashlib
from pathlib import Path
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from einops import rearrange
import pandas as pd

from virtues_t4_attention import install_t4_attention
install_t4_attention()
from spora_io import MultiplexImagingDataset
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

ROOT = Path(str(_submission_paths.root()))
OUT = _submission_paths.dataset()
CKPT = ROOT / 'checkpoints/virtues-sp32/model.safetensors'
EMB = ROOT / 'marker_embeddings/esm2_t30_150M_UR50D'
VENDOR = ROOT / 'virtues'
RES = '2.678571'
TILE = 128


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert torch.cuda.is_available()
    man = json.load(open(ROOT / 'phase2_spora_dataset_manifest.json'))
    expected = man['expected_uniprots']
    emb = load_marker_embeddings(str(EMB))
    midict = load_marker_embedding_dict(str(EMB))
    model = MultiplexVirtues(prior_bias_embeddings=emb)
    missing, unexpected = model.load_state_dict(load_file(str(CKPT), device='cpu'), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model = model.cuda().eval()
    patch = model.encoder.patch_size
    ds = MultiplexImagingDataset(name='rcc2025necrosis', path=str(OUT), modality='imc',
        standardization='quantile_clipping_log1p/uq_0.99_image', resolution=RES, tile_size=TILE,
        disable_quantile_mask=True, filter_list=['gaussian_blur'], use_mean_std=True)
    t = pd.read_parquet(OUT / 'metadata/tissues.parquet')
    picks = {sp: t.loc[t['split'] == sp, 'tissue_id'].iloc[0] for sp in ['train', 'validation']}
    report = {'checkpoint_sha256': sha256(CKPT), 'marker_embeddings': int(emb.shape[0]),
              'embedding_dim': int(emb.shape[1]), 'patch_size': int(patch), 'tile_size': TILE,
              'inference_path': 'virtues.modules.multiplex_virtues.MultiplexVirtues.encoder.forward_list (as spora_bench SporaVirTuesWrapper.embed_tile)',
              'cores': {}}
    for sp, tid in picks.items():
        torch.cuda.reset_peak_memory_stats()
        tissue = ds.get_tissue(tid, kind='uniprot_filtered', preprocess=True)
        assert tuple(tissue.image.shape) == (len(expected), 224, 224), (sp, tissue.image.shape)
        assert bool(torch.isfinite(tissue.image).all())
        assert list(tissue.uniprot_ids) == expected, 'marker order mismatch vs manifest'
        midxs = torch.tensor([midict[u] for u in tissue.uniprot_ids], dtype=torch.long).cuda()
        coords = ds.tile_coordinates.get(tid, [])
        tok_shape = None
        n_fwd = 0
        for k in range(len(coords)):
            tile = ds.get_tile(tid, k, kind='uniprot_filtered')
            assert tuple(tile.image.shape) == (len(expected), TILE, TILE), (sp, k, tile.image.shape)
            assert bool(torch.isfinite(tile.image).all())
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                out = model.encoder.forward_list([tile.image.cuda()], [midxs])
            toks = rearrange(out.patch_summary_tokens[0], 'h w d -> (h w) d')
            assert bool(torch.isfinite(toks).all())
            tok_shape = list(toks.shape)
            n_fwd += 1
        tile = ds.get_tile(tid, 0, kind='uniprot_filtered')
        mask = torch.zeros((len(expected), TILE // patch, TILE // patch), dtype=torch.bool)
        mask[3] = True
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            full = model.forward([tile.image.cuda()], [midxs], [mask.cuda()])
        recon = full.decoded_multiplex[0]
        assert tuple(recon.shape) == (len(expected), TILE, TILE)
        assert bool(torch.isfinite(recon).all())
        peak = torch.cuda.max_memory_allocated() / 2**30
        report['cores'][sp] = {'tissue_sha256': hashlib.sha256(tid.encode()).hexdigest()[:16],
            'image_shape': list(tissue.image.shape), 'n_valid_tiles': len(coords), 'tiles_forwarded': n_fwd,
            'patch_token_shape': tok_shape, 'recon_shape': list(recon.shape),
            'peak_gpu_gib': round(peak, 3), 'marker_order_matches_manifest': True}
        print('core_ok', sp, 'tiles', len(coords), 'token', tok_shape, 'peak_gib', round(peak, 3), flush=True)
    report['tiles_forwarded_total'] = sum(v['tiles_forwarded'] for v in report['cores'].values())
    json.dump(report, open(ROOT / 'phase3_virtues_smoke_report.json', 'w'), indent=2)
    print('PHASE3_VIRTUES_SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
