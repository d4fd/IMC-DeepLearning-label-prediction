"""Marker-channel occlusion importance for the dropout-0.8 ensemble (T4, no batching).

For every development core (train + validation) and every one of the 41 marker
channels, the channel is set to its standardized mean (zero) in the preprocessed
tile, the frozen VirTues encoder is re-run per tile, and the change in the
ensemble necrosis probability is recorded. One forward per tile for the baseline
plus one per channel per tile (41x), then the eight-seed head is applied.

Train/validation only; the held-out test partition is never read. Results are
written per core for resume plus a summary ranking.
"""
from __future__ import annotations
import submission_paths as _submission_paths

import argparse
import json
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

import phase5_finetune_mtl as phase5

ROOT = phase5.ROOT
OUT = _submission_paths.dataset()
CKPT = ROOT / "checkpoints/virtues-sp32/model.safetensors"
EMB = ROOT / "marker_embeddings/esm2_t30_150M_UR50D"
MANIFEST = ROOT / "phase1_marker_manifest.csv"
HEAD_DIR = ROOT / "final_model_08"
OUTPUT = ROOT / "outputs/marker_importance"
SEEDS = tuple(range(42, 50))


def load_heads(contract, device):
    states = []
    mean = std = None
    for seed in SEEDS:
        ck = torch.load(HEAD_DIR / f"head08_seed{seed}.pt", map_location="cpu", weights_only=True)
        states.append(ck["state_dict"])
        mean, std = ck["mean"].numpy(), ck["std"].numpy()
    models = []
    for state in states:
        m = phase5.CoreMTL(contract, 0.8).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    assert torch.cuda.is_available()

    contract = json.loads(phase5.CONTRACT_FILE.read_text())
    manifest = pd.read_csv(MANIFEST)
    uni2hugo = dict(zip(manifest["uniprot_id"], manifest["hugo_name"]))
    models, mean, std = load_heads(contract, device)

    emb = load_marker_embeddings(str(EMB))
    midict = load_marker_embedding_dict(str(EMB))
    model = MultiplexVirtues(prior_bias_embeddings=emb)
    model.load_state_dict(load_file(str(CKPT), device="cpu"), strict=True)
    model = model.cuda().eval()

    ds = MultiplexImagingDataset(name="rcc2025necrosis", path=str(OUT), modality="imc",
                                 standardization="quantile_clipping_log1p/uq_0.99_image",
                                 resolution="2.678571", tile_size=128, disable_quantile_mask=True,
                                 filter_list=["gaussian_blur"], use_mean_std=True)
    tissues = pd.read_parquet(OUT / "metadata/tissues.parquet")
    dev = tissues[tissues["split"].isin(["train", "validation"])].sort_values("tissue_id")
    if args.limit:
        dev = dev.head(args.limit)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_delta = {}
    channel_names = None
    for i, row in enumerate(dev.itertuples()):
        tid = row.tissue_id
        out_path = OUTPUT / f"{tid}.npz"
        if out_path.exists():
            z = np.load(out_path)
            all_delta[tid] = z["delta"]
            if channel_names is None:
                channel_names = list(z["channels"])
            continue
        coords = ds.tile_coordinates.get(tid, [])
        if not coords:
            continue
        base_feats, abl_feats = [], []
        uniprot_ids = None
        for k in range(len(coords)):
            tile = ds.get_tile(tid, k, kind="uniprot_filtered")
            img = tile.image.float().cuda()
            if uniprot_ids is None:
                uniprot_ids = list(tile.uniprot_ids)
                channel_names = [uni2hugo[u] for u in uniprot_ids]
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                res = model.encoder.forward_list([img], [torch.tensor([midict[u] for u in uniprot_ids], dtype=torch.long, device="cuda")])
            base_feats.append(rearrange(res.patch_summary_tokens[0], "h w d -> (h w) d").float().mean(0).cpu().numpy())
            per_chan = []
            for c in range(img.shape[0]):
                img2 = img.clone()
                img2[c] = 0.0
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                    res2 = model.encoder.forward_list([img2], [torch.tensor([midict[u] for u in uniprot_ids], dtype=torch.long, device="cuda")])
                per_chan.append(rearrange(res2.patch_summary_tokens[0], "h w d -> (h w) d").float().mean(0).cpu().numpy())
            abl_feats.append(np.stack(per_chan))  # (C, 512)

        base = np.stack(base_feats)  # (T, 512)
        abl = np.stack(abl_feats)    # (T, C, 512)
        core_base = base.mean(0)
        core_abl = abl.mean(0)       # (C, 512)

        def score(vec):
            v = (vec - mean) / std
            t = torch.from_numpy(v[None, None, :].astype(np.float32)).cuda()
            m = torch.ones((1, 1), dtype=torch.bool, device="cuda")
            probs = []
            with torch.inference_mode():
                for mm in models:
                    logits, _ = mm(t, m)
                    probs.append(torch.sigmoid(logits.float()).item())
            return float(np.mean(probs))

        p_base = score(core_base)
        delta = np.array([score(core_abl[c]) - p_base for c in range(core_abl.shape[0])], dtype=np.float64)
        np.savez_compressed(out_path, delta=delta, channels=np.array(channel_names), base_probability=p_base,
                            necrose=row.necrose if pd.notna(row.necrose) else -1)
        all_delta[tid] = delta
        if (i + 1) % 25 == 0:
            print("CORE", i + 1, len(dev), flush=True)

    mat = np.stack([all_delta[t] for t in all_delta])
    mean_delta = mat.mean(0)
    rank = np.argsort(-np.abs(mean_delta))
    summary = {
        "model": "0.8-dropout eight-seed ensemble",
        "n_dev_cores": int(mat.shape[0]),
        "channels": list(channel_names),
        "mean_signed_delta_p": mean_delta.tolist(),
        "mean_abs_delta_p": np.abs(mat).mean(0).tolist(),
        "ranking_by_abs_delta_p": [channel_names[j] for j in rank],
        "test_used": False,
    }
    (OUTPUT / "marker_importance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "marker_importance_matrix.npz", delta=mat,
                        cores=np.array(list(all_delta)), channels=np.array(channel_names))
    print("MARKER_IMPORTANCE_DONE", json.dumps(summary["ranking_by_abs_delta_p"][:12]), flush=True)


if __name__ == "__main__":
    main()
