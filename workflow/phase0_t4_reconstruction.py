"""Reproduce the pinned VirTues reconstruction demo on T4 with SDPA.

Requires a separately obtained upstream example and PROJECT_DEMO_TISSUE_ID.
No example tissue, image, label or upstream weights are bundled here.
"""

import hashlib
import json
import platform
import os
from pathlib import Path
import submission_paths as _submission_paths

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from virtues_t4_attention import install_t4_attention

# The vendor layer imports flash_attn at module import time. Bind the T4
# implementation before importing any VirTues model or dataset modules.
install_t4_attention()
from spora_io import MultiplexImagingDataset
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.masking import generate_mask
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict


ROOT = _submission_paths.root()
VENDOR = ROOT / "virtues"
CHECKPOINT = ROOT / "checkpoints/virtues-sp32/model.safetensors"
EXPECTED_SHA256 = "974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7"
TISSUE_ID = os.environ.get("PROJECT_DEMO_TISSUE_ID")


def main():
    if not TISSUE_ID:
        raise SystemExit("Set PROJECT_DEMO_TISSUE_ID to an authorized upstream example tissue.")
    torch.manual_seed(42)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7
    assert CHECKPOINT.exists()
    checksum = hashlib.file_digest(CHECKPOINT.open("rb"), "sha256").hexdigest()
    assert checksum == EXPECTED_SHA256, "Checkpoint digest mismatch"

    conf = OmegaConf.load(VENDOR / "configs/base_config.yaml")
    conf.experiments_dir = str(ROOT / "outputs")
    conf.experiment.name = "phase0-t4-reconstruction"
    conf.experiment.seed = 42
    conf.experiment.wandb_mode = "disabled"
    conf.datasets_config = str(VENDOR / "configs/datasets/example_config.yaml")
    conf.marker_embeddings_dir = str(VENDOR / "assets/example_dataset/marker_embeddings")
    ds_conf = OmegaConf.load(conf.datasets_config).datasets.example_dataset
    print("config", OmegaConf.to_yaml(conf, resolve=True), flush=True)
    print("environment", json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "checkpoint_sha256": checksum, "attention": "segmented torch SDPA (T4)",
    }), flush=True)

    embeddings = load_marker_embeddings(conf.marker_embeddings_dir)
    marker_indices = load_marker_embedding_dict(conf.marker_embeddings_dir)
    model = MultiplexVirtues(prior_bias_embeddings=embeddings)
    missing, unexpected = model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    assert not missing and not unexpected
    model = model.cuda().eval()
    print("checkpoint_loaded", len(marker_indices), "marker embeddings", flush=True)

    dataset = MultiplexImagingDataset(
        name=ds_conf.name,
        path=str(VENDOR / ds_conf.path),
        modality=ds_conf.modality,
        standardization=ds_conf.standardization,
        resolution=ds_conf.resolution,
        tile_size=conf.data.tile_size,
        disable_quantile_mask=True,
        filter_list=["gaussian_blur"],
        use_mean_std=True,
    )
    tissue = dataset.get_tissue(tissue_id=TISSUE_ID, kind="uniprot_filtered")
    x = tissue.image
    uniprot_ids = tissue.uniprot_ids
    assert len(uniprot_ids) == x.shape[0] and all(uid in marker_indices for uid in uniprot_ids)
    midxs = torch.tensor([marker_indices[uid] for uid in uniprot_ids], dtype=torch.long)
    crop = x[:, 200:328, 200:328]
    assert crop.shape[1:] == (128, 128) and torch.isfinite(crop).all()
    print("example_tissue", TISSUE_ID, "image", tuple(x.shape), "crop", tuple(crop.shape),
          "markers", len(uniprot_ids), flush=True)

    crop = crop.cuda()
    # The released demo creates midx = midxs.cuda(), then passes midxs to forward.
    # Keep the same forward arguments; the released model handles marker indices.
    for mode in ("partial", "full_channel"):
        if mode == "partial":
            mask = generate_mask(C=crop.shape[0], H=crop.shape[1] // 8,
                                 W=crop.shape[2] // 8, masking_ratio=(0.6, 1.0))
        else:
            mask = torch.zeros_like(mask)
            mask[3] = True
        mask = mask.cuda()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            output = model.forward([crop], [midxs], [mask])
            recon = output.decoded_multiplex[0]
        assert tuple(recon.shape) == tuple(crop.shape)
        assert torch.isfinite(recon).all()
        print("reconstruction", json.dumps({
            "mode": mode, "shape": list(recon.shape), "dtype": str(recon.dtype),
            "mask_fraction": round(mask.float().mean().item(), 5),
            "recon_min": round(recon.min().item(), 5),
            "recon_max": round(recon.max().item(), 5),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }), flush=True)
    print("PHASE0_T4_RECONSTRUCTION_OK", flush=True)


if __name__ == "__main__":
    main()
