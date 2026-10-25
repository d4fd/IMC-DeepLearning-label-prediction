"""Phase 1 marker-embeddings runner (T4).

Consumes the reviewed marker manifest, downloads UniProt canonical FASTA
sequences with the pinned VirTues utility, then computes ESM2 sequence
embeddings.

Deviation (documented, user-approved): the pinned
`virtues.utils.compute_esm_embeddings` unconditionally requests contact maps
(`return_contacts=True`), whose L^2 tensors OOM the T4 for long sequences.
This runner replicates the pinned script's exact reduction (mean of the
last-layer token representations, layer 30) with `return_contacts=False`, and
checks numerical equivalence against separately generated pinned-tool reference
outputs (required under `marker_embeddings/pinned_tool_reference/`).

Run only in a prepared external environment; no reference assets are bundled.

Only public UniProt sequences are used; no RCC image data is read and no
training is performed.
"""
import submission_paths as _submission_paths

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(str(_submission_paths.root()))
MANIFEST = ROOT / "phase1_marker_manifest.csv"
FASTA_DIR = ROOT / "marker_fastas"
EMB_ROOT = ROOT / "marker_embeddings"
MODEL = "esm2_t30_150M_UR50D"
EMB_DIR = EMB_ROOT / MODEL
PINNED_REF = EMB_ROOT / "pinned_tool_reference"
REPL_DIR = EMB_ROOT / ("_replicated_" + MODEL)
ESM_DEVICE = os.environ.get("VIRTUES_ESM_DEVICE", "cpu")
ESM_LAYER = 30
EQUIVALENCE_TOL = 1e-5
REQUIRED_MODULES = ["esm", "Bio", "loguru", "pandas", "torch", "requests"]


def sh(args):
    print("$", " ".join(str(a) for a in args), flush=True)
    p = subprocess.run(args, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.stderr:
        print(p.stderr, flush=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {args}")
    return p


def main():
    found = {m: importlib.util.find_spec(m) is not None for m in REQUIRED_MODULES}
    print("module_probe", found, flush=True)
    missing_mods = [m for m, ok in found.items() if not ok]
    if missing_mods:
        raise SystemExit(f"missing modules in env: {missing_mods}")

    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    print("manifest_sha256", manifest_sha, flush=True)

    df = pd.read_csv(MANIFEST)
    assert list(df.columns) == [
        "source_index", "source_channel", "original_name", "hugo_name",
        "uniprot_id", "qc_pass", "nuclear_marker", "disposition", "notes",
    ], "unexpected manifest schema"
    assert df.source_index.tolist() == list(range(len(df))), "source_index must be 0..N-1"
    assert set(df.disposition) <= {"use", "exclude"}, "disposition must be use/exclude"
    assert df.qc_pass.eq(1).all(), "qc_pass mismatch"

    use = df[df.disposition == "use"].copy()
    excluded = df[df.disposition == "exclude"].copy()
    used_ids = use.uniprot_id.tolist()
    assert use.uniprot_id.notna().all()
    assert excluded.uniprot_id.isna().all(), "excluded channels must not map to UniProt"
    dupes = sorted({i for i in used_ids if used_ids.count(i) > 1})
    print("channels", len(df), "use", len(use), "exclude", len(excluded), "duplicate_ids", dupes, flush=True)

    # --- Download canonical FASTA sequences with the pinned utility (skips existing) ---
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    fasta_input = ROOT / "_marker_fasta_inputs.csv"
    use[["uniprot_id"]].rename(columns={"uniprot_id": "protein_id"}).to_csv(fasta_input, index=False)
    sh([sys.executable, "-m", "virtues.utils.download_fastas",
        "--output_dir", str(FASTA_DIR), "--input", str(fasta_input), "--id_column", "protein_id"])
    fastas = sorted(p.name for p in FASTA_DIR.glob("*.fasta"))
    missing_fasta = [i for i in used_ids if f"{i}.fasta" not in fastas]
    print("fastas", len(fastas), "missing", missing_fasta, flush=True)
    assert not missing_fasta, missing_fasta

    # --- Preserve the pinned tool's outputs (first run only) as the equivalence reference ---
    if not PINNED_REF.exists():
        PINNED_REF.mkdir(parents=True)
        if EMB_DIR.exists():
            for p in EMB_DIR.glob("*.pt"):
                shutil.copy2(p, PINNED_REF / p.name)
    pinned_ids = sorted(p.stem for p in PINNED_REF.glob("*.pt"))
    print("pinned_reference", len(pinned_ids), pinned_ids, flush=True)

    # --- Compute embeddings via the contact-free replication of the pinned reduction ---
    import torch
    from Bio import SeqIO
    from esm.pretrained import load_model_and_alphabet

    model, alphabet = load_model_and_alphabet(MODEL)
    model.eval()
    model.to(ESM_DEVICE)
    converter = alphabet.get_batch_converter()
    REPL_DIR.mkdir(parents=True, exist_ok=True)
    print("esm_device", ESM_DEVICE, flush=True)

    for f in sorted(FASTA_DIR.glob("*.fasta")):
        pid = f.stem
        seq = str(next(SeqIO.parse(str(f), "fasta")).seq)
        _, _, tokens = converter([(pid, seq)])
        tokens = tokens.to(ESM_DEVICE)
        with torch.no_grad():
            out = model(tokens, repr_layers=[ESM_LAYER])
        hidden = out["representations"][ESM_LAYER]
        length = int((tokens != alphabet.padding_idx).sum(1)[0])
        emb = hidden[0, 1: length - 1].mean(0)
        torch.save(emb.cpu(), REPL_DIR / f"{pid}.pt")

    repl_ids = sorted(p.stem for p in REPL_DIR.glob("*.pt"))
    missing_pt = [i for i in used_ids if i not in repl_ids]
    print("replicated", len(repl_ids), "missing", missing_pt, flush=True)
    assert not missing_pt, missing_pt

    # --- Equivalence check against the pinned tool outputs ---
    diffs = {}
    for pid in pinned_ids:
        a = torch.load(PINNED_REF / f"{pid}.pt", weights_only=True)
        b = torch.load(REPL_DIR / f"{pid}.pt", weights_only=True)
        diffs[pid] = float((a - b).abs().max())
    max_diff = max(diffs.values()) if diffs else float("nan")
    print("equivalence_max_abs_diff", max_diff, "over", len(diffs), "pinned markers", flush=True)
    assert diffs, "no pinned reference outputs to compare against"
    assert max_diff <= EQUIVALENCE_TOL, f"equivalence failed: {max_diff}"

    # --- Finalize a uniform embedding set at the loader path ---
    if EMB_DIR.exists():
        shutil.rmtree(EMB_DIR)
    REPL_DIR.rename(EMB_DIR)

    # --- Verify the VirTues loader contract (ordering + finiteness) ---
    sys.path.insert(0, str(ROOT))
    from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

    matrix = load_marker_embeddings(str(EMB_DIR))
    index = load_marker_embedding_dict(str(EMB_DIR))
    assert torch.isfinite(matrix).all(), "non-finite embedding values"
    assert matrix.shape[0] == len(used_ids), matrix.shape
    assert set(index.keys()) == set(used_ids), "loader keys != manifest use ids"
    expected_order = [f[:-3] for f in sorted(f"{i}.pt" for i in used_ids)]
    assert list(index.keys()) == expected_order, "loader ordering is not sorted-filename order"
    print("embedding_matrix", tuple(matrix.shape), str(matrix.dtype), flush=True)

    ordering = pd.DataFrame(sorted(index, key=index.get), columns=["uniprot_id"])
    ordering["embedding_index"] = ordering.uniprot_id.map(index)
    ordering.to_csv(ROOT / "marker_embedding_order.csv", index=False)
    for _, r in use.iterrows():
        print(f"{r.source_channel}\t{r.hugo_name}\t{r.uniprot_id}\tidx={index[r.uniprot_id]}", flush=True)

    print("PHASE1_MARKER_EMBEDDINGS_OK", flush=True)


if __name__ == "__main__":
    main()
