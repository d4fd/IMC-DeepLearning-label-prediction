"""Final-model single evaluation on an explicitly authorized test partition.

Heads operate on externally generated frozen tile features. No study features,
checkpoints, historical thresholds or historical lock records are included.

Two-step protocol for a new authorized run:
  --lock      compute the validation-ensemble Youden-J threshold and write the lock.
  --evaluate  require the lock, verify hashes, then score the test cores once.

Historical study evaluation reused a test partition. This export contains no
study results and does not turn that historical assessment into independent validation.
"""
from __future__ import annotations
import submission_paths as _submission_paths

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import phase5_finetune_mtl as p5

R = p5.ROOT
CONTRACT_FILE = p5.CONTRACT_FILE
HEAD_DIR = R / "final_model_08"
LOCK_FILE = R / "phase6_final08_model_lock.json"
RESULT_FILE = R / "phase6_final08_results.json"
PRED_FILE = R / "phase6_final08_test_predictions.csv"
VAL_CACHE = _submission_paths.tokens() / 'validation'
TEST_CACHE = R / "outputs/phase6/test_features"
TISSUES = _submission_paths.dataset() / 'metadata/tissues.parquet'
SOURCE_META = Path(str(_submission_paths.source() / 'metadata.csv'))
SEEDS = tuple(range(42, 50))
BOOTSTRAP = dict(replicates=2000, seed=20261001)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked_inputs():
    """Bind generated contracts, preprocessing and code, never publish these hashes."""
    inputs = [CONTRACT_FILE, p5.SPLIT_FILE, p5.MARKER_MANIFEST, p5.TOKEN_MANIFEST, TISSUES,
              SOURCE_META, Path(__file__), Path(p5.__file__),
              _submission_paths.code() / 'submission_paths.py',
              _submission_paths.code() / 'submission_features.py',
              _submission_paths.code() / 'phase6_extract_test_features.py',
              _submission_paths.code() / 'virtues_t4_attention.py']
    # Check actual preprocessing assets, not just the manifest's declared digest.
    manifest = json.loads(p5.TOKEN_MANIFEST.read_text())
    for name, expected in manifest['preprocessing']['sha256'].items():
        path = _submission_paths.dataset() / name
        if sha256(path) != expected:
            raise ValueError('Preprocessing asset differs from development manifest')
        inputs.append(path)
    inputs += sorted((R / 'marker_embeddings/esm2_t30_150M_UR50D').glob('*.pt'))
    if p5.CHECKPOINT.is_file():
        inputs.append(p5.CHECKPOINT)
    return {str(Path(p).resolve()): sha256(p) for p in inputs}


def verify_lock_inputs(lock):
    if locked_inputs() != lock['input_hashes']:
        raise ValueError('Locked runtime inputs or evaluation code changed')
    for name, digest in lock['hashes'].items():
        if sha256(R / name) != digest:
            raise ValueError(f'checkpoint hash mismatch: {name}')


def load_heads(contract):
    models, norms = [], []
    for seed in SEEDS:
        ck = torch.load(HEAD_DIR / f"head08_seed{seed}.pt", map_location="cpu", weights_only=True)
        m = p5.CoreMTL(contract, 0.8)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
        norms.append((ck["mean"].numpy(), ck["std"].numpy()))
    mean, std = norms[0]
    for m, s in norms[1:]:
        if not (np.allclose(m, mean) and np.allclose(s, std)):
            raise ValueError("head normalizers differ across seeds")
    return models, mean, std


def core_probs(model, values, mask):
    out = []
    with torch.inference_mode():
        for i in range(0, len(values), 64):
            logits, _ = model(values[i:i + 64], mask[i:i + 64])
            out.append(torch.sigmoid(logits.float()).numpy())
    return np.concatenate(out)


def metrics(y, p, t):
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else None
    spec = float(tn / (tn + fp)) if (tn + fp) else None
    return {
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier_score": float(brier_score_loss(y, p)),
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2 if (both and sens is not None and spec is not None) else None,
        "accuracy": float((tn + tp) / len(y)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def clustered_intervals(frame, threshold, replicates=2000, seed=20261001):
    groups = [g.index.to_numpy() for _, g in frame.reset_index(drop=True).groupby("patient_id")]
    rng = np.random.default_rng(seed)
    y = frame.necrose.to_numpy(); p = frame.probability_ensemble.to_numpy()
    keys = [k for k in metrics(y, p, threshold) if k != "confusion_matrix"]
    draws = {k: [] for k in keys}
    for _ in range(replicates):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        for k, v in metrics(y[ix], p[ix], threshold).items():
            if k in draws and v is not None:
                draws[k].append(v)
    return {k: {"lower": float(np.quantile(v, .025)) if v else None,
                "upper": float(np.quantile(v, .975)) if v else None,
                "valid": len(v), "attempted": replicates} for k, v in draws.items()}


def do_lock(contract):
    if LOCK_FILE.exists():
        raise SystemExit("Lock already exists; inspect it, never overwrite")
    rows = p5.load_split_rows("validation", contract)
    feats = p5.load_cached_features(rows, "validation")
    models, mean, std = load_heads(contract)
    x, mk = p5.pack_features(feats, np.arange(len(rows)), mean, std)
    y = rows.necrose.astype(int).to_numpy()
    per = [core_probs(m, x, mk) for m in models]
    ens = np.mean(np.stack(per), axis=0)
    fpr, tpr, thr = roc_curve(y, ens)
    j = tpr - fpr
    finite = np.flatnonzero(np.isfinite(thr))
    threshold = float(thr[finite[int(np.argmax(j[finite]))]])
    lock = {
        "version": 1,
        "model_name": "0.8-dropout-final-model",
        "model_fit_id": "authorized-runtime-fit",
        "seeds": list(SEEDS),
        "ensemble": "arithmetic mean of eight sigmoid probabilities; classify probability >= threshold",
        "threshold": threshold,
        "threshold_source": "Youden J on the eight-seed ensemble validation predictions",
        "validation_ensemble_pr_auc": float(average_precision_score(y, ens)),
        "validation_ensemble_roc_auc": float(roc_auc_score(y, ens)),
        "hashes": {f"final_model_08/head08_seed{s}.pt": sha256(HEAD_DIR / f"head08_seed{s}.pt") for s in SEEDS},
        "input_hashes": locked_inputs(),
        "contract_sha256": sha256(CONTRACT_FILE),
        "evaluation_code_sha256": sha256(__file__),
        "split_sha256": sha256(p5.SPLIT_FILE),
        "marker_manifest_sha256": sha256(p5.MARKER_MANIFEST),
        "preprocessing_hash": json.loads(p5.TOKEN_MANIFEST.read_text())["preprocessing_hash"],
        "authorization": "Caller must separately authorize test access for this runtime fit.",
        "selection": "complete; no retraining, tuning, recalibration, threshold change, or seed/fit selection after test access",
        "partitions_at_lock": ["train", "validation"],
        "test_used": False,
    }
    LOCK_FILE.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    print("PHASE6_FINAL08_LOCK_CREATED", json.dumps({"threshold": threshold,
          "val_pr_auc": lock["validation_ensemble_pr_auc"]}))


def do_evaluate(contract, allow_test_access):
    if not allow_test_access:
        raise SystemExit('Test access requires --allow-test-access')
    if RESULT_FILE.exists() or PRED_FILE.exists():
        raise SystemExit('Evaluation outputs already exist; refusing a repeated assessment or overwrite')
    if not LOCK_FILE.exists():
        raise SystemExit("Lock missing; run --lock first")
    lock = json.loads(LOCK_FILE.read_text())
    verify_lock_inputs(lock)
    if (lock['contract_sha256'] != sha256(CONTRACT_FILE)
            or lock['evaluation_code_sha256'] != sha256(__file__)
            or lock['split_sha256'] != sha256(p5.SPLIT_FILE)
            or lock['marker_manifest_sha256'] != sha256(p5.MARKER_MANIFEST)
            or lock['preprocessing_hash'] != json.loads(p5.TOKEN_MANIFEST.read_text())['preprocessing_hash']):
        raise ValueError('Locked runtime inputs or evaluation code changed')
    for name, digest in lock["hashes"].items():
        if sha256(R / name) != digest:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    tissue = pd.read_parquet(TISSUES)
    te = tissue[tissue.split == "test"].reset_index(drop=True)
    if te.empty or te.tissue_id.duplicated().any() or te.patient_id.isna().any():
        raise ValueError('Invalid or empty test partition')
    development = tissue[tissue.split.isin(['train', 'validation'])]
    if set(te.patient_id.astype(str)) & set(development.patient_id.astype(str)):
        raise ValueError('Patient leakage into test partition')
    provenance = json.loads((TEST_CACHE / 'provenance.json').read_text())
    if provenance['lock_sha256'] != sha256(LOCK_FILE):
        raise ValueError('Test feature provenance belongs to a different lock')
    if set(provenance['feature_hashes']) != {f'{tid}.npz' for tid in te.tissue_id}:
        raise ValueError('Test feature provenance coverage mismatch')
    feats = []
    for tid in te.tissue_id:
        path = TEST_CACHE / f"{tid}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != provenance['feature_hashes'][path.name]:
            raise ValueError('Test feature hash mismatch')
        with np.load(path, allow_pickle=False) as cached:
            values = cached['features']
            if values.ndim != 2 or values.shape[1] != 512 or len(values) == 0 or not np.isfinite(values).all():
                raise ValueError('Invalid test tile features')
            feats.append(values)
    models, mean, std = load_heads(contract)
    x, mk = p5.pack_features(feats, np.arange(len(te)), mean, std)
    y = te.necrose.astype(int).to_numpy()
    per = [core_probs(m, x, mk) for m in models]
    ens = np.mean(np.stack(per), axis=0)
    frame = te[["tissue_id", "patient_id", "necrose", "original_file_name"]].copy()
    frame["probability_ensemble"] = ens
    frame["prediction"] = (ens >= lock["threshold"]).astype(int)
    meta = pd.read_csv(SOURCE_META, usecols=["sample_id", "TMA"])
    frame["TMA"] = frame.original_file_name.map(meta.set_index("sample_id").TMA).astype(int).astype(str)
    frame.to_csv(PRED_FILE, index=False)
    result = {
        "model_name": lock["model_name"], "model_fit_id": lock["model_fit_id"],
        "lock_sha256": sha256(LOCK_FILE), "threshold": lock["threshold"],
        "n_cores": len(frame), "n_pids": int(frame.patient_id.nunique()), "positive": int(y.sum()),
        "metrics": metrics(y, ens, lock["threshold"]),
        "intervals": clustered_intervals(frame, lock["threshold"], **BOOTSTRAP),
        "subgroups": {}, "partition": "caller-authorized test partition; independence must be established externally",
        "test_used": True,
    }
    for tma, g in frame.groupby("TMA"):
        result["subgroups"][tma] = {
            "n_cores": len(g), "n_pids": int(g.patient_id.nunique()), "positive": int(g.necrose.sum()),
            "metrics": metrics(g.necrose.to_numpy(), g.probability_ensemble.to_numpy(), lock["threshold"]),
            "intervals": clustered_intervals(g, lock["threshold"], **BOOTSTRAP),
        }
    result["prediction_sha256"] = sha256(PRED_FILE)
    RESULT_FILE.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("PHASE6_FINAL08_EVALUATE_COMPLETE", json.dumps(result["metrics"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-test-access", action="store_true")
    args = parser.parse_args()
    contract = json.loads(CONTRACT_FILE.read_text())
    if args.lock:
        do_lock(contract)
    elif args.evaluate:
        do_evaluate(contract, allow_test_access=args.allow_test_access)
    else:
        print("No action. Use --lock then --evaluate.")


if __name__ == "__main__":
    main()
