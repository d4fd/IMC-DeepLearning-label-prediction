"""Phase 5 follow-up: eight-seed validation-based common-epoch early stopping.

The hyperparameter recipe is fixed to the preceding runtime search selection.
Default mode performs a guarded, no-training preflight. ``--smoke`` exercises
the shared-epoch controller on synthetic tensors only. ``--execute`` trains all
eight seeds on labeled training cores and uses validation PR-AUC solely to
choose one common epoch and report the selected ensemble. Test data is never
loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.nn import functional as F

import phase5_finetune_mtl as phase5

ROOT = phase5.ROOT
CONTRACT_FILE = phase5.CONTRACT_FILE
TOKEN_MANIFEST = phase5.TOKEN_MANIFEST
TOKEN_CACHE_METADATA = phase5.TOKEN_CACHE_METADATA
SPLIT_FILE = phase5.SPLIT_FILE
MARKER_MANIFEST = phase5.MARKER_MANIFEST
CHECKPOINT = phase5.CHECKPOINT
SOURCE_RESULTS = ROOT / "phase5_finetune_mtl_results.json"
PROTOCOL_LOCK = ROOT / "phase5_validation_earlystop_8seed_protocol.json"
RESULT_FILE = ROOT / "phase5_validation_earlystop_8seed_results.json"
OUTPUT = ROOT / "outputs/phase5_validation_earlystop_8seed"

SEEDS = tuple(range(42, 50))
MAX_EPOCHS = 300
PATIENCE = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_recipe():
    report = json.loads(SOURCE_RESULTS.read_text())
    if report["test_used"] or report["partitions_used"] != ["train", "validation"]:
        raise ValueError("Unexpected source Phase 5 result scope")
    return report["selected_config"], report


def make_protocol_lock(contract, manifest, prior_result, config):
    return {
        "protocol_version": 3,
        "status": "predeclared_validation_early_stopping",
        "source_hyperparameter_selection": {
            "candidate_id": prior_result["selected_candidate_id"],
            "source_results_sha256": sha256(SOURCE_RESULTS),
            "selected_config": config,
            "inner_training_pid_cv_mean_pr_auc": prior_result["inner_selection"]["mean_pr_auc"],
        },
        "model": {
            "encoder": "frozen virtues-sp32; verified cached train/validation tile tokens",
            "selected_head": phase5.SELECTED_HEAD,
            "pooling": "mean over patch-summary features per tile, then mean tiles per core",
            "auxiliaries": contract["retained"],
            "all_auxiliary_losses_active": True,
        },
        "training": {
            "seeds": list(SEEDS),
            "training_partition": "all labeled train cores; PID remains grouping/audit only",
            "validation_partition": "used for epoch selection and final comparison",
            "validation_metric": "mean per-seed PR-AUC at each shared epoch",
            "common_epoch_for_all_seeds": True,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "best_epoch_rule": "highest mean validation PR-AUC across all eight seeds; strict improvement keeps the earliest tied epoch",
            "checkpoint_rule": "retain all eight seed checkpoints from the common best epoch",
        },
        "validation_role": "selection/stopping and descriptive assessment; score is not an independent estimate",
        "threshold": "Youden J from the selected-epoch eight-seed ensemble validation predictions",
        "hashes": {
            "aux_manifest_sha256": sha256(CONTRACT_FILE),
            "marker_manifest_sha256": sha256(MARKER_MANIFEST),
            "token_manifest_sha256": sha256(TOKEN_MANIFEST),
            "tile_cache_metadata_sha256": sha256(TOKEN_CACHE_METADATA),
            "preprocessing_hash": manifest["preprocessing_hash"],
            "split_sha256": sha256(SPLIT_FILE),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_results_sha256": sha256(SOURCE_RESULTS),
        },
        "partitions_permitted": ["train", "validation"],
        "test_used": False,
    }


def batch_loss(model, values, mask, y, aux, config, contract, device):
    logits, heads = model(values, mask)
    loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    raw_losses = {"necrose": float(loss.detach())}
    for name in contract["retained"]:
        labels = aux[name]
        kind = contract["targets"][name]["type"]
        present = torch.isfinite(labels) if kind == "continuous" else labels >= 0
        if int(present.sum()) == 0:
            continue
        raw = (
            F.huber_loss(heads[name].squeeze(-1)[present].float(), labels[present].float(), delta=1.0)
            if kind == "continuous"
            else F.cross_entropy(heads[name][present].float(), labels[present].long())
        )
        loss = loss + float(config["aux_scale"]) * raw
        raw_losses[name] = float(raw.detach())
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("Non-finite training loss")
    return loss, raw_losses


def fit_epoch(model, optimizer, values, mask, y, aux, config, contract, device, seed, epoch):
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(y))
    loss_values = []
    for start in range(0, len(order), int(config["batch_size"])):
        ix = torch.as_tensor(order[start:start + int(config["batch_size"])], dtype=torch.long)
        xb, mb, yb = values[ix].to(device), mask[ix].to(device), y[ix].to(device)
        ab = {name: labels[ix].to(device) for name, labels in aux.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, xb, mb, yb, ab, config, contract, device)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("Non-finite gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError("Non-finite trainable parameter")
        loss_values.append(float(loss.detach()))
    return float(np.mean(loss_values))


def fit_eight_seed_common_epoch(train_data, validation_data, config, contract, seeds, device,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE):
    train_x, train_mask, train_y, train_aux = train_data
    val_x, val_mask, val_y, val_aux = validation_data
    models, optimizers = [], []
    for seed in seeds:
        phase5.seed_everything(seed)
        model = phase5.CoreMTL(contract, float(config["dropout"])).to(device)
        optimizer_class = torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.AdamW
        optimizer = optimizer_class(model.parameters(), lr=float(config["learning_rate"]),
                                    weight_decay=float(config["weight_decay"]))
        models.append(model)
        optimizers.append(optimizer)

    best_mean_pr = -1.0
    best_epoch = 0
    best_states = None
    best_predictions = None
    history = []
    for epoch in range(1, max_epochs + 1):
        train_losses = []
        per_seed_metrics = []
        per_seed_predictions = []
        per_seed_coverage = []
        for index, (seed, model, optimizer) in enumerate(zip(seeds, models, optimizers)):
            train_losses.append(fit_epoch(model, optimizer, train_x, train_mask, train_y,
                                          train_aux, config, contract, device, seed, epoch))
            metrics, predictions, coverage = phase5.evaluate(
                model, val_x, val_mask, val_y, val_aux, contract, device, int(config["batch_size"]))
            per_seed_metrics.append(metrics)
            per_seed_predictions.append(predictions)
            per_seed_coverage.append(coverage)
        mean_pr = float(np.mean([metrics["pr_auc"] for metrics in per_seed_metrics]))
        history.append({
            "epoch": epoch,
            "mean_validation_pr_auc": mean_pr,
            "per_seed_validation_pr_auc": [metrics["pr_auc"] for metrics in per_seed_metrics],
            "per_seed_validation_roc_auc": [metrics["roc_auc"] for metrics in per_seed_metrics],
            "mean_training_batch_loss": float(np.mean(train_losses)),
        })
        if mean_pr > best_mean_pr:
            best_mean_pr = mean_pr
            best_epoch = epoch
            best_states = [
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                for model in models
            ]
            best_predictions = [probabilities.copy() for probabilities in per_seed_predictions]
            best_seed_metrics = copy.deepcopy(per_seed_metrics)
            best_aux_coverage = copy.deepcopy(per_seed_coverage)
        elif epoch - best_epoch >= patience:
            break
        print("EPOCH", epoch, "MEAN_SEED_VAL_PR_AUC", round(mean_pr, 6), flush=True)

    if best_states is None or best_predictions is None:
        raise RuntimeError("No valid common stopping epoch was selected")
    for model, state in zip(models, best_states):
        model.load_state_dict(state)
    return {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_mean_seed_validation_pr_auc": best_mean_pr,
        "best_seed_metrics": best_seed_metrics,
        "best_predictions": best_predictions,
        "best_states": best_states,
        "best_aux_coverage": best_aux_coverage,
        "epoch_history": history,
    }


def build_data(rows, features, contract, age_stats, normalizer, indices):
    mean, std = normalizer
    values, mask = phase5.pack_features(features, indices, mean, std)
    aux = phase5.encode_aux(rows, contract, age_stats)
    y, aux_tensors = phase5.pack_targets(rows, aux, contract)
    return values, mask, y, aux_tensors


def run_training(contract, manifest, config, train, train_features, validation, validation_features,
                 device, seeds=SEEDS, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    train_indices = np.arange(len(train))
    validation_indices = np.arange(len(validation))
    mean, std, tile_count = phase5.fit_normalizer(train_features, train_indices)
    age_stats = phase5.age_statistics(train)
    train_data = build_data(train, train_features, contract, age_stats, (mean, std), train_indices)
    validation_data = build_data(validation, validation_features, contract, age_stats,
                                 (mean, std), validation_indices)
    stopping = fit_eight_seed_common_epoch(train_data, validation_data, config, contract,
                                           seeds, device, max_epochs, patience)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for seed, state in zip(seeds, stopping["best_states"]):
        torch.save({"state_dict": state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                    "config": config, "seed": seed, "common_best_epoch": stopping["best_epoch"],
                    "contract_sha256": sha256(CONTRACT_FILE)},
                   OUTPUT / f"phase5_earlystop_seed{seed}.pt")

    y = validation.necrose.astype(int).to_numpy()
    seed_predictions = stopping["best_predictions"]
    ensemble = np.mean(np.stack(seed_predictions), axis=0)
    seed_results = []
    for seed, metrics, probs, coverage in zip(seeds, stopping["best_seed_metrics"],
                                               seed_predictions, stopping["best_aux_coverage"]):
        seed_results.append({"seed": seed, "common_epoch": stopping["best_epoch"],
                             "metrics": {**metrics, "brier_score": float(brier_score_loss(y, probs))},
                             "finite_checks_passed": True, "validation_aux_coverage": coverage})
    ensemble_metrics = {
        "pr_auc": float(average_precision_score(y, ensemble)),
        "roc_auc": float(roc_auc_score(y, ensemble)),
        "brier_score": float(brier_score_loss(y, ensemble)),
        **phase5.threshold_metrics(y, ensemble),
    }
    if not np.isfinite([value for value in ensemble_metrics.values() if isinstance(value, (int, float))]).all():
        raise ValueError("Non-finite ensemble validation metric")
    pd.DataFrame({"tissue_id": validation.tissue_id.astype(str), "necrose": y,
                  **{f"probability_seed{seed}": probs for seed, probs in zip(seeds, seed_predictions)},
                  "probability_ensemble": ensemble}).to_csv(
        OUTPUT / "phase5_earlystop_8seed_validation_predictions.csv", index=False)
    (OUTPUT / "phase5_earlystop_8seed_epoch_log.json").write_text(
        json.dumps(stopping["epoch_history"], indent=2, allow_nan=False) + "\n")
    report = {
        "protocol_version": 3,
        "encoder_frozen": True,
        "selected_hyperparameter_source": str(SOURCE_RESULTS.name),
        "selected_candidate_id": json.loads(SOURCE_RESULTS.read_text())["selected_candidate_id"],
        "selected_config": config,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "early_stopping": {
            "partition": "all validation labeled cores",
            "metric": "mean per-seed validation PR-AUC at the common epoch",
            "common_best_epoch": stopping["best_epoch"],
            "epochs_run": stopping["epochs_run"],
            "patience": patience,
            "max_epochs": max_epochs,
            "validation_used_for_stopping": True,
        },
        "seed_results_at_common_epoch": seed_results,
        "validation_ensemble": ensemble_metrics,
        "normalizer_fit_tiles": tile_count,
        "normalizer_source": "verified train token cache only",
        "aux_manifest_sha256": sha256(CONTRACT_FILE),
        "marker_manifest_sha256": manifest["marker_manifest_sha256"],
        "preprocessing_hash": manifest["preprocessing_hash"],
        "split_sha256": manifest["split_artifact_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "vendor_commit": manifest["vendor_commit"],
        "partitions_used": ["train", "validation"],
        "test_used": False,
        "validation_metric_role": "stopping and development assessment; expect selection optimism",
    }
    RESULT_FILE.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PHASE5_EIGHT_SEED_EARLYSTOP_COMPLETE", stopping["best_epoch"],
          stopping["best_mean_seed_validation_pr_auc"], ensemble_metrics["pr_auc"], flush=True)
    return report


def make_lock(contract, manifest, prior_result):
    return {
        "protocol_version": 3,
        "status": "predeclared_validation_early_stopping",
        "selected_candidate_id": prior_result["selected_candidate_id"],
        "selected_config": prior_result["selected_config"],
        "model": {"encoder_frozen": True, "selected_head": phase5.SELECTED_HEAD,
                  "auxiliaries": contract["retained"]},
        "fit": {"seeds": list(SEEDS), "training_cores": "all labeled train cores",
                "validation_role": "early stopping and development comparison",
                "stopping_metric": "mean per-seed validation PR-AUC at a common epoch",
                "patience": PATIENCE, "max_epochs": MAX_EPOCHS,
                "best_epoch_rule": "highest mean validation PR-AUC; strict improvement retains earliest tie",
                "test_used": False},
        "hashes": {"prior_phase5_results_sha256": sha256(SOURCE_RESULTS),
                   "aux_manifest_sha256": sha256(CONTRACT_FILE),
                   "marker_manifest_sha256": sha256(MARKER_MANIFEST),
                   "token_manifest_sha256": sha256(TOKEN_MANIFEST),
                   "tile_cache_metadata_sha256": sha256(TOKEN_CACHE_METADATA),
                   "split_sha256": sha256(SPLIT_FILE),
                   "checkpoint_sha256": manifest["checkpoint_sha256"],
                   "preprocessing_hash": manifest["preprocessing_hash"]},
        "partitions_permitted": ["train", "validation"],
        "test_used": False,
    }


def synthetic_smoke(contract, device):
    rng = np.random.default_rng(123)
    def bundle(count):
        values = torch.from_numpy(rng.normal(size=(count, 3, 512)).astype(np.float32))
        mask = torch.ones((count, 3), dtype=torch.bool)
        y = torch.tensor([0, 1] * (count // 2), dtype=torch.float32)
        aux = {}
        for name in contract["retained"]:
            spec = contract["targets"][name]
            if spec["type"] == "continuous":
                aux[name] = torch.from_numpy(rng.normal(size=count).astype(np.float32))
            else:
                aux[name] = torch.tensor([index % len(spec["vocabulary"]) for index in range(count)], dtype=torch.long)
        return values, mask, y, aux
    data = fit_eight_seed_common_epoch(bundle(16), bundle(8), phase5_result_config(), contract,
                                      (42, 43), device, max_epochs=2, patience=2)
    assert data["best_epoch"] in (1, 2) and len(data["best_states"]) == 2
    print("PHASE5_EIGHT_SEED_CONTROLLER_SMOKE_OK", data["best_epoch"], flush=True)


def phase5_result_config():
    return json.loads(SOURCE_RESULTS.read_text())["selected_config"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run eight-seed validation-based common-epoch stopping")
    parser.add_argument("--smoke", action="store_true", help="Exercise the stopping controller on synthetic tensors only")
    args = parser.parse_args()
    contract, manifest, _ = phase5.load_contracts()
    config, prior_result = selected_recipe()
    if config != phase5_result_config():
        raise ValueError("Selected recipe changed since Phase 5 hyperparameter search")
    train = phase5.load_train_rows(contract)
    folds = phase5.check_inner_folds(train)
    train_features = phase5.load_cached_features(train, "train")
    lock = make_lock(contract, manifest, prior_result)
    PROTOCOL_LOCK.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    print("PHASE5_8SEED_PREFLIGHT_OK", json.dumps({"train_cores": len(train),
          "train_pids": int(train.patient_id.nunique()), "folds": len(folds),
          "train_token_cores_verified": len(train_features), "seeds": list(SEEDS),
          "validation_used_for_stopping": True, "test_used": False,
          "execute": args.execute, "smoke": args.smoke}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        synthetic_smoke(contract, device)
        return
    if not args.execute:
        print("No weights trained; use --smoke or --execute after reviewing the lock.", flush=True)
        return
    # Validation is intentionally loaded only in execution mode for early stopping.
    validation = phase5.load_split_rows("validation", contract)
    if validation.empty or validation.necrose.astype(int).nunique() != 2:
        raise ValueError("Validation partition must contain both primary classes")
    if set(train.patient_id.astype(str)) & set(validation.patient_id.astype(str)):
        raise ValueError("PID leakage between train and validation")
    validation_features = phase5.load_cached_features(validation, "validation")
    run_training(contract, manifest, config, train, train_features, validation,
                 validation_features, device)


if __name__ == "__main__":
    main()
