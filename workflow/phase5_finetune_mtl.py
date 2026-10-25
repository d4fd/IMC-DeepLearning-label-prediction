"""Phase 5 head-only MTL hyperparameter search on frozen VirTues tokens.

Default mode performs a no-training preflight. ``--smoke`` fits one two-epoch
inner fold; ``--execute`` runs the full grouped search and one outer-validation
evaluation. The runner never loads test metadata or tokens.
"""
from __future__ import annotations
import submission_paths as _submission_paths
import submission_features as _submission_features

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

from phase4_prepare_aux_targets import value_of

ROOT = Path(str(_submission_paths.root()))
DATA = _submission_paths.dataset() / 'metadata/tissues.parquet'
SPLIT_FILE = ROOT / "splits/rcc2025necrosis_pid_split.csv"
CONTRACT_FILE = ROOT / "phase4_aux_targets_v1.json"
PHASE4_RESULT = ROOT / "phase4_head_depth_mtl_results.json"
TOKEN_MANIFEST = ROOT / "phase4_tile_tokens_manifest.json"
TOKEN_CACHE = _submission_paths.tokens()
TOKEN_CACHE_METADATA = TOKEN_CACHE / "cache_metadata.json"
MARKER_MANIFEST = ROOT / "phase1_marker_manifest.csv"
CHECKPOINT = ROOT / "checkpoints/virtues-sp32/model.safetensors"
LOCK_FILE = ROOT / "phase5_protocol_lock.json"
RESULT_FILE = ROOT / "phase5_finetune_mtl_results.json"
OUTPUT = ROOT / "outputs/phase5_finetune_mtl"

SELECTED_HEAD = "Linear(512,128)+ReLU -> Linear(128,64)+ReLU"
SEEDS = (42, 43, 44)
FOLDS = 3
FOLD_SEED = 42
SEARCH_SEED = 20260930
INITIAL_TRIALS = 32
REFINEMENT_TOP = 4
REFINEMENT_TRIALS = 16
MAX_EPOCHS = 300
PATIENCE = 30
DROPOUT_LEVELS = (0.0, 0.1, 0.25, 0.4)
BATCH_LEVELS = (16, 32, 64)
AUX_SCALE_LEVELS = (0.25, 0.5, 1.0, 2.0)
POS_WEIGHT_MODES = ("none", "sqrt_neg_pos", "neg_pos")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contracts():
    contract = json.loads(CONTRACT_FILE.read_text())
    phase4 = json.loads(PHASE4_RESULT.read_text())
    manifest = json.loads(TOKEN_MANIFEST.read_text())
    cache_meta = json.loads(TOKEN_CACHE_METADATA.read_text())
    if len(contract["retained"]) != 9:
        raise ValueError("Phase 5 requires the selected nine auxiliary tasks")
    if contract["split_artifact_sha256"] != sha256(SPLIT_FILE):
        raise ValueError("Auxiliary contract split hash mismatch")
    if phase4["comparison"]["selected_shared_head"] != SELECTED_HEAD:
        raise ValueError("Phase 4 selected-head contract mismatch")
    if not np.isfinite(phase4["comparison"]["deeper_mean_pr_auc"]):
        raise ValueError("Phase 4 comparator must be a finite runtime measurement")
    if manifest["split_artifact_sha256"] != sha256(SPLIT_FILE):
        raise ValueError("Token manifest split hash mismatch")
    if manifest["marker_manifest_sha256"] != sha256(MARKER_MANIFEST):
        raise ValueError("Token manifest marker hash mismatch")
    if manifest["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise ValueError("Token manifest checkpoint hash mismatch")
    if manifest["token_shape_per_tile"] != [256, 512]:
        raise ValueError("Unexpected cached token shape")
    for key in ("checkpoint_sha256", "marker_manifest_sha256", "preprocessing_hash", "split_artifact_sha256"):
        if cache_meta[key] != manifest[key if key != "split_artifact_sha256" else "split_artifact_sha256"]:
            raise ValueError(f"Cache metadata mismatch for {key}")
    return contract, manifest, cache_meta


def load_split_rows(split: str, contract: dict) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("Only train/validation metadata may be loaded by Phase 5")
    columns = ["tissue_id", "patient_id", "split", "necrose", *contract["retained"]]
    rows = pd.read_parquet(DATA, columns=columns, filters=[("split", "==", split)])
    if set(rows["split"].astype(str)) != {split}:
        raise ValueError(f"Parquet split filter did not isolate {split}")
    rows = rows[rows.necrose.notna()].reset_index(drop=True)
    if rows.tissue_id.duplicated().any() or rows.patient_id.isna().any():
        raise ValueError(f"Invalid {split} primary-labeled row identifiers")
    return rows


def load_train_rows(contract: dict) -> pd.DataFrame:
    rows = load_split_rows("train", contract)
    if rows.empty or rows.necrose.astype(int).nunique() != 2:
        raise ValueError("Training partition must be nonempty and contain both primary classes")
    return rows


def check_inner_folds(train: pd.DataFrame):
    y = train.necrose.astype(int).to_numpy()
    groups = train.patient_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(splitter.split(np.zeros(len(train)), y, groups))
    seen = np.zeros(len(train), dtype=np.int8)
    for fit_idx, valid_idx in folds:
        if set(groups[fit_idx]) & set(groups[valid_idx]):
            raise ValueError("PID leakage between inner train and validation folds")
        if len(np.unique(y[fit_idx])) != 2 or len(np.unique(y[valid_idx])) != 2:
            raise ValueError("An inner fold lacks a necrose class")
        seen[valid_idx] += 1
    if not np.all(seen == 1):
        raise ValueError("Inner folds do not partition training cores exactly once")
    return folds


def load_cached_features(rows, split):
    return _submission_features.load_cached_features(rows, split, TOKEN_CACHE)


def fit_normalizer(features, indices):
    # Fit over tile features from fold-training cores only, matching Phase 4.
    matrix = np.concatenate([features[int(i)] for i in indices], axis=0).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 512 or not np.isfinite(matrix).all():
        raise ValueError("Invalid tile-feature matrix for fold-local normalization")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32), len(matrix)


def age_statistics(rows: pd.DataFrame):
    values = pd.to_numeric(rows.age, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or float(values.std()) <= 0:
        raise ValueError("Cannot fit fold-local age normalization")
    return {"mean": float(values.mean()), "std": float(values.std())}


def encode_aux(rows: pd.DataFrame, contract: dict, age_stats: dict):
    encoded = {}
    for name in contract["retained"]:
        spec = contract["targets"][name]
        raw = rows[name].tolist()
        if spec["type"] == "continuous":
            values = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
            values = (values - age_stats["mean"]) / age_stats["std"]
            encoded[name] = values.astype(np.float32)
        else:
            vocab = spec["vocabulary"]
            encoded[name] = np.asarray([
                vocab.index(code) if (code := value_of(name, value)) in vocab else -1
                for value in raw
            ], dtype=np.int64)
    return encoded


def pack_features(features, indices, mean, std):
    selected = [features[int(i)] for i in indices]
    max_tiles = max(len(bag) for bag in selected)
    values = np.zeros((len(selected), max_tiles, 512), dtype=np.float32)
    mask = np.zeros((len(selected), max_tiles), dtype=np.bool_)
    for row, bag in enumerate(selected):
        n = len(bag)
        values[row, :n] = (bag - mean) / std
        mask[row, :n] = True
    return torch.from_numpy(values), torch.from_numpy(mask)


def pack_targets(rows, aux, contract):
    y = torch.tensor(rows.necrose.astype(float).to_numpy(), dtype=torch.float32)
    targets = {}
    for name in contract["retained"]:
        dtype = torch.float32 if contract["targets"][name]["type"] == "continuous" else torch.long
        targets[name] = torch.as_tensor(aux[name], dtype=dtype)
    return y, targets


class CoreMTL(nn.Module):
    def __init__(self, contract: dict, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.necrose = nn.Linear(64, 1)
        self.aux = nn.ModuleDict({
            name: nn.Linear(64, 1 if contract["targets"][name]["type"] == "continuous"
                           else len(contract["targets"][name]["vocabulary"]))
            for name in contract["retained"]
        })

    def forward(self, values, mask):
        weights = mask.unsqueeze(-1).to(values.dtype)
        core = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        shared = self.shared(core)
        return self.necrose(shared).squeeze(-1), {name: head(shared) for name, head in self.aux.items()}


def config_key(config):
    normalized = dict(config)
    for name in ("learning_rate", "weight_decay", "aux_scale"):
        normalized[name] = round(float(normalized[name]), 10)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def baseline_config():
    return {"optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0,
            "dropout": 0.0, "batch_size": 32, "aux_scale": 1.0, "pos_weight_mode": "none"}


def sample_initial_configs():
    rng = np.random.default_rng(SEARCH_SEED)
    configs = [baseline_config()]
    keys = {config_key(configs[0])}
    while len(configs) < INITIAL_TRIALS:
        config = {
            "optimizer": str(rng.choice(["Adam", "AdamW"])),
            "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
            "weight_decay": 0.0 if rng.random() < 0.2 else float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-2)))),
            "dropout": float(rng.choice(DROPOUT_LEVELS)),
            "batch_size": int(rng.choice(BATCH_LEVELS)),
            "aux_scale": float(rng.choice(AUX_SCALE_LEVELS)),
            "pos_weight_mode": str(rng.choice(POS_WEIGHT_MODES)),
        }
        key = config_key(config)
        if key not in keys:
            keys.add(key)
            configs.append(config)
    for field, levels in [("optimizer", ["Adam", "AdamW"]), ("dropout", DROPOUT_LEVELS),
                          ("batch_size", BATCH_LEVELS), ("aux_scale", AUX_SCALE_LEVELS),
                          ("pos_weight_mode", POS_WEIGHT_MODES)]:
        if not set(levels) <= {config[field] for config in configs}:
            raise AssertionError(f"Initial search missed a {field} level")
    return configs


def adjacent(values, value):
    values = list(values)
    index = values.index(value)
    return [values[j] for j in (index - 1, index + 1) if 0 <= j < len(values)]


def local_neighbors(config):
    result = []
    for factor in (0.5, 2.0):
        candidate = dict(config)
        candidate["learning_rate"] = float(np.clip(config["learning_rate"] * factor, 1e-4, 3e-3))
        result.append(candidate)
    wd_values = ([1e-5, 1e-3] if config["weight_decay"] == 0 else
                 [float(np.clip(config["weight_decay"] / 10, 1e-6, 1e-2)),
                  float(np.clip(config["weight_decay"] * 10, 1e-6, 1e-2))])
    for value in wd_values:
        candidate = dict(config)
        candidate["weight_decay"] = value
        result.append(candidate)
    for field, levels in [("dropout", DROPOUT_LEVELS), ("batch_size", BATCH_LEVELS),
                          ("aux_scale", AUX_SCALE_LEVELS), ("pos_weight_mode", POS_WEIGHT_MODES)]:
        for value in adjacent(levels, config[field]):
            candidate = dict(config)
            candidate[field] = value
            result.append(candidate)
    candidate = dict(config)
    candidate["optimizer"] = "AdamW" if config["optimizer"] == "Adam" else "Adam"
    result.append(candidate)
    return result


def candidate_rank(record):
    return (-record["mean_pr_auc"], record["std_pr_auc"],
            record["baseline_distance"], record["candidate_id"])


def make_refinement_configs(initial_records):
    ranking = sorted(initial_records, key=candidate_rank)
    top, seen_top = [], set()
    for record in ranking:
        key = config_key(record["config"])
        if key not in seen_top:
            seen_top.add(key)
            top.append(record["config"])
        if len(top) == REFINEMENT_TOP:
            break
    queues = [local_neighbors(config) for config in top]
    chosen, seen = [], {config_key(record["config"]) for record in initial_records}
    depth = 0
    while len(chosen) < REFINEMENT_TRIALS and any(depth < len(queue) for queue in queues):
        for queue in queues:
            if depth >= len(queue):
                continue
            candidate = queue[depth]
            key = config_key(candidate)
            if key not in seen:
                seen.add(key)
                chosen.append(candidate)
                if len(chosen) == REFINEMENT_TRIALS:
                    break
        depth += 1
    if len(chosen) != REFINEMENT_TRIALS:
        raise AssertionError(f"Expected {REFINEMENT_TRIALS} unique refinement configs, got {len(chosen)}")
    return chosen


def baseline_distance(config):
    baseline = baseline_config()
    return sum(config_key({key: config[key]}) != config_key({key: baseline[key]}) for key in baseline)


def choose_record(records):
    best_mean = max(record["mean_pr_auc"] for record in records)
    eligible = [record for record in records if record["mean_pr_auc"] >= best_mean - 0.002]
    best_std = min(record["std_pr_auc"] for record in eligible)
    eligible = [record for record in eligible if record["std_pr_auc"] <= best_std + 1e-12]
    return min(eligible, key=lambda record: (record["baseline_distance"], record["candidate_id"]))


def fold_contexts(train, features, folds, contract):
    contexts = []
    for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
        fit_rows = train.iloc[fit_idx].reset_index(drop=True)
        valid_rows = train.iloc[valid_idx].reset_index(drop=True)
        age_stats = age_statistics(fit_rows)
        fit_aux = encode_aux(fit_rows, contract, age_stats)
        valid_aux = encode_aux(valid_rows, contract, age_stats)
        mean, std, tile_count = fit_normalizer(features, fit_idx)
        x_fit, m_fit = pack_features(features, fit_idx, mean, std)
        x_valid, m_valid = pack_features(features, valid_idx, mean, std)
        y_fit, aux_fit = pack_targets(fit_rows, fit_aux, contract)
        y_valid, aux_valid = pack_targets(valid_rows, valid_aux, contract)
        contexts.append({"fold": fold_number, "fit_rows": fit_rows, "valid_rows": valid_rows,
                         "x_fit": x_fit, "m_fit": m_fit, "y_fit": y_fit, "aux_fit": aux_fit,
                         "x_valid": x_valid, "m_valid": m_valid, "y_valid": y_valid,
                         "aux_valid": aux_valid, "mean": mean, "std": std,
                         "normalizer_fit_tiles": tile_count})
    return contexts


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, values, mask, y, aux, contract, device, batch_size):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))
            logits, _ = model(values[start:end].to(device), mask[start:end].to(device))
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError("Non-finite prediction")
            probabilities.extend(probs.tolist())
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = y.numpy().astype(int)
    if len(np.unique(truth)) != 2:
        raise ValueError("Validation fold must contain both primary classes")
    metrics = {"pr_auc": float(average_precision_score(truth, probabilities)),
               "roc_auc": float(roc_auc_score(truth, probabilities))}
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Non-finite validation metric")
    coverage = {}
    for name, labels in aux.items():
        count = int(torch.isfinite(labels).sum()) if contract["targets"][name]["type"] == "continuous" else int((labels >= 0).sum())
        coverage[name] = {"labeled_cores": count, "total_cores": len(labels)}
    return metrics, probabilities, coverage


def train_cv_config(config, context, contract, device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    seed_everything(seed)
    model = CoreMTL(contract, float(config["dropout"])).to(device)
    optimizer_cls = torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=float(config["learning_rate"]),
                              weight_decay=float(config["weight_decay"]))
    y_fit = context["y_fit"]
    positives = float((y_fit == 1).sum())
    negatives = float((y_fit == 0).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("Training fold must include both necrose classes")
    ratio = negatives / positives
    pos_weight = {"none": 1.0, "sqrt_neg_pos": math.sqrt(ratio), "neg_pos": ratio}[config["pos_weight_mode"]]
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    best = {"pr_auc": -1.0}
    history = []
    parameters = list(model.parameters())
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(y_fit))
        raw_sum = {"necrose": 0.0, **{name: 0.0 for name in contract["retained"]}}
        raw_counts = {name: 0 for name in raw_sum}
        batch_losses = []
        max_grad_norm = 0.0
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start:start + int(config["batch_size"])]
            ix = torch.as_tensor(indices, dtype=torch.long)
            values = context["x_fit"][ix].to(device)
            mask = context["m_fit"][ix].to(device)
            y = context["y_fit"][ix].to(device)
            labels_by_task = {name: task_values[ix].to(device)
                              for name, task_values in context["aux_fit"].items()}
            optimizer.zero_grad(set_to_none=True)
            logits, heads = model(values, mask)
            primary_raw = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight_tensor)
            loss = primary_raw
            raw_sum["necrose"] += float(primary_raw.detach()) * len(indices)
            raw_counts["necrose"] += len(indices)
            for name in contract["retained"]:
                labels = labels_by_task[name]
                kind = contract["targets"][name]["type"]
                present = torch.isfinite(labels) if kind == "continuous" else labels >= 0
                count = int(present.sum())
                if count == 0:
                    continue
                raw = (F.huber_loss(heads[name].squeeze(-1)[present].float(), labels[present].float(), delta=1.0)
                       if kind == "continuous" else F.cross_entropy(heads[name][present].float(), labels[present].long()))
                loss = loss + float(config["aux_scale"]) * raw
                raw_sum[name] += float(raw.detach()) * count
                raw_counts[name] += count
            if not bool(torch.isfinite(loss.detach())):
                raise ValueError("Non-finite training loss")
            loss.backward()
            squared_norm = 0.0
            for parameter in parameters:
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise ValueError("Non-finite gradient")
                    squared_norm += float(parameter.grad.detach().float().pow(2).sum())
            grad_norm = math.sqrt(squared_norm)
            if not math.isfinite(grad_norm):
                raise ValueError("Non-finite gradient norm")
            max_grad_norm = max(max_grad_norm, grad_norm)
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise ValueError("Non-finite trainable parameter")
            batch_losses.append(float(loss.detach()))
        metrics, predictions, coverage = evaluate(
            model, context["x_valid"], context["m_valid"], context["y_valid"],
            context["aux_valid"], contract, device, int(config["batch_size"]))
        record = {"epoch": epoch, "val_pr_auc": metrics["pr_auc"], "val_roc_auc": metrics["roc_auc"],
                  "raw_losses": {name: raw_sum[name] / raw_counts[name] if raw_counts[name] else None for name in raw_sum},
                  "effective_aux_scale": float(config["aux_scale"]), "primary_pos_weight": pos_weight,
                  "training_label_counts": raw_counts, "validation_aux_coverage": coverage,
                  "mean_batch_total_loss": float(np.mean(batch_losses)), "gradient_norm_max": max_grad_norm}
        history.append(record)
        if metrics["pr_auc"] > best["pr_auc"]:
            best = {"pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"], "epoch": epoch,
                    "predictions": predictions.copy(),
                    "record": record, "coverage": coverage}
        elif epoch - best["epoch"] >= patience:
            break
    return {"best_epoch": int(best["epoch"]), "epochs_run": int(epoch),
            "pr_auc": float(best["pr_auc"]), "roc_auc": float(best["roc_auc"]),
            "selected_epoch": best["record"], "validation_aux_coverage": best["coverage"],
            "epoch_history": history}, best


def candidate_record(candidate_id, stage, config, fold_results):
    scores = [fold["pr_auc"] for fold in fold_results]
    return {"candidate_id": candidate_id, "stage": stage, "config": config,
            "fold_results": fold_results, "fold_pr_auc": scores,
            "mean_pr_auc": float(np.mean(scores)), "std_pr_auc": float(np.std(scores)),
            "baseline_distance": baseline_distance(config)}


def baseline_distance(config):
    baseline = baseline_config()
    distance = 0
    for key, value in config.items():
        base = baseline[key]
        if isinstance(value, float) or isinstance(base, float):
            distance += not math.isclose(float(value), float(base), rel_tol=1e-8, abs_tol=1e-10)
        else:
            distance += value != base
    return int(distance)


def save_fold_predictions(candidate_id, context, best):
    directory = OUTPUT / "inner_predictions"
    directory.mkdir(parents=True, exist_ok=True)
    rows = context["valid_rows"]
    pd.DataFrame({"tissue_id": rows.tissue_id.astype(str).to_numpy(),
                  "necrose": rows.necrose.astype(int).to_numpy(),
                  "probability": best["predictions"]}).to_csv(
        directory / f"{candidate_id}_fold{context['fold']}.csv", index=False)


def run_candidate(candidate_id, stage, config, contexts, contract, device):
    results = []
    for context in contexts:
        seed = SEARCH_SEED + context["fold"]
        record, best = train_cv_config(config, context, contract, device, seed)
        save_fold_predictions(candidate_id, context, best)
        history_dir = OUTPUT / "epoch_logs"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / f"{candidate_id}_fold{context['fold']}.json").write_text(
            json.dumps(record["epoch_history"], indent=2, allow_nan=False) + "\n")
        fold = {"fold": context["fold"], "fit_pids": int(context["fit_rows"].patient_id.nunique()),
                "validation_pids": int(context["valid_rows"].patient_id.nunique()),
                "fit_cores": len(context["fit_rows"]), "validation_cores": len(context["valid_rows"]),
                "normalizer_fit_tiles": context["normalizer_fit_tiles"],
                "normalizer_mean_sha256": hashlib.sha256(context["mean"].tobytes()).hexdigest(),
                "normalizer_std_sha256": hashlib.sha256(context["std"].tobytes()).hexdigest(),
                **{key: value for key, value in record.items() if key != "epoch_history"}}
        results.append(fold)
        del best, record
    return candidate_record(candidate_id, stage, config, results)


def choose_record(records):
    best_mean = max(record["mean_pr_auc"] for record in records)
    eligible = [record for record in records if record["mean_pr_auc"] >= best_mean - 0.002]
    best_std = min(record["std_pr_auc"] for record in eligible)
    eligible = [record for record in eligible if record["std_pr_auc"] <= best_std + 1e-12]
    return min(eligible, key=lambda record: (record["baseline_distance"], record["candidate_id"]))


def save_progress(records, refinement_records):
    data = {"completed_configurations": len(records), "initial_trials": INITIAL_TRIALS,
            "refinement_trials": REFINEMENT_TRIALS, "initial_records": records[:INITIAL_TRIALS],
            "refinement_records": refinement_records, "test_used": False}
    (OUTPUT / "phase5_inner_search_progress.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_search(train, features, folds, contract, device):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contexts = fold_contexts(train, features, folds, contract)
    all_records = []
    for index, config in enumerate(sample_initial_configs()):
        candidate_id = f"initial_{index:03d}"
        record = run_candidate(candidate_id, "initial", config, contexts, contract, device)
        all_records.append(record)
        save_progress(all_records, [])
        print("INITIAL", candidate_id, round(record["mean_pr_auc"], 6), round(record["std_pr_auc"], 6), flush=True)
    refinement_records = []
    refinement_configs = make_refinement_configs(all_records)
    for index, config in enumerate(refinement_configs):
        candidate_id = f"refine_{index:03d}"
        record = run_candidate(candidate_id, "refinement", config, contexts, contract, device)
        all_records.append(record)
        refinement_records.append(record)
        save_progress(all_records, refinement_records)
        print("REFINEMENT", candidate_id, round(record["mean_pr_auc"], 6), round(record["std_pr_auc"], 6), flush=True)
    selected = choose_record(all_records)
    epochs = int(np.median([fold["best_epoch"] for fold in selected["fold_results"]]))
    selection = {"selected_candidate_id": selected["candidate_id"], "selected_stage": selected["stage"],
                 "selected_config": selected["config"], "inner_fold_pr_auc": selected["fold_pr_auc"],
                 "mean_inner_pr_auc": selected["mean_pr_auc"], "std_inner_pr_auc": selected["std_pr_auc"],
                 "baseline_distance": selected["baseline_distance"],
                 "inner_best_epochs": [fold["best_epoch"] for fold in selected["fold_results"]],
                 "final_epochs": epochs,
                 "tie_break_rule": "within 0.002 mean PR-AUC: lowest fold SD, then fewest changed baseline parameters, then candidate ID",
                 "test_used": False}
    selected_path = OUTPUT / "phase5_selected_config.json"
    selected_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    audit = {"search_seed": SEARCH_SEED, "fold_seed": FOLD_SEED,
             "initial_trials": INITIAL_TRIALS, "refinement_top": REFINEMENT_TOP,
             "refinement_trials": REFINEMENT_TRIALS,
             "initial_records": all_records[:INITIAL_TRIALS],
             "refinement_records": refinement_records, "selection": selection, "test_used": False}
    (OUTPUT / "phase5_inner_search_results.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return selection


def threshold_metrics(y, probabilities):
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    j_values = tpr - fpr
    finite = np.flatnonzero(np.isfinite(thresholds))
    if len(finite) == 0:
        raise ValueError("No finite Youden-J threshold")
    index = int(finite[np.argmax(j_values[finite])])
    threshold = float(thresholds[index])
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"threshold": threshold, "youden_j": float(j_values[index]),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "accuracy": float(accuracy_score(y, predicted)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def fit_final_model(config, train, train_features, train_aux, contract, mean, std,
                    seed, epochs, device):
    seed_everything(seed)
    indices = np.arange(len(train))
    values, mask = pack_features(train_features, indices, mean, std)
    y, aux = pack_targets(train, train_aux, contract)
    model = CoreMTL(contract, float(config["dropout"])).to(device)
    optimizer_cls = torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=float(config["learning_rate"]),
                              weight_decay=float(config["weight_decay"]))
    positives, negatives = float((y == 1).sum()), float((y == 0).sum())
    ratio = negatives / positives
    pos_weight = {"none": 1.0, "sqrt_neg_pos": math.sqrt(ratio), "neg_pos": ratio}[config["pos_weight_mode"]]
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        for start in range(0, len(order), int(config["batch_size"])):
            ix = torch.as_tensor(order[start:start + int(config["batch_size"])], dtype=torch.long)
            xb, mb, yb = values[ix].to(device), mask[ix].to(device), y[ix].to(device)
            batch_aux = {name: labels[ix].to(device) for name, labels in aux.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, heads = model(xb, mb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), yb, pos_weight=pos_weight_tensor)
            for name in contract["retained"]:
                labels = batch_aux[name]
                kind = contract["targets"][name]["type"]
                present = torch.isfinite(labels) if kind == "continuous" else labels >= 0
                if int(present.sum()) == 0:
                    continue
                raw = (F.huber_loss(heads[name].squeeze(-1)[present].float(), labels[present].float(), delta=1.0)
                       if kind == "continuous" else F.cross_entropy(heads[name][present].float(), labels[present].long()))
                loss = loss + float(config["aux_scale"]) * raw
            if not bool(torch.isfinite(loss.detach())):
                raise ValueError("Non-finite final training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError("Non-finite final gradient")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ValueError("Non-finite final parameter")
    return model


def execute(contract, manifest, train, train_features, folds, device):
    selection = run_search(train, train_features, folds, contract, device)
    # The selected recipe is persisted before any outer-validation row/token is loaded.
    selected_path = OUTPUT / "phase5_selected_config.json"
    disk_selection = json.loads(selected_path.read_text())
    if disk_selection["selected_config"] != selection["selected_config"]:
        raise ValueError("Selected recipe lock changed before outer validation")
    validation = load_split_rows("validation", contract)
    if validation.empty or validation.necrose.astype(int).nunique() != 2:
        raise ValueError("Validation partition must contain both primary classes")
    if set(train.patient_id.astype(str)) & set(validation.patient_id.astype(str)):
        raise ValueError("PID leakage between training and outer validation")
    validation_features = load_cached_features(validation, "validation")
    age_stats = age_statistics(train)
    train_aux = encode_aux(train, contract, age_stats)
    validation_aux = encode_aux(validation, contract, age_stats)
    mean, std, normalizer_tiles = fit_normalizer(train_features, np.arange(len(train)))
    val_x, val_mask = pack_features(validation_features, np.arange(len(validation)), mean, std)
    _, val_aux = pack_targets(validation, validation_aux, contract)
    config, epochs = selection["selected_config"], selection["final_epochs"]
    seed_predictions, seed_records = [], []
    for seed in SEEDS:
        model = fit_final_model(config, train, train_features, train_aux, contract,
                                mean, std, seed, epochs, device)
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(validation), int(config["batch_size"])):
                end = min(start + int(config["batch_size"]), len(validation))
                logits, _ = model(val_x[start:end].to(device), val_mask[start:end].to(device))
                probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite outer validation probability")
        y = validation.necrose.astype(int).to_numpy()
        metrics = {"pr_auc": float(average_precision_score(y, probabilities)),
                   "roc_auc": float(roc_auc_score(y, probabilities)),
                   "brier_score": float(brier_score_loss(y, probabilities))}
        seed_predictions.append(probabilities)
        seed_records.append({"seed": seed, "epochs": epochs, "metrics": metrics,
                             "finite_checks_passed": True})
        torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "mean": torch.from_numpy(mean),
                    "std": torch.from_numpy(std), "config": config, "seed": seed,
                    "epochs": epochs, "contract_sha256": sha256(CONTRACT_FILE)},
                   OUTPUT / f"phase5_head_seed{seed}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble = np.mean(np.stack(seed_predictions), axis=0)
    y = validation.necrose.astype(int).to_numpy()
    ensemble_metrics = {"pr_auc": float(average_precision_score(y, ensemble)),
                        "roc_auc": float(roc_auc_score(y, ensemble)),
                        "brier_score": float(brier_score_loss(y, ensemble)),
                        **threshold_metrics(y, ensemble)}
    outer_aux_coverage = {}
    for name, labels in val_aux.items():
        count = int(torch.isfinite(labels).sum()) if contract["targets"][name]["type"] == "continuous" else int((labels >= 0).sum())
        outer_aux_coverage[name] = {"labeled_cores": count, "total_cores": len(labels)}
    numeric_metrics = [value for value in ensemble_metrics.values() if isinstance(value, (int, float))]
    if not np.isfinite(numeric_metrics).all():
        raise ValueError("Non-finite outer-validation ensemble metric")
    pd.DataFrame({"tissue_id": validation.tissue_id.astype(str), "necrose": y,
                  **{f"probability_seed{seed}": probs for seed, probs in zip(SEEDS, seed_predictions)},
                  "probability_ensemble": ensemble}).to_csv(OUTPUT / "phase5_outer_validation_predictions.csv", index=False)
    report = {"protocol_version": 2, "selected_head": SELECTED_HEAD, "encoder_frozen": True,
              "selected_candidate_id": selection["selected_candidate_id"], "selected_config": config,
              "inner_selection": {"mean_pr_auc": selection["mean_inner_pr_auc"],
                                  "fold_pr_auc": selection["inner_fold_pr_auc"],
                                  "std_pr_auc": selection["std_inner_pr_auc"], "final_epochs": epochs},
              "outer_seed_results": seed_records, "outer_validation_ensemble": ensemble_metrics,
              "outer_validation_aux_coverage": outer_aux_coverage,
               "frozen_phase4_comparator_pr_auc": json.loads(PHASE4_RESULT.read_text())["comparison"]["deeper_mean_pr_auc"],
               "delta_over_phase4_comparator": ensemble_metrics["pr_auc"] - json.loads(PHASE4_RESULT.read_text())["comparison"]["deeper_mean_pr_auc"],
              "normalizer_fit_tiles": normalizer_tiles, "normalizer_source": "verified train token cache only",
              "aux_manifest_sha256": sha256(CONTRACT_FILE),
              "marker_manifest_sha256": manifest["marker_manifest_sha256"],
              "preprocessing_hash": manifest["preprocessing_hash"],
              "split_sha256": manifest["split_artifact_sha256"],
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "vendor_commit": manifest["vendor_commit"],
              "partitions_used": ["train", "validation"], "test_used": False}
    RESULT_FILE.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PHASE5_HEAD_TUNING_COMPLETE", selection["selected_candidate_id"],
          selection["mean_inner_pr_auc"], ensemble_metrics["pr_auc"], flush=True)


def make_lock(contract, manifest, cache_meta):
    vendor = subprocess.check_output(["git", "-C", str(ROOT / "virtues"), "rev-parse", "HEAD"], text=True).strip()
    return {"protocol_version": 2, "status": "predeclared",
            "model": {"encoder": "frozen virtues-sp32; verified cached train/validation tokens",
                      "pooling": "mean patch-summary features per tile, then mean tiles per core",
                      "selected_head": SELECTED_HEAD, "auxiliaries": contract["retained"],
                      "all_auxiliary_losses_active": True},
            "search": {"initial_trials": INITIAL_TRIALS, "search_seed": SEARCH_SEED,
                       "refinement_top_regions": REFINEMENT_TOP, "refinement_trials": REFINEMENT_TRIALS,
                       "refinement": "16 round-robin local neighbors of top initial recipes; LR x0.5/x2 and weight-decay /10/x10 (or 1e-5/1e-3 when baseline zero), then categorical neighbors if needed to ensure unique configs",
                       "optimizer": ["Adam", "AdamW"], "learning_rate_log_uniform": [1e-4, 3e-3],
                       "weight_decay": {"zero_allowed": True, "log_uniform_nonzero": [1e-6, 1e-2]},
                       "dropout": list(DROPOUT_LEVELS), "batch_cores": list(BATCH_LEVELS),
                       "global_auxiliary_loss_multiplier": list(AUX_SCALE_LEVELS),
                       "primary_pos_weight": list(POS_WEIGHT_MODES), "max_epochs": MAX_EPOCHS,
                       "early_stopping_patience": PATIENCE,
                       "selection_metric": "mean primary PR-AUC across three inner folds",
                       "tie_break": "within 0.002 of best mean PR-AUC, then lowest fold SD, nearest baseline, candidate ID"},
            "inner_cv": {"method": "StratifiedGroupKFold", "folds": FOLDS, "seed": FOLD_SEED,
                         "group": "patient_id (anonymized PID group key)", "partition": "training PIDs only",
                         "normalization": "current fold-training cores only"},
            "final_fit": {"seeds": list(SEEDS), "epochs": "median selected-recipe best epoch across folds",
                          "outer_validation": "read after recipe selection; no early stopping or recipe changes"},
            "threshold": "Youden J from locked three-seed ensemble validation predictions",
            "hashes": {"aux_manifest_sha256": sha256(CONTRACT_FILE),
                       "marker_manifest_sha256": sha256(MARKER_MANIFEST),
                       "token_manifest_sha256": sha256(TOKEN_MANIFEST),
                       "tile_cache_metadata_sha256": sha256(TOKEN_CACHE_METADATA),
                       "preprocessing_hash": manifest["preprocessing_hash"],
                       "split_sha256": sha256(SPLIT_FILE), "checkpoint_sha256": manifest["checkpoint_sha256"],
                       "vendor_commit": vendor},
            "partitions_permitted": ["train", "validation"], "test_used": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run full frozen-token grouped-CV tuning")
    parser.add_argument("--smoke", action="store_true", help="Fit one two-epoch inner fold; never reads outer validation/test")
    args = parser.parse_args()
    contract, manifest, cache_meta = load_contracts()
    train = load_train_rows(contract)
    folds = check_inner_folds(train)
    train_features = load_cached_features(train, "train")
    lock = make_lock(contract, manifest, cache_meta)
    LOCK_FILE.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    print("PHASE5_PREFLIGHT_OK", json.dumps({"train_cores": len(train),
          "train_pids": int(train.patient_id.nunique()), "fold_sizes": [len(valid) for _, valid in folds],
          "fold_pids_disjoint": True, "train_token_cores_verified": len(train_features),
          "auxiliaries": len(contract["retained"]), "test_used": False,
          "execute": args.execute, "smoke": args.smoke}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        context = fold_contexts(train, train_features, folds[:1], contract)[0]
        result, _ = train_cv_config(baseline_config(), context, contract, device,
                                    SEARCH_SEED + 1, max_epochs=2, patience=2)
        print("PHASE5_TRAINING_ONLY_SMOKE_OK", json.dumps({"device": str(device),
              "epochs": result["epochs_run"], "pr_auc": result["pr_auc"], "test_used": False}), flush=True)
        return
    if not args.execute:
        print("No weights trained. Use --smoke for a training-fold check or --execute for the approved search.", flush=True)
        return
    execute(contract, manifest, train, train_features, folds, device)


if __name__ == "__main__":
    main()
