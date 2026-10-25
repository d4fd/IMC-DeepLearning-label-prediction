"""Train the dropout-0.8 eight-seed head on authorized development tokens.

Use the recipe selected by the runtime grouped search and save every seed state
plus the tile-feature normalizer. Train/validation only; test is never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

import phase5_finetune_mtl as phase5

ROOT = phase5.ROOT
SEEDS = tuple(range(42, 50))
DROPOUT = 0.8
MAX_EPOCHS = 300
PATIENCE = 30
OUTPUT = ROOT / "final_model_08"


def selected_config():
    report = json.loads((ROOT / "phase5_finetune_mtl_results.json").read_text())
    config = dict(report["selected_config"])
    config["dropout"] = DROPOUT
    return config


def fit_epoch(model, optimizer, values, mask, y, aux, config, contract, device, seed, epoch):
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(y))
    losses = []
    for start in range(0, len(order), int(config["batch_size"])):
        ix = torch.as_tensor(order[start:start + int(config["batch_size"])], dtype=torch.long)
        xb, mb, yb = values[ix].to(device), mask[ix].to(device), y[ix].to(device)
        ab = {name: labels[ix].to(device) for name, labels in aux.items()}
        optimizer.zero_grad(set_to_none=True)
        logits, heads = model(xb, mb)
        loss = F.binary_cross_entropy_with_logits(logits.float(), yb)
        for name in contract["retained"]:
            labels = ab[name]
            present = torch.isfinite(labels) if contract["targets"][name]["type"] == "continuous" else labels >= 0
            if int(present.sum()) == 0:
                continue
            raw = (F.huber_loss(heads[name].squeeze(-1)[present].float(), labels[present].float(), delta=1.0)
                   if contract["targets"][name]["type"] == "continuous"
                   else F.cross_entropy(heads[name][present].float(), labels[present].long()))
            loss = loss + float(config["aux_scale"]) * raw
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError('Final-head output is not empty; use a fresh runtime directory')
    contract = json.loads(phase5.CONTRACT_FILE.read_text())
    config = selected_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = phase5.load_train_rows(contract)
    validation = phase5.load_split_rows("validation", contract)
    if set(train.patient_id.astype(str)) & set(validation.patient_id.astype(str)):
        raise ValueError("PID leakage")
    tf = phase5.load_cached_features(train, "train")
    vf = phase5.load_cached_features(validation, "validation")
    mean, std, tiles = phase5.fit_normalizer(tf, np.arange(len(train)))
    age = phase5.age_statistics(train)
    tx, tm = phase5.pack_features(tf, np.arange(len(train)), mean, std)
    ty, taux = phase5.pack_targets(train, phase5.encode_aux(train, contract, age), contract)
    vx, vm = phase5.pack_features(vf, np.arange(len(validation)), mean, std)
    vy, vaux = phase5.pack_targets(validation, phase5.encode_aux(validation, contract, age), contract)

    models, optimizers = [], []
    for seed in SEEDS:
        phase5.seed_everything(seed)
        model = phase5.CoreMTL(contract, DROPOUT).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                               weight_decay=float(config["weight_decay"]))
        models.append(model)
        optimizers.append(opt)

    best = {"mean_pr": -1.0, "epoch": 0, "states": None}
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        prs = []
        for seed, model, opt in zip(SEEDS, models, optimizers):
            fit_epoch(model, opt, tx, tm, ty, taux, config, contract, device, seed, epoch)
            metrics, _, _ = phase5.evaluate(model, vx, vm, vy, vaux, contract, device, int(config["batch_size"]))
            prs.append(metrics["pr_auc"])
        mean_pr = float(np.mean(prs))
        if mean_pr > best["mean_pr"]:
            best = {"mean_pr": mean_pr, "epoch": epoch,
                    "states": [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models]}
        elif epoch - best["epoch"] >= PATIENCE:
            break
        print("EPOCH", epoch, "MEAN_VAL_PR", round(mean_pr, 6), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    y = validation.necrose.astype(int).to_numpy()
    preds = []
    for seed, state in zip(SEEDS, best["states"]):
        torch.save({"state_dict": state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                    "config": config, "seed": seed, "common_best_epoch": best["epoch"]},
                   OUTPUT / f"head08_seed{seed}.pt")
        model = phase5.CoreMTL(contract, DROPOUT).to(device)
        model.load_state_dict(state)
        metrics, probs, _ = phase5.evaluate(model, vx, vm, vy, vaux, contract, device, int(config["batch_size"]))
        preds.append(probs)
    ensemble = np.mean(np.stack(preds), axis=0)
    report = {"dropout": DROPOUT, "seeds": list(SEEDS), "common_best_epoch": best["epoch"],
              "epochs_run": epoch, "mean_validation_pr_auc": best["mean_pr"],
              "ensemble_validation_pr_auc": float(average_precision_score(y, ensemble)),
              "ensemble_validation_roc_auc": float(roc_auc_score(y, ensemble)),
              "normalizer_fit_tiles": tiles, "device": str(device),
              "test_used": False}
    (OUTPUT / "head08_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL_HEAD08_DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
