"""Phase 4 item 3: image-only VirTues heads (mean pooling vs attention MIL) on frozen tile tokens.

Predeclared protocol:
- Bag = one core; elements = its valid tiles; tile feature = mean of the 256 patch-summary tokens.
- Frozen encoder (tokens are cached); only the pooling + classifier head is trained.
- Tile-feature standardization uses TRAIN-split tile statistics only.
- Loss: unweighted BCEWithLogitsLoss. Optimizer: Adam(lr=1e-3). Batch: 32 cores. No class weighting.
- Up to 300 epochs; early stopping keeps the epoch with the best VALIDATION PR-AUC (patience 30).
- Seeds 42, 43, 44. Pooling selected by mean validation PR-AUC across seeds.
- Train on Phase 2 train cores; evaluate on Phase 2 validation cores. Test is never read.
"""
import submission_paths as _submission_paths
import submission_features as _submission_features
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score

ROOT = Path(str(_submission_paths.root()))
OUT = _submission_paths.dataset()
TT = _submission_paths.tokens()
PRED = ROOT / 'head_outputs'
SEEDS = [42, 43, 44]
EPOCHS, PATIENCE, BATCH, LR, ATTN_DIM = 300, 30, 32, 1e-3, 128


class MeanPoolHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.classifier = nn.Linear(dim, 1)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)


class AttnMILHead(nn.Module):
    def __init__(self, dim, attn_dim):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, x, mask):
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))  # (B,N,1)
        a = a.masked_fill(~mask.unsqueeze(-1), -1e9)
        a = torch.softmax(a, dim=1)
        pooled = (a * x).sum(1)
        return self.classifier(pooled).squeeze(-1)


def load_split(split):
    tissues = pd.read_parquet(OUT / 'metadata' / 'tissues.parquet')
    sub = tissues[(tissues['split'] == split) & tissues['necrose'].notna()]
    bags, labels, ids = [], [], []
    for r in sub.itertuples():
        z = np.load(TT / split / (r.tissue_id + '.npz'))
        feat = _submission_features.tile_features(z)  # (n_tiles, 512)
        bags.append(feat); labels.append(int(r.necrose)); ids.append(r.tissue_id)
    return bags, np.asarray(labels, dtype=np.float32), ids


def make_batches(bags, labels, batch, shuffle, rng):
    idx = np.arange(len(bags))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch):
        sel = idx[s:s + batch]
        n = max(bags[i].shape[0] for i in sel)
        d = bags[sel[0]].shape[1]
        x = np.zeros((len(sel), n, d), dtype=np.float32)
        m = np.zeros((len(sel), n), dtype=bool)
        for k, i in enumerate(sel):
            x[k, :bags[i].shape[0]] = bags[i]
            m[k, :bags[i].shape[0]] = True
        yield torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(labels[sel])


def evaluate(model, bags, labels, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, m, _ in make_batches(bags, labels, BATCH, False, None):
            probs.append(torch.sigmoid(model(x.to(device), m.to(device))).cpu().numpy())
    p = np.concatenate(probs)
    pr = average_precision_score(labels, p)
    roc = roc_auc_score(labels, p)
    return pr, roc, p


def run(seed, pooling, tr, va, device, mu, sd):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tr_bags = [((f - mu) / sd) for f in tr[0]]
    va_bags = [((f - mu) / sd) for f in va[0]]
    model = (MeanPoolHead(512) if pooling == 'mean' else AttnMILHead(512, ATTN_DIM)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best = {'pr': -1.0, 'epoch': -1, 'pred': None}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, m, y in make_batches(tr_bags, tr[1], BATCH, True, rng):
            opt.zero_grad()
            out = model(x.to(device), m.to(device))
            loss = lossf(out, y.to(device))
            loss.backward(); opt.step()
        pr, roc, pred = evaluate(model, va_bags, va[1], device)
        if pr > best['pr']:
            best = {'pr': pr, 'epoch': epoch, 'pred': pred, 'roc': roc}
        elif epoch - best['epoch'] >= PATIENCE:
            break
    return best


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr = load_split('train'); va = load_split('validation')
    alltiles = np.concatenate(tr[0], axis=0)
    mu = alltiles.mean(0).astype(np.float32)
    sd = (alltiles.std(0) + 1e-6).astype(np.float32)
    PRED.mkdir(exist_ok=True)
    report = {'protocol': {'tile_feature': 'mean of 256 patch-summary tokens', 'feature_standardization': 'train-split tile mean/std',
        'loss': 'BCEWithLogitsLoss (unweighted)', 'optimizer': 'Adam lr=1e-3', 'batch_cores': BATCH,
        'epochs_max': EPOCHS, 'early_stopping_patience': PATIENCE, 'monitor': 'validation PR-AUC',
        'seeds': SEEDS, 'device': device, 'n_train_cores': len(tr[0]), 'n_val_cores': len(va[0]),
        'train_pos': int(tr[1].sum()), 'val_pos': int(va[1].sum()), 'test_used': False}, 'results': {}}
    for pooling in ['mean', 'attn']:
        rows = []
        for seed in SEEDS:
            best = run(seed, pooling, tr, va, device, mu, sd)
            rows.append({'seed': seed, 'val_pr_auc': float(best['pr']), 'val_roc_auc': float(best['roc']), 'best_epoch': int(best['epoch'])})
            np.save(PRED / f'val_pred_{pooling}_seed{seed}.npy', best['pred'])
            print('  %s seed %d val_pr_auc %.4f roc %.4f epoch %d' % (pooling, seed, best['pr'], best['roc'], best['epoch']), flush=True)
        prs = np.array([r['val_pr_auc'] for r in rows])
        rocs = np.array([r['val_roc_auc'] for r in rows])
        report['results'][pooling] = {'per_seed': rows, 'val_pr_auc_mean': float(prs.mean()),
            'val_pr_auc_std': float(prs.std(ddof=0)), 'val_roc_auc_mean': float(rocs.mean()), 'val_roc_auc_std': float(rocs.std(ddof=0))}
    selected = 'mean' if report['results']['mean']['val_pr_auc_mean'] >= report['results']['attn']['val_pr_auc_mean'] else 'attn'
    report['selected_pooling'] = selected
    report['selection_rule'] = 'higher mean validation PR-AUC across seeds 42/43/44'
    # Save val predictions with labels for the selected pooling
    va_ids = va[2]
    for seed in SEEDS:
        p = np.load(PRED / f'val_pred_{selected}_seed{seed}.npy')
        pd.DataFrame({'tissue_id': va_ids, 'necrose': va[1].astype(int), 'prob': p}).to_csv(PRED / f'val_pred_{selected}_seed{seed}.csv', index=False)
    json.dump(report, open(ROOT / 'phase4_head_results.json', 'w'), indent=2)
    print(json.dumps({k: {'pr_mean': report['results'][k]['val_pr_auc_mean'], 'pr_std': report['results'][k]['val_pr_auc_std']} for k in ['mean', 'attn']}, indent=2), flush=True)
    print('selected_pooling', selected, flush=True)
    print('PHASE4_HEADS_OK', flush=True)


if __name__ == '__main__':
    main()
