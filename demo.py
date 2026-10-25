"""Small, CPU-only synthetic check of the exported model; no files are written."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'workflow'))

import numpy as np
import torch

from phase5_finetune_mtl import CoreMTL, fit_normalizer, pack_features
from phase6_train_final import fit_epoch


def synthetic_contract():
    # Invented task names and classes, not the study's target contract or labels.
    targets = {'toy_continuous': {'type': 'continuous'}}
    targets.update({f'toy_category_{i}': {'type': 'categorical', 'vocabulary': ['class_a', 'class_b']}
                    for i in range(8)})
    return {'retained': list(targets), 'targets': targets}


def run_demo():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    bags = [rng.normal(size=(tiles, 512)).astype(np.float32) for tiles in (2, 3, 1, 4, 2, 1)]
    # Only the four artificial fitting cores contribute to normalization.
    mean, std, count = fit_normalizer(bags, np.arange(4))
    values, mask = pack_features(bags, np.arange(len(bags)), mean, std)
    contract = synthetic_contract()
    model = CoreMTL(contract, dropout=0.8).cpu()
    y = torch.tensor([0., 1., 0., 1., 0., 1.])
    aux = {'toy_continuous': torch.tensor([0., 1., float('nan'), -1., .5, .2])}
    aux.update({name: torch.tensor([0, 1, -1, 0, 1, 0]) for name in contract['retained'][1:]})
    # One tiny update exercises the ACTUAL exported final trainer and its missing-label masks.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = fit_epoch(model, optimizer, values[:4], mask[:4], y[:4],
                     {k: v[:4] for k, v in aux.items()},
                     {'batch_size': 4, 'aux_scale': .25}, contract, torch.device('cpu'), 7, 1)
    if not np.isfinite(loss):
        raise AssertionError('Synthetic loss is non-finite')
    model.eval()
    with torch.inference_mode():
        logits, heads = model(values, mask)
        padded = values.clone()
        padded[~mask] = 1e6
        padded_logits, _ = model(padded, mask)
    if logits.shape != (6,) or len(heads) != 9 or not torch.isfinite(logits).all():
        raise AssertionError('Synthetic model contract failed')
    torch.testing.assert_close(logits, padded_logits)
    return {'cores': len(bags), 'normalizer_fit_tiles': count, 'auxiliary_heads': len(heads), 'device': 'cpu'}


if __name__ == '__main__':
    info = run_demo()
    print('SYNTHETIC_DEMO_OK:', info)
    print('Generated tensors only; no study inputs, saved weights, metrics or files used.')
