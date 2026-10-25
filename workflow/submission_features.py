"""Read exact compact tile means or the original full frozen token arrays."""

from pathlib import Path
import numpy as np


def tile_features(cached):
    if 'tile_features' in cached.files:
        features = cached['tile_features']
        if features.dtype != np.float32:
            raise ValueError('Compact tile features must be float32')
    else:
        tokens = cached['tokens']
        if tokens.ndim != 3 or tokens.shape[1:] != (256, 512):
            raise ValueError(f'Unexpected full token shape: {tokens.shape}')
        features = tokens.astype(np.float32).mean(axis=1)
    if features.ndim != 2 or features.shape[1] != 512 or len(features) < 1:
        raise ValueError(f'Unexpected tile-feature shape: {features.shape}')
    if not np.isfinite(features).all():
        raise ValueError('Non-finite tile features')
    if cached['coords'].shape != (len(features), 2):
        raise ValueError('Tile-coordinate count mismatch')
    return features


def load_cached_features(rows, split, cache_root):
    if split not in {'train', 'validation'}:
        raise ValueError('Training loaders only accept train/validation caches')
    result = []
    for tissue_id in rows.tissue_id.astype(str):
        path = Path(cache_root) / split / f'{tissue_id}.npz'
        with np.load(path, allow_pickle=False) as cached:
            result.append(tile_features(cached))
    return result
