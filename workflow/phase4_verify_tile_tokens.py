"""Phase 4 item 2 verification: independently check the cached tile tokens against the dataset tiling."""
import submission_paths as _submission_paths
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(str(_submission_paths.root()))
OUT = _submission_paths.dataset()
TT = _submission_paths.tokens()


def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    tissues = pd.read_parquet(OUT / 'metadata' / 'tissues.parquet')
    coords = pd.read_parquet(OUT / 'tiling/2_678571mpp/default/128_tile_coordinates.parquet')
    expected_tiles = coords.groupby('tissue_id').size().to_dict()
    report = {'splits': {}, 'token_shape_per_tile': None}
    for sp in ['train', 'validation']:
        files = sorted((TT / sp).glob('*.npz'))
        cores = len(files); tiles = 0; bad = []
        for f in files:
            tid = f.stem
            z = np.load(f)
            t = z['tokens']; c = z['coords']
            if t.dtype != np.float16 or t.shape[1:] != (256, 512) or c.shape != (t.shape[0], 2):
                bad.append((tid, str(t.dtype), list(t.shape), list(c.shape))); continue
            if not np.isfinite(t).all():
                bad.append((tid, 'nonfinite')); continue
            if t.shape[0] != expected_tiles.get(tid, -1):
                bad.append((tid, 'tile_count', int(t.shape[0]), expected_tiles.get(tid))); continue
            tiles += int(t.shape[0])
            report['token_shape_per_tile'] = list(t.shape[1:])
        report['splits'][sp] = {'cores': cores, 'tiles': tiles, 'bad': bad}
    report['n_files_total'] = sum(v['cores'] for v in report['splits'].values())
    report['n_tiles_total'] = sum(v['tiles'] for v in report['splits'].values())
    report['cache_metadata_sha256'] = sha256(TT / 'cache_metadata.json')
    json.dump(report, open(ROOT / 'phase4_tile_tokens_verification.json', 'w'), indent=2)
    print(json.dumps(report, indent=2), flush=True)
    assert all(not v['bad'] for v in report['splits'].values()), 'cache verification failed'
    print('PHASE4_TILE_TOKENS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
