"""Build the project-owned Spora-format dataset for RCC necrosis (Phase 2).

Reads separately authorized external inputs. Writes only to the configured runtime directory.
Resolution 2.678571 um/px (600/224), modality imc, release-compatible 128px default tiling.
Source-pinned standardizer + tiler from spora_io. Two-pass-free (single image pass).
User-approved decisions: dataset name rcc2025necrosis; 70/15/15 PID split seed 42; auxiliaries in
tissues.parquet; TMA kept only in the split audit (not in the dataset).
"""
import submission_paths as _submission_paths
import os, json, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile, zarr
from zarr.codecs import BloscCodec
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from sklearn.model_selection import StratifiedShuffleSplit
from spora_io.utils.helpers.tile import best_mask_tiling_try_to_stop

SRC = Path(str(_submission_paths.source()))
ROOT = Path(str(_submission_paths.root()))
OUT = Path(os.environ.get('PHASE2_OUT', str(_submission_paths.dataset())))
MANIFEST = ROOT / 'phase1_marker_manifest.csv'
DATASET = 'rcc2025necrosis'
MODALITY = 'imc'
RES_STR = '2.678571'
MPP = 600.0 / 224.0
RES_DIR = RES_STR.replace('.', '_') + 'mpp'
TILE, STRIDE, TOLERANCE, COVERAGE, MIN_GAIN = 128, 64, 0.95, 0.95, 0.1
# Pseudonyms are not anonymization. Keep the secret and every generated map private.
SALT = os.environ.get('PROJECT_PID_SALT')
SEED = 42
LIMIT = int(os.environ.get('PHASE2_LIMIT', '0'))
SMOKE = os.environ.get('PHASE2_SMOKE', '0') == '1'
AUX = ['gender', 'age', 'subtype_combined', 'lymphocytes_combined', 'pT_combined',
       'ISUP_combined', 'PDL1_score', 'genomic_subtype', 'genomic_risk', 'metabolic_subtype']


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def phash(pid):
    if not SALT:
        raise ValueError('Set a private PROJECT_PID_SALT before generating patient pseudonyms')
    return hashlib.sha256((SALT + str(pid)).encode()).hexdigest()[:8]


def write_ome(img, path, name):
    if path.exists():
        shutil.rmtree(path)
    g = zarr.open_group(str(path), mode='w', zarr_format=3)
    a = g.create_array('0', shape=img.shape, chunks=(int(img.shape[0]), 224, 224), dtype='float32',
                       compressors=[BloscCodec(cname='lz4', clevel=3, shuffle='bitshuffle')], fill_value=0.0)
    a[:] = img
    g.attrs['ome'] = {'version': '0.5', 'multiscales': [{'name': name,
        'axes': [{'name': 'c', 'type': 'channel'},
                 {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                 {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
        'datasets': [{'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, MPP, MPP]}]}]}]}


def make_mask(img):
    total = np.clip(img, 0, None).sum(axis=0)
    thr = (np.percentile(total, 5) + np.percentile(total, 60)) / 2.0
    m = total > thr
    m = remove_small_objects(m, min_size=500)
    if not m.any():
        return np.ones(total.shape, dtype=bool)
    return binary_fill_holes(m).astype(bool)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Dataset output is not empty; choose a fresh runtime dataset directory')
    panel = pd.read_csv(SRC / 'panel.csv')
    man = pd.read_csv(MANIFEST)
    assert len(panel) == len(man) == 46
    assert panel['channel'].tolist() == man['source_channel'].tolist()
    names = man['hugo_name'].tolist()
    assert len(set(names)) == 46
    channels = pd.DataFrame({
        'channel_name': names,
        'qc_pass': man['disposition'].eq('use'),
        'uniprot_id': man['uniprot_id'].fillna('').astype(str),
        'is_nuclear_marker': man['nuclear_marker'].fillna('').eq('yes'),
        'description': man['notes'].fillna('').astype(str),
        'original_channel_name': man['original_name'].astype(str),
    })
    expected_uniprots = man.loc[man['disposition'].eq('use'), 'uniprot_id'].tolist()

    md = pd.read_csv(SRC / 'metadata.csv')
    md = md[md['PID'].notna()].copy().reset_index(drop=True)
    md['patient_id'] = md['PID'].map(phash)
    md = md.sort_values(['PID', 'sample_id']).reset_index(drop=True)
    md['specimen_id'] = md.groupby('PID').cumcount().map(lambda i: '%04d' % i)
    md['tissue_id'] = DATASET + '_' + md['patient_id'] + '_' + md['specimen_id']
    assert md['tissue_id'].is_unique
    for s in md['sample_id']:
        assert (_submission_paths.images() / (s + '.tiff')).exists(), s

    pid_label = md.groupby('patient_id')['necrose'].apply(lambda s: int((s == 1).any()))
    pids = sorted(md['patient_id'].unique())
    y = np.array([int(pid_label[p]) for p in pids]); X = np.arange(len(pids))
    tr, rest = next(StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(X, y))
    vrel, trel = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(X[rest], y[rest]))
    split = {}
    for i in tr: split[pids[i]] = 'train'
    for i in rest[vrel]: split[pids[i]] = 'validation'
    for i in rest[trel]: split[pids[i]] = 'test'
    md['split'] = md['patient_id'].map(split)
    assert md['split'].notna().all()

    tissues = pd.DataFrame({
        'tissue_id': md['tissue_id'], 'dataset': DATASET, 'patient_id': md['patient_id'],
        'specimen_id': md['specimen_id'], 'modality': MODALITY, 'alignment': np.nan,
        'split': md['split'], 'necrose': md['necrose'], 'original_file_name': md['sample_id'],
    })
    for c in AUX:
        tissues[c] = md[c].values
    tissues = tissues.sort_values('tissue_id').reset_index(drop=True)
    if LIMIT:
        tissues = tissues.head(LIMIT).copy()
    n, C = len(tissues), 46

    img_dir = OUT / MODALITY / RES_DIR / 'images'
    mask_dir = OUT / 'segmentations' / RES_DIR / 'tissue_masks'
    std_dir = OUT / MODALITY / RES_DIR / 'standardization' / 'quantile_clipping_log1p' / 'uq_0.99_image'
    tile_dir = OUT / 'tiling' / RES_DIR / 'default'
    for d in [OUT / 'metadata', OUT / MODALITY, img_dir, mask_dir, std_dir, tile_dir]:
        d.mkdir(parents=True, exist_ok=True)
    channels.to_parquet(OUT / MODALITY / 'channels.parquet', index=False)
    cpt = pd.DataFrame(True, index=tissues['tissue_id'], columns=names); cpt.index.name = 'tissue_id'
    cpt.to_parquet(OUT / MODALITY / 'channels_per_tissue.parquet')
    tissues.to_parquet(OUT / 'metadata' / 'tissues.parquet', index=False)
    print('tables written', n, 'tissues', flush=True)

    upper = np.full((n, C), np.nan); sums = np.zeros((n, C)); sq = np.zeros((n, C)); cnt = np.zeros(n)
    for k, r in enumerate(tissues.itertuples()):
        img = tifffile.imread(_submission_paths.images() / (r.original_file_name + '.tiff')).astype(np.float32)
        assert img.shape == (C, 224, 224), (r.tissue_id, img.shape)
        write_ome(img, img_dir / (r.tissue_id + '.ome.zarr'), r.tissue_id)
        m = make_mask(img)
        np.savez_compressed(mask_dir / (r.tissue_id + '.npz'), mask=m)
        if m.any():
            fg = img[:, m]
            up = np.quantile(fg, 0.99, axis=1).astype(np.float64)
            upper[k] = up
            lg = np.log1p(np.clip(img, 0.0, up[:, None, None].astype(np.float32)))
            fg2 = lg[:, m].astype(np.float64)
            sums[k] = fg2.sum(axis=1); sq[k] = (fg2 ** 2).sum(axis=1); cnt[k] = m.sum()
        if (k + 1) % 50 == 0 or SMOKE:
            print('processed %d/%d mask_frac=%.4f' % (k + 1, n, m.mean()), flush=True)
    print('image pass done', flush=True)

    upper_df = pd.DataFrame(upper, index=tissues['tissue_id'], columns=names); upper_df.index.name = 'tissue_id'
    upper_df.to_parquet(std_dir / 'image_level_upper_quantiles.parquet')
    trm = tissues['split'].eq('train').values
    tot = sums[trm].sum(axis=0); tot2 = sq[trm].sum(axis=0); c = cnt[trm].sum()
    mean = np.where(c > 0, tot / c, np.nan)
    std = np.sqrt(np.clip(np.where(c > 0, tot2 / c - mean ** 2, np.nan), 0, None))
    pd.DataFrame([mean], columns=names).to_parquet(std_dir / 'global_level_means.parquet')
    pd.DataFrame([std], columns=names).to_parquet(std_dir / 'global_level_stds.parquet')
    print('standardization written (global fit on %d train tissues)' % int(trm.sum()), flush=True)

    coords = []; stats = []; gidx = 0
    for tid in tissues['tissue_id']:
        m = np.load(mask_dir / (tid + '.npz'))['mask']
        tiles, st, covered = best_mask_tiling_try_to_stop(m, tile_size=TILE, stride=STRIDE,
            tolerance=TOLERANCE, coverage_goal=COVERAGE, min_gain_ratio=MIN_GAIN)
        for j, tl in enumerate(tiles):
            coords.append({'index': gidx, 'tissue_id': tid, 'tile_id': j, 'row': int(tl.y), 'col': int(tl.x)})
            gidx += 1
        stats.append({'tissue_id': tid, 'mask_height': int(m.shape[0]), 'mask_width': int(m.shape[1]),
            'tissue_mask_frac': float(m.mean()), 'num_tiles': int(st.get('num_tiles', 0)),
            'coverage_ratio': float(st.get('coverage_ratio', 0.0)),
            'covered_valid_pixels': int(st.get('covered_valid_pixels', 0)),
            'total_valid_pixels': int(st.get('total_valid_pixels', 0)),
            'candidate_count': int(st.get('candidate_count', 0)), 'stop_reason': str(st.get('stop_reason', '')),
            'tile_size': TILE, 'stride': STRIDE, 'tolerance': TOLERANCE,
            'coverage_goal': COVERAGE, 'min_gain_ratio': MIN_GAIN})
    pd.DataFrame(coords, columns=['index', 'tissue_id', 'tile_id', 'row', 'col']).to_parquet(
        tile_dir / ('%d_tile_coordinates.parquet' % TILE), index=False)
    st_df = pd.DataFrame(stats).set_index('tissue_id'); st_df.index.name = 'tissue_id'
    st_df.to_parquet(tile_dir / ('%d_tile_stats.parquet' % TILE))
    print('tiling written: tiles=%d zero_tile_tissues=%d' % (len(coords), int((st_df['num_tiles'] == 0).sum())), flush=True)

    if not LIMIT:
        priv = ROOT / '.private'; priv.mkdir(exist_ok=True)
        md[['patient_id', 'PID']].drop_duplicates().to_csv(priv / 'rcc2025necrosis_pid_map.csv', index=False)
        sp = ROOT / 'splits'; sp.mkdir(exist_ok=True)
        pdf = pd.DataFrame({'patient_id': pids, 'split': [split[p] for p in pids],
                            'pid_label': [int(pid_label[p]) for p in pids]})
        pdf['n_images'] = pdf['patient_id'].map(md.groupby('patient_id').size())
        pdf.to_csv(sp / 'rcc2025necrosis_pid_split.csv', index=False)
        audit = {'seed': SEED, 'stratification': 'pid_label = any image necrose==1', 'splits': {}}
        for s in ['train', 'validation', 'test']:
            sub = md[md['split'] == s]
            audit['splits'][s] = {'n_pids': int(sub['patient_id'].nunique()), 'n_images': int(len(sub)),
                'necrose_pos': int((sub['necrose'] == 1).sum()), 'necrose_neg': int((sub['necrose'] == 0).sum()),
                'necrose_missing': int(sub['necrose'].isna().sum()),
                'tma_counts': {str(k): int(v) for k, v in sub['TMA'].value_counts().items()}}
        json.dump(audit, open(sp / 'rcc2025necrosis_split_audit.json', 'w'), indent=2)
        print('split written', json.dumps({s: audit['splits'][s]['n_pids'] for s in audit['splits']}), flush=True)

        mf = {'dataset': DATASET, 'modality': MODALITY, 'resolution': RES_STR, 'resolution_dir': RES_DIR,
              'mpp': MPP, 'n_tissues': n, 'n_pids': int(tissues['patient_id'].nunique()),
              'channel_names': names, 'expected_uniprots': expected_uniprots,
              'tile_size': TILE, 'stride': STRIDE, 'tolerance': TOLERANCE, 'coverage_goal': COVERAGE,
              'min_gain_ratio': MIN_GAIN,
              'mask_recipe': 'thr=(p5+p60)/2 of sum_channels; remove_small_objects(min_size=500); binary_fill_holes',
              'salt_sha256': hashlib.sha256(SALT.encode()).hexdigest(), 'filter_config': ['gaussian_blur'],
              'standardization': 'quantile_clipping_log1p/uq_0.99_image', 'files': {}}
        rels = ['metadata/tissues.parquet', 'imc/channels.parquet', 'imc/channels_per_tissue.parquet',
                'imc/%s/standardization/quantile_clipping_log1p/uq_0.99_image/image_level_upper_quantiles.parquet' % RES_DIR,
                'imc/%s/standardization/quantile_clipping_log1p/uq_0.99_image/global_level_means.parquet' % RES_DIR,
                'imc/%s/standardization/quantile_clipping_log1p/uq_0.99_image/global_level_stds.parquet' % RES_DIR,
                'tiling/%s/default/%d_tile_coordinates.parquet' % (RES_DIR, TILE),
                'tiling/%s/default/%d_tile_stats.parquet' % (RES_DIR, TILE)]
        for rel in rels:
            mf['files'][rel] = sha256(OUT / rel)
        mf['files']['splits/rcc2025necrosis_pid_split.csv'] = sha256(sp / 'rcc2025necrosis_pid_split.csv')
        mf['n_tiles'] = len(coords)
        json.dump(mf, open(ROOT / 'phase2_spora_dataset_manifest.json', 'w'), indent=2)
        print('manifest written', flush=True)

    if not SMOKE:
        import torch
        from spora_io import MultiplexImagingDataset as DS
        ds = DS(name=DATASET, path=str(OUT), modality=MODALITY,
                standardization='quantile_clipping_log1p/uq_0.99_image', resolution=RES_STR, tile_size=TILE,
                disable_quantile_mask=True, filter_list=['gaussian_blur'], use_mean_std=True)
        chk = pd.read_parquet(OUT / MODALITY / 'channels.parquet')
        assert chk['channel_name'].tolist() == names
        t2 = pd.read_parquet(OUT / 'metadata' / 'tissues.parquet')
        assert t2.groupby('patient_id')['split'].nunique().max() == 1
        for s in ['train', 'validation', 'test']:
            tid = t2.loc[t2['split'] == s, 'tissue_id'].iloc[0]
            t = ds.get_tissue(tid, kind='uniprot_filtered')
            assert t.image.shape[0] == len(expected_uniprots), (tid, t.image.shape)
            assert tuple(t.image.shape[1:]) == (224, 224)
            assert torch.isfinite(t.image).all()
            assert list(t.uniprot_ids) == expected_uniprots
            tl = ds.get_tile(tid, 0, kind='uniprot_filtered')
            assert torch.isfinite(tl.image).all() and tuple(tl.image.shape[1:]) == (TILE, TILE)
            full = ds.get_tissue(tid, kind='complete')
            assert full.image.shape[0] == 46
            _ = ds.get_tissue_mask(tid)
            print('split_ok %s %s shape=%s tiles=%d' % (s, tid, tuple(t.image.shape),
                  len(ds.tile_coordinates.get(tid, []))), flush=True)
        print('PHASE2_SPORA_DATASET_OK', flush=True)


if __name__ == '__main__':
    main()
