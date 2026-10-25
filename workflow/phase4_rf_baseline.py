"""Phase 4 item 1: re-run the existing RF baselines on the immutable Phase 2 split.

Faithful port of the RF pipelines built in Untitled_restored.ipynb (same features, same
RandomForestClassifier settings, same 5-fold StratifiedGroupKFold diagnostics on the training
partition only), but the split is replaced by the Phase 2 split artifact. The notebook's original
inline split and numbers are superseded by this run.

Reports three models: image_only, metadata_only, image_plus_metadata.
The metadata models are analysis baselines only; metadata is never an inference input.
"""
import submission_paths as _submission_paths
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SRC = Path(str(_submission_paths.source()))
ROOT = Path(str(_submission_paths.root()))
OUT = _submission_paths.dataset()
SEED = 42
EXCLUDED_META = ['sample_id', 'PID', 'necrose', 'TMA']
RF = dict(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=SEED, n_jobs=-1)
SCORING = {'balanced_accuracy': 'balanced_accuracy', 'roc_auc': 'roc_auc', 'pr_auc': 'average_precision'}


def chan_last(image, n):
    if image.shape[-1] == n:
        return image
    if image.shape[0] == n:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f'no {n} channels in {image.shape}')


def main():
    panel = pd.read_csv(SRC / 'panel.csv')
    channels = panel['clean_target'].fillna(panel['channel']).astype(str).tolist()
    meta = pd.read_csv(SRC / 'metadata.csv', na_values=['NA'])
    tissues = pd.read_parquet(OUT / 'metadata' / 'tissues.parquet',
                             filters=[('split', 'in', ['train', 'validation'])])[
        ['tissue_id', 'patient_id', 'split', 'necrose', 'original_file_name']]

    feature_columns = [f'{name}_{stat}' for name in channels for stat in ['mean', 'std', 'median', 'p95', 'positive_fraction']]
    cache = ROOT / 'rf_image_features.parquet'
    reused = False
    if cache.exists():
        cached = pd.read_parquet(cache)
        if set(feature_columns).issubset(cached.columns):
            feat = cached[['sample_id'] + feature_columns].copy()
            reused = True
    if not reused:
        rows = []
        for r in tissues.itertuples():
            img = chan_last(tifffile.imread(_submission_paths.images() / (r.original_file_name + '.tiff')), len(channels)).astype(np.float32)
            px = img.reshape(-1, img.shape[-1])
            f = {'sample_id': r.original_file_name}
            for i, name in enumerate(channels):
                v = px[:, i]
                f[f'{name}_mean'] = v.mean(); f[f'{name}_std'] = v.std(); f[f'{name}_median'] = np.median(v)
                f[f'{name}_p95'] = np.percentile(v, 95); f[f'{name}_positive_fraction'] = (v > 0).mean()
            rows.append(f)
        feat = pd.DataFrame(rows)

    data = tissues.merge(feat, left_on='original_file_name', right_on='sample_id', how='inner', validate='one_to_one')
    data = data[data['necrose'].notna()].reset_index(drop=True)
    data['necrose'] = data['necrose'].astype(int)
    if data.empty:
        raise ValueError('No labeled development cores')

    merged = data.merge(meta[['sample_id'] + [c for c in meta.columns if c not in EXCLUDED_META]],
                        on='sample_id', how='left', validate='one_to_one')
    metadata_columns = [c for c in meta.columns if c not in EXCLUDED_META]
    numeric_metadata = merged[metadata_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_metadata = [c for c in metadata_columns if c not in numeric_metadata]

    tr = data['split'].eq('train').values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    def score(estimator, X, y, groups):
        s = cross_validate(estimator, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=1)
        return ({k: float(s['test_' + k].mean()) for k in SCORING},
                {k: float(s['test_' + k].std()) for k in SCORING})

    image_only = RandomForestClassifier(**RF)
    img_mean, img_std = score(image_only, data.loc[tr, feature_columns], data.loc[tr, 'necrose'], data.loc[tr, 'patient_id'])

    combined_pre = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), feature_columns + numeric_metadata),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('one_hot', OneHotEncoder(handle_unknown='ignore'))]), categorical_metadata),
    ], verbose_feature_names_out=False)
    combined_model = Pipeline([('preprocessor', combined_pre), ('random_forest', RandomForestClassifier(**RF))])
    comb_mean, comb_std = score(combined_model, merged.loc[tr, feature_columns + metadata_columns],
                                merged.loc[tr, 'necrose'], merged.loc[tr, 'patient_id'])

    meta_pre = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric_metadata),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('one_hot', OneHotEncoder(handle_unknown='ignore'))]), categorical_metadata),
    ], verbose_feature_names_out=False)
    meta_model = Pipeline([('preprocessor', meta_pre), ('random_forest', RandomForestClassifier(**RF))])
    meta_mean, meta_std = score(meta_model, merged.loc[tr, metadata_columns],
                                merged.loc[tr, 'necrose'], merged.loc[tr, 'patient_id'])

    report = {
        'split_artifact': 'splits/rcc2025necrosis_pid_split.csv',
        'split_used': 'Phase 2 immutable 70/15/15 PID-grouped split (supersedes the notebook inline split)',
        'seed': SEED, 'n_labeled_images': int(len(data)),
        'n_pids': int(data['patient_id'].nunique()),
        'feature_cache_reused': bool(reused),
        'features': f'{len(feature_columns)} image features: per-channel mean/std/median/p95/positive_fraction over the {len(channels)} clean_target channels',
        'rf': RF, 'cv': 'StratifiedGroupKFold(5, shuffle=True, seed 42) on the training partition only',
        'split_counts': {s: int((data['split'] == s).sum()) for s in ['train', 'validation']},
        'split_positive': {s: int(data.loc[data['split'] == s, 'necrose'].sum()) for s in ['train', 'validation']},
        'metadata_columns': metadata_columns,
        'numeric_metadata': numeric_metadata,
        'categorical_metadata': categorical_metadata,
        'results': {
            'image_only': {'mean': img_mean, 'std': img_std},
            'metadata_only': {'mean': meta_mean, 'std': meta_std},
            'image_plus_metadata': {'mean': comb_mean, 'std': comb_std},
        },
        'notes': 'metadata_only and image_plus_metadata are non-inference analysis baselines (metadata is never a model input); image_only is the plan-required transparent baseline.',
    }
    json.dump(report, open(ROOT / 'phase4_rf_baseline_report.json', 'w'), indent=2)
    print(json.dumps(report['results'], indent=2), flush=True)
    print('split_counts', report['split_counts'], 'positive', report['split_positive'], flush=True)
    print('PHASE4_RF_BASELINE_OK', flush=True)


if __name__ == '__main__':
    main()
