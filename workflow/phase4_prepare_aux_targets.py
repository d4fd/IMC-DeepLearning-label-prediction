"""Prepare immutable v1 auxiliary-target contract using train/validation rows only.

Independence and pre-fitting annotation availability must be confirmed for each
new authorized dataset. Test rows and labels are not read.
"""
from __future__ import annotations
import submission_paths as _submission_paths
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(str(_submission_paths.root()))
DATA = _submission_paths.dataset() / 'metadata/tissues.parquet'
OUTPUT = ROOT / 'phase4_aux_targets_v1.json'
SOURCE = 'external metadata.csv -> runtime metadata/tissues.parquet'
DEFS = {
    'gender': ('categorical', ['f', 'm'], 'Recorded sex category; fixed binary source vocabulary.'),
    'age': ('continuous', None, 'Age in years is genuinely quantitative; train-standardized Huber loss.'),
    'subtype_combined': ('categorical', ['ccRCC'], 'Histologic subtype; exclude if train has only one category.'),
    'lymphocytes_combined': ('categorical', ['0', '1/2'], 'Grouped pathology categories, not equal-interval quantities.'),
    'pT_combined': ('categorical', ['1/2', '3/4'], 'Grouped pathologic T-stage categories.'),
    'ISUP_combined': ('categorical', ['1/2', '3/4'], 'Grouped ISUP grade categories.'),
    'PDL1_score': ('categorical', ['0', '1'], 'Observed binary scored categories; no interval assumption.'),
    'genomic_subtype': ('categorical', ['BAP1_driven', 'PBRM1_driven', 'VHL_monodriver', 'VHL_wildtype', 'multiple_drivers'], 'Distinct named genomic mechanisms.'),
    'genomic_risk': ('categorical', ['low_risk', 'medium_risk', 'high_risk'], 'Ordinal risk categories; numerical intervals not established.'),
    'metabolic_subtype': ('categorical', ['1', '2', '3', '4'], 'Numeric codes denote distinct metabolic subtype groups, not a continuous scale.'),
}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def value_of(name, x):
    if pd.isna(x):
        return None
    if name in ('PDL1_score', 'metabolic_subtype'):
        v = float(x)
        assert v.is_integer(), (name, v)
        return str(int(v))
    return str(x)


def main():
    if os.environ.get('PROJECT_AUX_PROVENANCE_CONFIRMED') != '1':
        raise SystemExit('Confirm independent, pre-fitting auxiliary annotations with PROJECT_AUX_PROVENANCE_CONFIRMED=1')
    if OUTPUT.exists():
        raise FileExistsError(f'Versioned contract exists: {OUTPUT}')
    fields = ['split', 'patient_id', 'necrose', *DEFS]
    df = pd.read_parquet(DATA, columns=fields, filters=[('split', 'in', ['train', 'validation'])])
    assert set(df['split']) == {'train', 'validation'}
    manifest = {
        'version': 1,
        'split_artifact_sha256': digest(ROOT / 'splits/rcc2025necrosis_pid_split.csv'),
        'source': SOURCE,
        'provenance': {'independent_of_necrose': True,
                       'available_before_model_fitting': True, 'assessment_timestamps_in_source': False},
        'scope': 'train/validation only; test annotations sealed until Phase 5',
        'excluded_inference_fields': ['sample_id', 'PID', 'necrose', 'width_px', 'height_px', 'TMA', 'type', 'age_bin', 'metabolic_subtype_name', 'surv_censor_all', 'surv_censor_evnt'],
        'targets': {},
    }
    for name, (kind, allowed, rationale) in DEFS.items():
        train = df[(df['split'] == 'train') & df['necrose'].notna()].copy()
        train['label'] = train[name].map(lambda x: value_of(name, x)) if kind == 'categorical' else train[name]
        valid = train[train['label'].notna()]
        groups = int(valid['patient_id'].nunique())
        desc = {'type': kind, 'source': SOURCE, 'timing': manifest['provenance'],
                'rationale': rationale, 'train_labeled_pids': groups, 'missingness_by_split': {}}
        for split in ('train', 'validation'):
            subset = df[df['split'] == split]
            primary = subset[subset['necrose'].notna()]
            desc['missingness_by_split'][split] = {
                'total_cores': int(len(subset)), 'primary_labeled_cores': int(len(primary)),
                'aux_missing_cores': int(subset[name].isna().sum()),
                'aux_missing_among_primary_labeled': int(primary[name].isna().sum()),
                'aux_labeled_primary_cores': int(primary[name].notna().sum()),
                'aux_labeled_primary_pids': int(primary.loc[primary[name].notna(), 'patient_id'].nunique()),
            }
        if kind == 'categorical':
            observed = set(valid['label'].unique())
            assert observed <= set(allowed), (name, observed - set(allowed))
            counts = {v: int(valid.loc[valid['label'] == v, 'patient_id'].nunique()) for v in allowed}
            desc['vocabulary'] = allowed
            desc['class_recoding'] = 'Identity for named classes; integral 0/1 or 1-4 codes converted to stable strings.'
            desc['train_pid_count_by_class'] = counts
            desc['n_distinct_train_classes'] = sum(n > 0 for n in counts.values())
            desc['eligible'] = groups >= 30 and len(observed) >= 2 and all(counts[v] >= 10 for v in observed)
            desc['reason'] = 'meets >=30 labeled training PIDs and >=10 per class' if desc['eligible'] else 'ineligible: insufficient classes or PID coverage'
            validation = df[(df['split'] == 'validation') & df['necrose'].notna()]
            desc['validation_unknown_primary_cores'] = int(sum(value_of(name, v) not in allowed for v in validation[name] if pd.notna(v)))
        else:
            vals = pd.to_numeric(valid['label'], errors='raise').to_numpy(dtype=float)
            desc['train_mean'] = float(vals.mean()) if len(vals) else None
            desc['train_std'] = float(vals.std(ddof=0)) if len(vals) else None
            desc['eligible'] = bool(groups >= 20 and len(vals) > 1 and np.isfinite(vals).all() and vals.std(ddof=0) > 0)
            desc['reason'] = 'meets >=20 labeled training PIDs and nonzero variance' if desc['eligible'] else 'ineligible: insufficient PID coverage or variance'
            desc['standardization'] = 'subtract training-primary-labeled mean and divide by training-primary-labeled population std'
        manifest['targets'][name] = desc
    manifest['retained'] = [name for name in DEFS if manifest['targets'][name]['eligible']]
    OUTPUT.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('RETAINED', manifest['retained'])
    print('AUX_MANIFEST_V1_OK')


if __name__ == '__main__':
    main()
