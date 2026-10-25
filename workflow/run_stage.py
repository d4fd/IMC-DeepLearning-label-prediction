"""Explicit opt-in launcher; preflight does not import or execute scientific stages."""

import argparse
import importlib.util
import subprocess
import sys

import submission_paths as paths


STAGES = {
    'reconstruction': ('phase0_t4_reconstruction.py', 'encoder'),
    'markers': ('phase1_marker_embeddings.py', 'markers'),
    'prepare': ('phase2_build_spora_dataset.py', 'images'),
    'encoder-smoke': ('phase3_virtues_smoke.py', 'encoder'),
    'baseline': ('phase4_rf_baseline.py', 'images'),
    'extract': ('phase4_extract_tile_tokens.py', 'encoder'),
    'verify-tokens': ('phase4_verify_tile_tokens.py', 'tokens'),
    'pooling': ('phase4_heads.py', 'tokens'),
    'auxiliaries': ('phase4_prepare_aux_targets.py', 'dataset'),
    'equal-mtl': ('phase4_equal_mtl.py', 'tokens'),
    'uncertainty': ('phase4_uncertainty_mtl.py', 'tokens'),
    'leave-one-out': ('phase4_loao_mtl.py', 'tokens'),
    'head-depth': ('phase4_head_depth_mtl.py', 'tokens'),
    'search': ('phase5_finetune_mtl.py', 'tokens'),
    'early-stop': ('phase5_validation_earlystop_8seed.py', 'tokens'),
    'final-train': ('phase6_train_final.py', 'tokens'),
    'extract-test': ('phase6_extract_test_features.py', 'encoder'),
    'evaluate': ('phase6_final08_evaluate.py', 'evaluation'),
    'occlusion': ('phase6_marker_importance.py', 'encoder'),
}


def preflight(stage):
    script, kind = STAGES[stage]
    required = []
    if kind in {'images', 'markers', 'encoder'}:
        required.append(paths.root() / 'phase1_marker_manifest.csv')
    if kind == 'images':
        required += [paths.source() / 'metadata.csv', paths.source() / 'panel.csv', paths.images()]
    if kind in {'dataset', 'tokens', 'evaluation', 'encoder'} and stage != 'reconstruction':
        required.append(paths.dataset() / 'metadata/tissues.parquet')
    if kind == 'tokens':
        required += [paths.tokens() / 'train', paths.tokens() / 'validation']
    if kind == 'encoder':
        required += [paths.root() / 'virtues', paths.root() / 'checkpoints/virtues-sp32/model.safetensors']
    if kind == 'evaluation':
        required += [paths.root() / 'phase4_aux_targets_v1.json', paths.root() / 'final_model_08']
    if stage == 'extract-test':
        required.append(paths.root() / 'phase6_final08_model_lock.json')
    modules = {'numpy', 'pandas', 'torch', 'sklearn', 'pyarrow'}
    if kind in {'images', 'encoder'}:
        modules |= {'tifffile', 'scipy', 'skimage'}
    if stage == 'prepare':
        modules |= {'zarr', 'spora_io'}
    if kind == 'encoder':
        modules |= {'omegaconf', 'safetensors', 'einops', 'spora_io', 'virtues'}
    if kind == 'markers':
        modules |= {'esm', 'Bio', 'requests', 'loguru'}
    missing = [str(p) for p in required if not p.exists()]
    missing += ['Python dependency: ' + name for name in sorted(modules) if importlib.util.find_spec(name) is None]
    return script, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['list', *STAGES])
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='Check basic prerequisites only; no data is loaded')
    action.add_argument('--execute', action='store_true', help='Explicitly run a stage; may train or use a GPU')
    args, extra = parser.parse_known_args(argv)
    if args.stage == 'list':
        for name, (script, _) in STAGES.items():
            print(f'{name:16} {script}')
        return 0
    if extra and not args.execute:
        parser.error('Stage arguments require --execute')
    script, missing = preflight(args.stage)
    if missing:
        print('Stage not run. Missing external prerequisites:', file=sys.stderr)
        for item in missing:
            print('  ' + item, file=sys.stderr)
        print('See docs/INPUTS.md; this repository intentionally contains no study data.', file=sys.stderr)
        return 2
    if not args.execute:
        print('Basic prerequisites found; no data loaded or stage executed. Review docs/WORKFLOW.md before --execute.')
        return 0
    work = paths.root()
    # A runtime directory must not be placed inside the tracked source folders.
    if work == paths.code().parent or any(work == p or p in work.parents for p in
                                          (paths.code(), paths.code().parent / 'docs', paths.code().parent / 'tests')):
        raise SystemExit('PROJECT_WORK_DIR must not overlap tracked source or documentation')
    work.mkdir(parents=True, exist_ok=True)
    if extra[:1] == ['--']:
        extra = extra[1:]
    return subprocess.call([sys.executable, str(paths.code() / script), *extra], cwd=work)


if __name__ == '__main__':
    raise SystemExit(main())
