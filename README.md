# IMC Deep Learning Label Prediction

Code from a course project investigating **image-level prediction of
pathology-assessed tumour necrosis (`necrose`) from imaging mass cytometry (IMC)**.
This repository shows the methods and implementation, not study results.

## What we built

The workflow combines a transparent image-statistics Random Forest baseline
with a frozen **VirTues** encoder and a core-level multitask classifier:

```text
External IMC core → marker mapping → masking / normalization → tiles
                → frozen VirTues → mean patch and tile pooling
                → shared 512 → 128 → 64 head → necrosis probability
                                         └→ auxiliary training heads
```

One core is one prediction unit. Patients, not images, determine the partitions.
Clinical metadata is auxiliary supervision during training, **not an inference
input**. The final design uses dropout 0.8 and averages eight seed-level
probabilities. Marker identity and train-fitted per-marker normalization are
required before encoder use.

## Try the synthetic demo

Python **3.12 or newer** is recommended. In a fresh virtual environment:

On Windows, prefer a standalone CPython virtual environment rather than one
inheriting an older Conda runtime; conflicting DLLs can prevent PyTorch loading.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python demo.py
```

The demo generates artificial tile features in memory, exercises the exported
multitask head and missing-label losses with one tiny CPU update, and checks
that padded tiles do not affect predictions. It writes no files, downloads no
models, and uses no study data or trained weights. It does **not** demonstrate
IMC preprocessing, encoder inference or scientific performance.

For checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python verify_publication.py --history
python workflow/run_stage.py list
python workflow/run_stage.py final-train --check
```

The last command deliberately reports missing prerequisites in a clean clone;
no real-data stage is run. Actual stages require explicit `--execute` and may
be expensive. Do not run them without data permission and a reviewed protocol.

## Repository map

| Location | Contents |
|---|---|
| `workflow/phase0_*.py`–`phase3_*.py` | Encoder compatibility, marker preparation and dataset conversion |
| `workflow/phase4_*.py` | RF baseline, frozen features, pooling and multitask comparisons |
| `workflow/phase5_*.py` | Patient-grouped recipe search and common-epoch stopping |
| `workflow/phase6_train_final.py` | Eight-seed final-head training |
| `workflow/phase6_extract_test_features.py` | Locked, explicitly authorized test-feature extraction |
| `workflow/phase6_final08_evaluate.py` | Validation-derived threshold lock and explicitly authorized evaluation |
| `workflow/phase6_marker_importance.py` | Development-only marker-channel occlusion |
| `workflow/virtues_t4_attention.py` | Project-owned, eval-only T4 attention adapter |
| `docs/WORKFLOW.md` | Scientific stage map and export adaptations |
| `docs/INPUTS.md` | External input contracts and optional environment setup |
| `docs/PRIVACY.md` | Publication boundary and audit commands |
| `demo.py`, `tests/` | Synthetic-only demonstration and checks |

These are curated copies of the scientific scripts, not a newly validated
end-to-end product. Real-data runs require external inputs and stage-generated
contracts. Dependencies are convenience requirements, not an exact historical
environment lock. See [workflow notes](docs/WORKFLOW.md) for limitations.

## No study data is distributed

There are no original images or metadata, patient identifiers or pseudonyms,
split tables, fitted normalization statistics, feature caches, predictions,
trained heads, figures, reports, notebooks or private Git history. Runtime
outputs are excluded from version control. No original panel or marker table
is bundled; new users must supply their own authorized, reviewed mapping.

The private production plan is intentionally not part of this repository.
Automated audits are guardrails, not a guarantee that arbitrary future additions
are safe. Review every publication, including text inside source files.

## Attribution and license

Team code: [MIT](LICENSE). This grants no rights to study data or upstream model
weights. VirTues, Spora and ESM2 are separate dependencies; see
[third-party notices](THIRD_PARTY_NOTICES.md). In particular, the VirTues encoder
weights have a separate **CC BY-NC 4.0** license and are not included.

Research/educational code only; not a clinically validated diagnostic tool.
