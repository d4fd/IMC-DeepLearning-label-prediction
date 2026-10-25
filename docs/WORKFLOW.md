# Scientific workflow

The source filenames retain phase numbering so the implemented work can be
followed without publishing the private working repository or outputs.

| Stage | Source | Purpose / prerequisites |
|---|---|---|
| Compatibility | `phase0_t4_reconstruction.py`, `virtues_t4_attention.py` | Adapted upstream reconstruction check; separately obtained upstream example, weights and T4 |
| Marker identity | `phase1_marker_embeddings.py` | Reviewed marker manifest; external protein sequences and ESM2 layer-30 mean embeddings |
| Dataset | `phase2_build_spora_dataset.py` | External images and annotations; patient-grouped split, custom mask, OME-Zarr, train-fitted statistics and tiling |
| Model-path smoke | `phase3_virtues_smoke.py` | Prepared images, marker embeddings, upstream code/weights; finite input and encoder/decoder checks |
| RF reference | `phase4_rf_baseline.py` | Per-channel mean/std/median/p95/positive-fraction features; training-patient grouped CV |
| Frozen representation | `phase4_extract_tile_tokens.py`, `phase4_verify_tile_tokens.py` | Development-only token extraction and cache validation |
| Pooling | `phase4_heads.py` | Mean versus attention-MIL over tiles within each core |
| Target contract | `phase4_prepare_aux_targets.py` | Training-only eligibility, categorical vocabularies, continuous scaling, missingness masks |
| Multitask learning | `phase4_equal_mtl.py`, `phase4_uncertainty_mtl.py` | Matched image-only control, fixed versus learned task weighting |
| Ablations | `phase4_loao_mtl.py`, `phase4_head_depth_mtl.py` | Leave-one-auxiliary-out and shared-head depth comparisons |
| Recipe search | `phase5_finetune_mtl.py` | Head-only grouped training-patient CV and normalization/target/model helpers |
| Stopping | `phase5_validation_earlystop_8seed.py` | One common epoch selected by mean validation average precision across seeds |
| Final training | `phase6_train_final.py` | Frozen encoder, mean pooling, deeper head, dropout 0.8, seeds 42–49 |
| Test extraction | `phase6_extract_test_features.py` | Original evaluator's extraction logic, split out behind explicit test authorization and the final lock |
| Final evaluation | `phase6_final08_evaluate.py` | Validation-derived Youden-J threshold lock, ensemble probabilities and patient-clustered intervals |
| Interpretation | `phase6_marker_importance.py` | Development-core marker occlusion; probability changes measure model reliance, not causation |

The RF script retains metadata-only and combined **analysis references**. Only
the image-only RF is a fair inference-input comparator. Those analysis references
are not the image-only deployment model and must not be mistaken for it.

Categorical auxiliaries use cross-entropy; the continuous auxiliary uses
train-standardized Huber loss. Missing or unknown auxiliary labels are masked
without dropping the primary task. The encoder is frozen; the deeper head is
`512 → 128 → 64`, with one primary and task-specific auxiliary output layers.

## Intentional source-export adaptations

- Runtime paths resolve through `submission_paths.py` (historical helper name).
  Outputs go under an ignored work directory, not beside source files.
- No historical data files, fitted contracts, hashes of private artifacts,
  performance scores, operating thresholds or patient examples are included.
  Cohort-size assertions and a literal comparator score were replaced with
  runtime validation and the caller's generated comparison record.
- Patient pseudonym generation now requires a locally supplied secret. Generated
  pseudonyms and lookup tables remain private; hashing does not anonymize them.
- Auxiliary provenance must be confirmed for a new dataset rather than inheriting
  a confirmation made for the original study.
- The selected recipe identifier comes from the new run's search output rather
  than requiring a historical candidate ID. The final trainer has a neutral
  filename and writes heads directly to its runtime final-model directory.
- The RF source restricts image-feature extraction and diagnostics to development
  partitions. Evaluation no longer depends on private predecessor locks/results;
  it requires explicit test-access authorization, verifies its runtime lock and
  refuses to overwrite an existing evaluation. Nonempty dataset and final-head
  output directories are also rejected to reduce accidental overwrites.
- Test extraction preserves the original float16-token to float32-mean calculation,
  but requires an empty cache and writes lock-bound feature hashes rather than
  silently reusing old files. Evaluation verifies that cache provenance.
- The original compact/full feature loader remains: per-tile means can be read
  directly, or calculated from full patch-summary tokens. No such cache is bundled.
- `run_stage.py` offers a no-execution prerequisite check; `demo.py` uses the
  actual exported model/trainer on invented in-memory inputs.

These changes are portability/publication safeguards, **not a claim of numerical
parity** with the historical pipeline. Real-data execution of these adapted
copies has not been validated here. Do not substitute the synthetic smoke for
encoder compatibility, marker-order, schema or full pipeline validation.

## Limits that matter when reading or reusing the code

- The percentile tissue mask is project-owned, not the released Spora mask. It
  uses all source channels, including channels excluded from encoder input.
  No mask-accuracy or release-equivalence validation is claimed. Standardization
  mirrors the release sequence but lacks a recorded equivalence check.
- A historical mask diagnostic included some test images. Do not describe all
  original preprocessing inspection as development-only.
- The historical nominal `600/224` pixel scale is retained in dataset paths and
  code; upstream physical geometry was not independently established. Reuse
  requires reviewing geometry rather than treating that scalar as universal.
- The scripts retain the original channel/input-shape contract. A different
  panel or geometry requires deliberate adaptation and new validation.
- Head-only CV uses encoder features with preprocessing statistics fitted on
  the original training partition, not a fully fold-local end-to-end pipeline.
- Validation used for recipe/epoch/threshold selection is not an independent
  performance assessment. The historical final assessment reused a previously
  evaluated test partition; this export does not erase that limitation.
- The T4 SDPA adapter is eval-only and rejects attention dropout. It is not a
  general FlashAttention replacement or evidence of performance equivalence.

No results, reports, post-test comparison utilities, migration tools or original
notebooks are published. This is a methods/source showcase, not a clinical claim.
