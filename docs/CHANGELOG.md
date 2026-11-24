# Change Log

## Phase 1 — Existing system stabilisation (2026-11-24)

Scope: stabilise the existing LSTM pipeline. No retraining, no synthetic data generation,
no marketplace integration.

### Research decisions recorded
1. Communication range stays at **1000 m** so the trained LSTM can be reused. The link is
   described as an **abstracted vehicular sidelink/access link**, not Wi-Fi Direct.
2. The redundant secondary throughput objective is removed; total-payment minimisation is
   the primary objective. Elastic throughput is kept as a future extension.
3. Insufficient capacity is handled by **admission control**: maximise weighted admitted
   requests first, then minimise cost.
4. The reliability threshold is an experimental parameter selected on validation data
   (0.50–0.90).
5. Three separate PUE capacity constraints are kept: Internet backhaul, aggregate sidelink
   access, and individual PUE–CUE sidelink capacity.

See `docs/formulation.md`.

### Protection of existing artifacts
- `outputs/PROTECTED_ARTIFACTS.sha256` — SHA-256 snapshot of 53 files (all checkpoints,
  scalers, model configs, metrics, plots, reports, processed datasets, splits and root-level
  result CSVs). Taken **before** any change. Verified unchanged at the end of Phase 1 and by
  `tests/test_protected_artifacts.py`.
- No model was retrained. `best_model.pt` files were only read.

### Bugs fixed
| Bug | Fix |
|---|---|
| `scripts/run_large_dataset_pipeline.py:253` called `save_checkpoint` positionally; the epoch number went into the `optimizer` slot and the run crashed after training, so loss curves were never saved. | Keyword arguments. `save_checkpoint` now takes keyword-only arguments after `model` (`src/training/checkpoints.py`), so this mistake raises `TypeError` immediately. |
| `scripts/evaluate_large_dataset.py::update_checkpoint_metadata()` called `torch.save` directly onto `best_model.pt`. | Function removed. Corrected metadata goes in a separate file. |
| `scripts/evaluate_large_dataset.py::fix_pipeline_checkpoint_bug()` rewrote another script's source at runtime. | Function removed. |
| Re-running `run_large_dataset_pipeline.py` would overwrite the trained checkpoint. | Added `--output-dir` (default unchanged) and a guard that refuses to run if `best_model.pt` exists, unless `--allow-overwrite` is given. |
| The large-dataset `best_model.pt` stores `best_threshold = 0.5`, `val_f1 = 0.0` (never updated because of the crash). | Not edited. New `checkpoint_metadata.json` holds the correct values; `LinkSurvivalPredictor` accepts `metadata_path=` and warns when it reads the 0.5 default. |

**Correction to an existing report:** `outputs/large_dataset_experiment/reports/experiment_summary.md`
states the checkpoint bug "has been fixed in `scripts/run_large_dataset_pipeline.py`". That was
not true when written; it is true as of Phase 1. The original report is left unedited (protected).

### Files modified
- `src/training/checkpoints.py` — keyword-only arguments after `model`; docstring.
- `scripts/run_large_dataset_pipeline.py` — keyword `save_checkpoint` call; `--output-dir`,
  `--allow-overwrite`, overwrite guard.
- `scripts/evaluate_large_dataset.py` — removed the two functions above (your earlier staged
  edits are preserved).
- `src/models/predictor.py` — optional `metadata_path`; `threshold_source` attribute;
  warning on the default threshold. Default behaviour is unchanged.

### Files created
- `requirements.txt`, `docs/formulation.md`, `docs/CHANGELOG.md`
- `src/utils/integrity.py`
- `scripts/verify_predictor.py`, `scripts/sweep_reliability_threshold.py`
- `tests/test_checkpoints.py`, `tests/test_predictor.py`, `tests/test_split_manifest.py`,
  `tests/test_protected_artifacts.py`, `tests/test_metrics.py`
- `outputs/PROTECTED_ARTIFACTS.sha256`
- `outputs/large_dataset_experiment/verification/predictor_verification.json`
- `outputs/large_dataset_experiment/reliability_threshold/` (sweep CSV, selection JSON,
  test metrics at θ*, plot)
- `outputs/large_dataset_experiment/checkpoints/checkpoint_metadata.json`

### Environment
- `pytest 9.2.0` installed into `.venv`.

### Results
- Predictor reproduces every recorded test metric at θ = 0.53 exactly
  (ROC-AUC 0.8506, F1 0.788, confusion matrix identical).
- Reliability threshold: **θ* = 0.62** (selected on validation; precision 0.9001, recall 0.6672).
- Held-out test at θ* = 0.62: precision 0.8803, recall 0.6343, specificity 0.8595.
  **Test precision is below the 0.90 target**; θ* was not changed in response.

### Open issues carried forward
- `scripts/run_stage3_pipeline.py` and `scripts/evaluate_large_dataset.py` still write to
  fixed output paths and would overwrite the pilot checkpoint or the recorded large-dataset
  reports if re-run. No guard added (outside approved scope).
- The `val_f1` field written by the pipelines actually stores the tuned metric's score
  (balanced accuracy by default).
- SUMO network path is absolute and outside the repository.
