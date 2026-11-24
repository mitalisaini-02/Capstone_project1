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

## Phase 2 — Synthetic marketplace dataset generation (2026-11-24)

Scope: reproducible synthetic PUE / CUE / gNodeB datasets on top of existing SUMO mobility, plus
derived connectivity and ground truth. No LSTM changes, no retraining, no optimiser, no MEC integration.

### Decisions applied
- Link capacity: provisional, technology-agnostic `abstract_sidelink_v0` (log-distance + Shannon).
  Candidate discovery stays the 1000 m geographic range (the LSTM label definition).
- One-factor-at-a-time scenarios plus stress combinations. **Correction:** the approved factors give
  **13** configurations (1 baseline + 9 single-factor + 3 stress), not 12 → 156 scenarios per split.
- No battery model. No new SUMO runs. Bulk CSVs excluded from git.
- θ = 0.62 stays the primary baseline. θ = 0.65 added as a validation-only sensitivity variant.

### Calibration (validation runs 000/001/006/013 only; frozen before any test data existed)
- `p_register`: low 0.10 (supply/demand 0.46), med 0.20 (1.03), high 0.35 (1.95).
- gNodeB total capacity: 30 Mbps for both levels. The levels differ through load fraction
  (0.3 vs 0.8): validation carry share 0.984 vs 0.671. "Sufficient" stays sufficient under high
  demand (0.971 on validation).

### Files created
- `src/datagen/{__init__,mobility_source,connectivity,scenarios,generators,metrics,validate,io_utils}.py`
- `src/network/{__init__,link_capacity}.py`
- `scripts/generate_marketplace_data.py`, `scripts/threshold_grouped_ci.py`
- `config/marketplace/{base,scenarios,link_profiles}.json`, `config/marketplace/frozen/{frozen_config,calibration_report}.json`
- `docs/datasets.md`
- `tests/datagen_fixtures.py`, `tests/test_{link_capacity,mobility_source,connectivity,generators}.py`
- `data/derived/connectivity/<8 runs>/`, `data/synthetic/marketplace/{val,test}/` (156 scenarios each)
- `outputs/large_dataset_experiment/reliability_threshold_robust/`

### Files modified
- `docs/formulation.md` — C9 written as a per-GB price limit consistent with the total budget;
  definition of `B_ij`; four-concept distinction; θ sensitivity variant noted.
- `.gitignore` — excludes bulk generated CSVs.
- `docs/CHANGELOG.md` — this entry.

### Verification
- 59 tests pass (39 from Phase 1 + 20 new).
- 6,216 scenario checks per split + 11 connectivity checks per run: all passed.
- Reproducibility: 1,560 file hashes regenerated and compared, 0 mismatches.
- Connectivity rows reproduce the LSTM dataset's (time, pair, label) sets exactly (run_000, run_013).
- Protected artifacts: 53/53 unchanged.

### Findings to carry forward (not acted on)
- Under the provisional link profile, about 82 % of test candidate rows (71 % on validation) have
  SNR < −5 dB. The median candidate `B_ij` is about 1.5 Mbps. The PUE contribution will be small
  unless the link model or range changes.
- Only about 19 % of active request-seconds have any geographic candidate PUE. Demand per scenario is
  small (about 15 requests, mean concurrency 1.4), a property of the existing SUMO traces.
- About 46 % of requests have `max_price_per_gb` below the gNodeB price, so budget-driven
  rejection will be common under the assumed U(0.6, 1.5) budget range.

## Pre-Phase-3 — Architecture refactor and NR link model (2026-11-24)

Behaviour-preserving refactor plus a new, separate NR sidelink link profile. No changes to the
LSTM, its labels, θ (0.62 primary / 0.65 sensitivity), Phase 2 datasets or frozen configs.

### Decisions applied
- Technology framing: NR sidelink UE-to-Network relay (PUE = relay UE, CUE = remote UE, PC5 sidelink,
  cellular backhaul, MEC = coordinator). Smartphone support is not assumed.
- New link profile `nr_sidelink_tr37885_v1` (and `_highway`) in a **new** file
  `config/network/link_profiles_nr.json`. `abstract_sidelink_v0` is unchanged.
- Reliability Option A: existing 1000 m LSTM. Discovery, LSTM reliability and link feasibility
  are kept as three separate stages.

### Refactor (with backward-compatible shims)
- `src/mobility/` (moved from `src/datagen/mobility_source.py`, which is now a re-export shim).
- `src/reliability/`: `ReliabilityModel` interface, LSTM adapter, kinematic model, `ThresholdPolicy`.
  `LinkSurvivalPredictor.is_link_reliable` delegates to the policy (same behaviour).
- `src/network/`: `LinkModel` interface (`base.py`), registry, technology profiles, `coverage.py`
  (gNodeB sites and serving cell moved out of datagen), `tr37885.py`.
- `src/common/units.py`: `MBIT_PER_GB` and cost conversion (single source of truth).
- `src/datagen/stats.py`: scenario statistics moved out of `scripts/generate_marketplace_data.py`.
  Statistics use `LinkModel.reporting_snr_floor_db` instead of reading v0 internals.
- `src/utils/config_loader.py`: strict. It raises `ConfigNotFoundError` and never substitutes another config.

### Verification
- 88 tests pass (59 before + 29 new).
- LSTM recorded test metrics reproduced (`scripts/verify_predictor.py`: all checks passed).
- Predictor and LSTM adapter outputs are **bitwise identical** to pre-refactor outputs (val, test,
  single-sample). The `is_link_reliable` decisions are identical.
- `abstract_sidelink_v0` capacity/SNR/table bitwise identical. gNodeB serving cells identical.
- Phase 2 regenerated in a scratch directory: **1,560/1,560 file hashes identical**, per-scenario
  statistics identical, all validations passed.
- Protected artifacts 53/53 unchanged. Phase 2 snapshot 1,921/1,921 files unchanged.
- No import cycles. `network` ↔ `reliability`/`models` have no cross-imports.

### Found and fixed during the refactor
- Gauss–Hermite quadrature was inaccurate for the discontinuous outage/capacity integrands
  (0.0030 vs exact 0.0042). Replaced with closed forms (outage) and dense trapezoidal grids
  (expected capacity), checked against Monte Carlo.
- The moved statistics read `link_model.p[...]` (v0 internals). Replaced by an interface property.

## Phase 3 (part 1) — Building-aware channel-state classifier (2026-11-24)

Implements the approved building-footprint LOS/NLOS classification. No changes to the LSTM, labels,
thresholds, Phase 2 data, frozen configs, `abstract_sidelink_v0` or the verified NR equations.

### Decisions applied
- Buildings from `map (1).osm` via SUMO `polyconvert` (exact network coordinates). Treated as
  incomplete, so G is an optimistic estimate. No external datasets.
- Vehicle blockage: TR probability model only (no geometric override).
- Urban LOS/NLOSv applied to building-clear paths, documented as a geometry-based extension of TR 37.885.
- Endpoint inside a building: explicit `endpoint_inside_building` flag, labelled fallback,
  excluded from primary results. Never LOS.
- 2-D test; all mapped footprints block (no heights).
- Per-episode reproducible state; building obstruction recalculated every second.

### Files created
- `config/network/geometry.json`, `scripts/build_building_geometry.py`
- `data/derived/geometry/{buildings.poly.xml, coverage.poly.xml, geometry_manifest.json}` (poly XML git-ignored)
- `src/network/buildings.py`, `src/network/channel_state.py`
- `tests/test_channel_state.py` (25 tests)

### Files modified (additive only)
- `src/network/tr37885.py`: added `expected_capacity_given_states` and `outage_probability_given_states`.
  The file minus this block is byte-identical to the verified version, and all 17 model output arrays
  are bitwise identical.
- `src/mobility/trace.py`: added `load_road_ids` (RunMobility unchanged).
- `docs/network_models.md` (§6 classifier, §7 limitations), `docs/architecture.md`, `.gitignore`.

### Verification
- 113 tests pass (88 + 25). The real geometry loads 3,951 polygons with 0 rejected; the spatial
  index agrees with brute force on random and real-map segments.
- Predictor recorded metrics reproduced; predictor/adapter bitwise identical; Phase 2 regenerated
  1,560/1,560 hashes identical; protected 53/53; Phase 2 snapshot 1,921/1,921 unchanged.

### Found during implementation
- The polygon validator reported a bow-tie as `zero_area` (its lobes cancel). Self-intersection is
  now checked first, so the reason is correct.

## Phase 3 — Mobility and network simulation (2026-11-24)

Integrates SUMO mobility, geographic discovery, the unchanged LSTM (θ 0.62 primary / 0.65
sensitivity), building-aware channel states (G primary; F1/F2/F3 sensitivity), the NR link model,
PUE resources (C3/C4/C5), gNodeB coverage/capacity, provider availability and link departures.
No pricing, selection, allocation or MEC session logic.

### Expected vs realized capacity
- Before this phase there was no realized-capacity function, only expected capacity and the
  median-state `capacity_mbps(d, state)`.
- Added (additive) `realized_snr_db` / `realized_capacity_mbps` to the `LinkModel` interface and
  the NR model. They take per-episode standard-normal draws for shadowing and NLOSv blockage.
  With zero draws they reproduce the verified median-state values exactly.
- Outputs carry distinct columns: `expected_radio_capacity_mbps` (reporting only),
  `realized_radio_capacity_mbps` and `realized_link_capacity_mbps` (feasibility and allocation input).
  A test proves expected values cannot change any feasibility output.

### Files created
- `config/simulation/phase3.json`, `scripts/run_phase3_simulation.py`
- `src/reliability/online_features.py` (training preprocessing; windows bitwise equal to the training data)
- `src/simulation/{__init__,channel_layer,reliability_layer,feasibility,validate,stats}.py`
- `tests/test_simulation.py` (16 tests)
- `docs/simulation.md`
- `data/derived/phase3_cache/<8 runs>/`, `data/simulation/phase3/{val,test}/` (bulk CSVs git-ignored)

### Files modified (additive only)
- `src/network/base.py` (realized methods with deterministic defaults), `src/network/tr37885.py`
  (realized methods). Each file minus its additive block is identical to the previous version.
- `docs/architecture.md`, `.gitignore`, `docs/CHANGELOG.md`.

### Found during the run
- Reading the cache produced a pandas DtypeWarning (mixed bool/NaN columns inferred chunk-wise).
  The values were verified correct: real booleans, and building-blocked rows exactly equal to
  geometry NLOS rows. The cache reader now uses whole-file inference, and a validator check was
  added (`building_reason_*_only_on_geometry_blocked`). Validation outputs re-validated from the
  written files: 156 scenarios, 2,652 checks, 0 failures.
