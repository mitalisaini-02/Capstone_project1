# Code Architecture (after the pre-Phase-3 refactor)

## Packages and responsibilities

| Package | Responsibility | Main inputs → outputs | Depends on (src) |
|---|---|---|---|
| `src/mobility` | Load SUMO traces; roles and lifetimes from data | `mobility.csv` → `RunMobility` | `data.validator` |
| `src/data` | Validation; legacy pilot loader; dataset splitter | DataFrames → checks / splits | — |
| `src/features` | Kinematics, LSTM labels, sequence building, scaler | traces → (N,10,7) windows, labels | `features.*` |
| `src/models` | LSTM network and `LinkSurvivalPredictor` (unchanged weights and preprocessing) | windows → probabilities | `features.scaler`, `reliability.policy` |
| `src/training` | Training loop, checkpoints, dataset | — | `training.*` |
| `src/reliability` | **ReliabilityModel interface**, LSTM adapter, kinematic model, **ThresholdPolicy**, **OnlineFeatureBuilder** (training preprocessing, bitwise-equal windows) | windows → P(survive); P → eligible | `models.predictor`, `evaluation.baselines` |
| `src/network` | **LinkModel interface**, `abstract_sidelink_v0`, `nr_sidelink_tr37885_v1`, registry, technology profiles, **gNodeB coverage**, **building geometry** (`buildings.py`), **channel-state classifiers G/F1/F2/F3** (`channel_state.py`) | distance/state → SNR, capacity, outage; positions → serving cell | `network.*` only |
| `src/common` | Units and cost conversion | — | none |
| `src/datagen` | Synthetic PUE/CUE/gNodeB generation, connectivity, validation, statistics | configs + traces → scenario files | `mobility`, `network.coverage`, `common`, `datagen.*` |
| `src/evaluation` | Classification metrics, baselines, plots | — | — |
| `src/utils` | Strict config loader, logging, seeding, file integrity | — | — |
| `src/simulation` | Phase 3: per-run channel/reliability/episode caches; per-scenario link feasibility (stages + reasons, expected vs realized capacity, C3/C4/C5, gNodeB coverage); validation; stats | caches + Phase 2 scenarios → request_seconds, candidate_links | `network`, `reliability`, `datagen.validate.Checks` |
| `src/marketplace` | Phase 4: EpochProblem builder (eligibility from Phase 3 files), posted pricing, admission/rejection reasons, greedy and two-stage MILP allocators, independent audit, evaluation | Phase 2/3 files → allocation_results, epoch_summary, metrics | `common.units` only (no reliability/radio imports) |
| `src/mec` | **Not yet implemented** (Phase 5: sessions, reallocation) | | |

## Interfaces

```python
# src/reliability/base.py
class ReliabilityModel:                  # name, label_range_m, history_window_s, horizon_s, feature_names
    def predict_proba(windows) -> np.ndarray      # unscaled (N, T, F) -> (N,) in [0, 1]
    def range_mismatch(discovery_range_m) -> bool

# src/reliability/policy.py
ThresholdPolicy(theta).is_reliable(p) / .mask(ps)
ThresholdPolicy.primary_from_metadata(...)   -> 0.62 (read from checkpoint_metadata.json)
ThresholdPolicy.sensitivity_from_file(...)   -> 0.65 (read from sensitivity_threshold.json)

# src/network/base.py
class LinkModel:                          # name
    def snr_db(d, state=None); def capacity_mbps(d, state=None)
    def link_capacity_mbps(d, link_peak, state=None)        # C3: min(link_peak_i, capacity)
    def expected_capacity_mbps(d, context=None); def outage_probability(d, thr_db, context=None)
    def capacity_table(distances); reporting_snr_floor_db (optional)

# src/network/registry.py
load_link_model(profiles_path, name)      # dispatch on the profile's "model" field
register_model_type(model_type, factory)  # add a technology without touching other modules

# src/network/buildings.py
load_building_layer(buildings_poly, coverage_poly) -> BuildingLayer   # validated, grid-indexed
BuildingLayer.blocking_buildings(px,py,qx,qy) / endpoint_building(x,y) / in_coverage(x,y)

# src/network/channel_state.py
ChannelStateClassifier.classify(pairs) -> rows with source/fallback/state probabilities
BuildingGeometryClassifier (G) | RoadTopologyClassifier (F1) | AllBuildingBlocked (F2) | AllBuildingClear (F3)
assign_episode_states(classified, run_id, seed)   # per-episode reproducible state

# src/network/technology.py
load_technology(path, name) -> TechnologyProfile(discovery_range_m, link_profile, ...).link_model()
```

## Dependency rules (checked by an AST cycle check)

- No import cycles among the `src` modules.
- `network` imports nothing from `models`, `features`, `reliability`, `datagen` or `training`.
- `reliability` and `models` import nothing from `network` or `datagen`.
- The future optimiser will receive eligible links, capacities, prices and constraints as plain
  data. It must not import `reliability` or `models`.
- Configuration lives in `config/`. Code takes file paths as parameters. Defaults are only
  conveniences and are never substituted for a missing file (`load_config` raises).

## Backward compatibility kept

- `src/datagen/mobility_source.py` re-exports `src.mobility.trace`.
- `src.datagen.generators.load_gnb_sites` and `src.datagen.metrics.serving_cell` delegate to
  `src.network.coverage`.
- `generators.MBIT_PER_GB` is the same object as `src.common.units.MBIT_PER_GB`.
- `LinkSurvivalPredictor.is_link_reliable` is unchanged in behaviour (it now delegates to `ThresholdPolicy`).
- `LinkCapacityModel` and `load_link_profile` are unchanged in numerics (bitwise verified).
