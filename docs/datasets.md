# Marketplace Datasets (Phase 2)

> **SYNTHETIC SIMULATION DATA.** Every marketplace attribute (PUE capacities and prices,
> CUE requests and budgets, gNodeB capacities, load and coverage, PUE withdrawals) is
> generated from **assumed** parameters. None of them are real-world measurements.
> Only mobility (positions, speeds, timestamps, roles, lifetimes) is real SUMO output,
> reused unchanged.

## 1. Four distinct concepts

These are deliberately kept separate in data, code and reporting:

| Concept | Where | Meaning | What it is NOT |
|---|---|---|---|
| **Geographic candidate discovery** | `pairwise_connectivity.geo_in_range` | Pair distance ≤ 1000 m at time t (same range as the LSTM labels) | Not a radio-quality or usability claim |
| **LSTM-predicted distance survival** | Phase 5 (`p̂_ij`, θ) | Probability the pair stays within 1000 m for the next 30 s | Not a prediction of link capacity, SNR or throughput |
| **Individual link capacity (C3)** | `src/network/link_capacity.py` | `B_ij = min(link_peak_i, f(d_ij))` under a provisional abstract model | Not measured performance. A positive value does not prove a usable link |
| **Allocation feasibility** | Phase 7 optimiser | Whether C1–C9 can be satisfied jointly for the admitted set | Not decided by any single column in these datasets |

## 2. Layout

```
data/derived/connectivity/<run_id>/            (derived from REAL SUMO mobility)
    pairwise_connectivity.csv.gz               MEC-visible: past/current information only
    pairwise_ground_truth.csv.gz               EVALUATION ONLY: uses future positions
    connectivity_meta.json
data/synthetic/marketplace/<split>/            (split = val | test)
    marketplace_scenarios.csv                  manifest: scenario, config, run, replicate, seed, hashes
    generation_config.json                     complete configuration incl. frozen values
    dataset_stats.json  validation_report.json  reproducibility_report.json  SYNTHETIC_DATA.md
    <config_id>__<run_id>__r<rep>/
        pue_resources.csv.gz  cue_requests.csv.gz  gnodeb_resources.csv.gz
        gnodeb_sites.csv      pue_withdrawals.csv  scenario_meta.json
config/marketplace/
    base.json  scenarios.json  link_profiles.json
    frozen/frozen_config.json  frozen/calibration_report.json
```

Bulk per-scenario CSVs are excluded from git (`.gitignore`). They are regenerated
byte-identically from the config files and seeds (`scripts/generate_marketplace_data.py`).

## 3. Schemas and units

Units: seconds (s), metres (m), m/s, Mbps, decimal GB (1 GB = 8000 Mb), price units per GB.
The gNodeB price is normalised to `1.0` price unit/GB.

**pairwise_connectivity** (one row per in-range PUE–CUE pair per second):
`run_id, time_s, pue_id, cue_id, distance_m, geo_in_range, pue_observed_s, cue_observed_s, history_ok`
- `*_observed_s` = seconds observed up to and including t. `history_ok` = both ≥ 10, so an LSTM window [t−9, t] exists.
- For each run, the rows with `history_ok` and a non-null label reproduce exactly the (time, pair, label) set used to train and evaluate the LSTM (checked on run_000 and run_013).

**pairwise_ground_truth** (evaluation only, same rows):
`survives_30s_lstm_def` (1/0/null, identical to the LSTM label definition), `time_to_out_of_range_s`,
`pue_remaining_s`, `cue_remaining_s`, `pue_end_is_trace_end`, `cue_end_is_trace_end`, `first_break_s`,
`first_break_cause` ∈ {`out_of_range`, `pue_departure`, `cue_departure`, `both_departure`, `censored_trace_end`}.
A vehicle whose last observation is the end of the SUMO run is **censored**, not a departure.

**pue_resources** (one row per provider per second present in SUMO):
`scenario_id, run_id, time_s, pue_id, x_m, y_m, speed_mps, registered, available,
backhaul_capacity_mbps (C5), access_capacity_mbps (C4), link_peak_mbps (cap inside C3), price_per_gb`.
Capacity and price columns are blank when the PUE is not registered (nothing is advertised).

**pue_withdrawals**: `scenario_id, pue_id, withdraw_time_s, reason`. A withdrawal is permanent
and is distinct from a SUMO departure.

**cue_requests** (one row per request):
`scenario_id, run_id, request_id, cue_id, request_time_s, x_m, y_m, qos_class, required_rate_mbps,
min_rate_mbps, service_duration_s, data_volume_gb, max_price_per_gb, max_budget, priority_weight`.
- `data_volume_gb = required_rate_mbps · service_duration_s / 8000` (exact).
- `max_budget = max_price_per_gb · data_volume_gb` (exact).
- Per-epoch form used by the optimiser (C9): `cost_j(t) ≤ max_price_per_gb · d_j · Δt / 8000`.
  Summed over the full nominal duration this equals `max_budget`, so the two limits are consistent.
- `min_rate_mbps = required_rate_mbps` in v1 (no elastic service).
- At most one active request per CUE. Arrivals are open-loop (independent of allocation outcomes).

**gnodeb_sites**: `gnb_id, x_m, y_m, coverage_radius_m, source` (hand-placed POIs from `infrastructure.add.xml`).

**gnodeb_resources** (per site per second): `scenario_id, time_s, gnb_id, total_capacity_mbps,
background_load_mbps, available_capacity_mbps, price_per_gb`.

**allocation_results** (Phase 7, format reserved): `scenario_id, method, theta, time_s, request_id, cue_id,
admitted, pue_id (single, nullable), pue_rate_mbps, p_hat, gnb_id, gnb_rate_mbps, cost, rejection_reason`.

## 4. Generation methodology

**Seeds and common random numbers.** Each random stream is keyed by
`SeedSequence([base_seed, crc32(run_id), replicate, stream, entity_id])`, independent of the
configuration. Each entity draws a fixed set of random numbers, which are then transformed by the
parameters. Identical seeds give byte-identical files. Changing one factor never changes the draws
of another, and every marketplace method must use the same scenario files.

**PUE.** Registers with probability `p_register` after a delay U{0..30} s, and stays registered until it
departs or withdraws. Availability is an on/off Markov chain (mean on/off durations). Backhaul is
`base · clip(exp(z_t), 0.5, 1.2)`, with `z` an AR(1) process (ρ = 0.95, σ = 0.2). Access capacity,
link peak and price are fixed per PUE. Optional permanent withdrawals use a per-second hazard.

**CUE.** A fraction 0.6 of consumers ever request service. Idle gaps are exponential with rate λ by demand
level. The class (web / video_sd / video_hd / bulk_download) is drawn from a level-dependent mix, and
rate and duration are uniform within the class ranges. Requests never overlap per CUE.

**gNodeB.** Six fixed sites. Background load is `ρ_load · G_tot · exp(z_t)` (AR(1)), clipped to
[0, 0.95 · G_tot]. Coverage radius is 2000 m (full) or 1500 m (gaps). Serving cell = nearest site within radius.

**Calibration (validation runs only), then frozen.** `p_register` per availability level and
`G_tot` per gNodeB level were chosen on runs 000/001/006/013 under the reference conditions, using
simple capacity accounting (no optimisation, no LSTM). They are frozen in
`config/marketplace/frozen/frozen_config.json` together with the SHA-256 of every config file.
Test generation refuses to run if any config changed after freezing.

## 5. Assumed parameter values

All values below are **placeholders/assumptions**, not measured values.

| Parameter | Value |
|---|---|
| PUE backhaul base | U(5, 40) Mbps |
| PUE aggregate access capacity | U(20, 80) Mbps |
| PUE link peak | U(20, 50) Mbps |
| PUE price | gNodeB price × U(0.4, 1.2) mixed / U(0.3, 0.7) cheap / U(0.8, 1.5) expensive |
| Availability on/off means | 300 / 60 s (low: 300 / 120 s) |
| Withdrawal hazard (some) | 1/600 per s |
| `p_register` (calibrated) | low 0.10, med 0.20, high 0.35 |
| CUE ever requests | 0.6 |
| Arrival rate λ | 1/300, 1/120, 1/45 per s |
| Max price per GB | gNodeB price × U(0.6, 1.5) |
| gNodeB total capacity (calibrated) | 30 Mbps (both levels). Load fraction 0.3 (sufficient) vs 0.8 (insufficient) |
| Coverage radius | 2000 m / 1500 m |

## 6. Link profile `abstract_sidelink_v0` (provisional)

Log-distance path loss with free-space loss at 1 m, then Shannon capacity with an efficiency
factor and a spectral-efficiency cap:

| Parameter | Value | Unit |
|---|---|---|
| Carrier frequency (sets free-space loss at 1 m only) | 5.9 | GHz |
| Bandwidth | 20 | MHz |
| Transmit power | 23 | dBm |
| Antenna gains (Tx + Rx) | 0 | dBi |
| Path-loss exponent | 2.7 | – |
| Thermal noise / noise figure | −174 / 9 | dBm/Hz / dB |
| Implementation efficiency | 0.6 | – |
| Spectral-efficiency cap | 6 | bit/s/Hz |
| Reporting SNR floor (statistics only) | −5 | dB |

Calculated values (before the per-PUE link-peak cap):

| Distance | Path loss | SNR | f(d) | Below −5 dB floor |
|---|---|---|---|---|
| 50 m | 93.7 dB | 21.3 dB | 72.0 Mbps (cap-limited) | no |
| 100 m | 101.9 dB | 13.1 dB | 53.1 Mbps | no |
| 200 m | 110.0 dB | 5.0 dB | 24.7 Mbps | no |
| 500 m | 120.7 dB | −5.8 dB | 4.1 Mbps | **yes** |
| 1000 m | 128.9 dB | −13.9 dB | 0.7 Mbps | **yes** |

A positive Shannon value at −14 dB SNR is a mathematical bound, **not** evidence that a real link
would work at 1000 m. This profile is a placeholder until the wireless technology is decided. It is
not a model of measured Wi-Fi performance.
