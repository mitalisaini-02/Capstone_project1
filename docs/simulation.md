# Phase 3 — Mobility and Network Simulation

Scope: mobility, discovery, reliability, channel, link capacity, PUE resources and gNodeB coverage.
**Not included:** pricing, provider selection, MILP/greedy allocation, MEC sessions or reallocation.

## 1. Pipeline

```
SUMO trace (real) ─┬─ geographic discovery (≤ 1000 m)       data/derived/connectivity (Phase 2)
                   ├─ LSTM survival p̂ (unchanged model)     reliability_layer + online_features
                   ├─ channel state G/F1/F2/F3               network.channel_state (+ building geometry)
                   ├─ realized channel per link episode      network.tr37885.realized_*  (+ episode draws)
                   └─ link episodes and why they end          ground truth (evaluation only)
           ── per-run cache: data/derived/phase3_cache/<run>/{channel,reliability,episodes}.csv.gz
Phase 2 scenario (synthetic) ── provider status, C3/C4/C5, requests, gNodeB capacity/coverage
           ── per scenario: data/simulation/phase3/<split>/<scenario>/{request_seconds,candidate_links}.csv.gz
```

The per-run cache depends only on real mobility, building geometry, the NR link model, the LSTM
and the channel seed. It is computed once per run and reused by all 39 scenarios of that run.
This replaces 39 repetitions of the geometry test per run with one, without changing results.

## 2. Expected vs realized capacity (never confused)

| Column | Meaning | Use |
|---|---|---|
| `expected_radio_capacity_mbps` | E[capacity] over channel-state probabilities, shadowing and NLOSv blockage | **Reporting / statistics only** |
| `expected_outage_probability` | P(SNR < −10 dB) under the same distribution | Reporting only |
| `realized_state` | Sampled persistent state of this link episode (LOS/NLOSv/NLOS) | Feasibility |
| `realized_snr_db` | SNR of the sampled channel: state pathloss − σ_state·z_shadow − [NLOSv] max(0, μ_b + σ_b·z_blockage) | Feasibility |
| `realized_radio_capacity_mbps` | TR 36.942 mapping of `realized_snr_db` | **Feasibility (wireless stage)** |
| `realized_link_capacity_mbps` | C3 = min(link_peak_i, realized radio capacity) | **Allocation input (later phase)** |

`feasibility.py` never reads `expected_*`. A test swaps the expected values for contradictory
ones and checks that every feasibility output is unchanged. The bare name `capacity_mbps(d, state)`
in the link model is the capacity at the *median* SNR of a given state, neither expected nor
realized. It is not used in Phase 3 outputs.

**Per-episode realized channel.** In each continuous link episode the state uniform U and the two
standard-normal draws (z_shadow, z_blockage) are fixed. They are keyed by (seed, run, PUE, CUE,
episode start), shared across G/F1/F2/F3, and use separate random streams. Building obstruction and
distance are re-evaluated every second, so SNR changes smoothly with distance and the state changes
only when P(LOS) crosses U or building status changes. **This is not spatially correlated shadowing.**

## 3. Feasibility stages and recorded reasons

Each row is one in-range (PUE, requesting CUE) pair per second per channel scenario. Every stage
has its own flag, and `first_failure_<θ>` gives the earliest failing stage:

| Order | Stage | Failure value |
|---|---|---|
| (pre) | geographic range | counted per request-second (`n_out_of_range_pue`) |
| 1 | provider status (Phase 2) | `provider_unavailable` (`provider_status` = not_registered / unavailable / withdrawn) |
| 2 | LSTM window exists | `insufficient_history` |
| 3 | p̂ ≥ θ (`primary` 0.62, `sensitivity` 0.65) | `reliability_below_threshold` |
| 4 | realized radio capacity > 0 | `wireless_outage_building_blocked` / `_vehicle_blocked` / `_los` / `_nlos_assumed` (F1/F2) / `_fallback` |
| 5 | min(C3, C4, C5) > 0 | `resource_zero` |
| — | all passed | `feasible` |

Also recorded: `binding_constraint` (C3_link / C4_access / C5_backhaul) and `covers_required_rate`
(the single-link bound reaches the request's rate). Joint sharing of C4/C5 between CUEs is
**not** resolved here; that is the optimiser's job. Link departures are in `episodes.csv.gz`
(`end_cause`: out_of_range, pue_departure, cue_departure, both_departure, trace_end).

**gNodeB per request-second:** coverage (nearest site within the scenario radius), serving cell and
that cell's available capacity (`gnb_available_capacity_mbps`, not divided among users).

## 4. Configuration and reproducibility
- `config/simulation/phase3.json`: technology `nr_sidelink_u2n_relay` (link profile
  `nr_sidelink_tr37885_v1`, discovery 1000 m), geometry paths, channel scenarios G/F1/F2/F3 (G
  primary, G fallback F2), channel seed 20261125, and the LSTM checkpoint/metadata. Thresholds are
  read from the recorded files (0.62, 0.65).
- No parameter is tuned. Validation and test runs go through the identical pipeline.
- Outputs are write-once. `experiment_config.json` records all input hashes. A reproducibility
  check rebuilds caches and outputs in a scratch directory and compares every file hash.

## 5. Limitations
- All channel limitations of `docs/network_models.md` §7 apply. In particular: incomplete OSM
  buildings (G is optimistic), no SINR, no correlated shadowing, LTE-derived throughput mapping, and
  the antenna pattern is ignored.
- Per-link single-link bounds (`pue_max_rate_mbps`) ignore other CUEs sharing the same PUE.
- Requests are open-loop (from Phase 2); there is no admission or session logic.
- The LSTM predicts geometric survival within 1000 m, not radio quality. Reliability and wireless
  feasibility are separate stages and both are required.
