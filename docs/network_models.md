# Wireless Network Models

## 1. Technology framing (decision, Phase 3 preparation)

**NR sidelink UE-to-Network (U2N) relay** — 3GPP Rel-17 (TR 38.836 study; stage-2 TS 23.304).

| Marketplace entity | 3GPP role |
|---|---|
| PUE | U2N **relay UE** |
| CUE | U2N **remote UE** |
| PUE–CUE hop | **PC5 sidelink** |
| PUE Internet connectivity | cellular **Uu** backhaul via the gNodeB |
| MEC server | marketplace coordinator (matching, pricing, monitoring) |
| gNodeB | residual cellular supply; schedules sidelink resources (mode 1, assumed) |

**Device assumptions.** Vehicle-mounted UEs with NR sidelink (PC5) and U2N relay capability.
This is **not** a claim that ordinary smartphones support the architecture. Consumer handsets
generally do not expose sidelink relay.

**Network assumptions.** In-coverage operation. gNodeB-scheduled sidelink resources (mode 1). The
operator authorises relay and remote UEs. Relay-level details (L2 vs L3 relay, PDU sessions,
path switching) are abstracted away.

**Three separate eligibility stages** (never merged):
1. **Geographic discovery:** distance ≤ technology discovery range (1000 m).
2. **LSTM reliability:** P(pair stays within 1000 m for 30 s) ≥ θ (0.62 primary, 0.65 sensitivity).
   This is geometric survival. It is **not** a guarantee of NR link quality.
3. **Wireless link feasibility:** from the link model (SNR / capacity / outage), below.

## 2. Profiles

| Profile | File | Status |
|---|---|---|
| `abstract_sidelink_v0` | `config/marketplace/link_profiles.json` | Phase 2 provisional profile, **unchanged** (hash-frozen with Phase 2) |
| `nr_sidelink_tr37885_v1` (urban) | `config/network/link_profiles_nr.json` | New, TR 37.885-based |
| `nr_sidelink_tr37885_v1_highway` | `config/network/link_profiles_nr.json` | New, highway variant |

Technology profiles (`config/network/technologies.json`): `nr_sidelink_u2n_relay` → `nr_sidelink_tr37885_v1`,
`abstract_sidelink` → `abstract_sidelink_v0`.

## 3. Verified equations for `nr_sidelink_tr37885_v1`

Sources, read from the official 3GPP archive:
- **[37.885]** 3GPP TR 37.885 V15.3.0 (2019-06), "Study on evaluation methodology of new V2X use cases for LTE and NR", §6.1–6.2.
- **[36.942]** 3GPP TR 36.942 V19.0.0, Annex A.1, Tables A.1/A.2.

Units: d in m, f_c in GHz, all losses and SNR in dB, bandwidth in Hz, capacity in Mbps.

**Link states** [37.885 §6.2]: LOS (same street, unblocked), NLOSv (same street, blocked by
vehicles), NLOS (different streets, blocked by buildings).

**LOS probability for same-street links** [37.885 Table 6.2-1], with P(NLOSv) = 1 − P(LOS):
- Highway: d ≤ 475 m: P_LOS = min(1, a·d² + b·d + c), with a = 2.1013·10⁻⁶, b = −0.002, c = 1.0193.
  d > 475 m: P_LOS = max(0, 0.54 − 0.001·(d − 475)).
- Urban: P_LOS = min(1, 1.05·exp(−0.0114·d)).

**Pathloss** [37.885 Table 6.2.1-1], with d = d_3D:
- LOS and NLOSv, highway: PL = 32.4 + 20·log10(d) + 20·log10(f_c)
- LOS and NLOSv, urban: PL = 38.77 + 16.7·log10(d) + 18.2·log10(f_c)
- NLOS (urban only): PL = 36.85 + 30·log10(d) + 18.9·log10(f_c)

**Shadow fading std** [37.885 Table 6.2.1-1]: 3 dB for LOS and NLOSv, 4 dB for NLOS.

**NLOSv extra vehicle-blockage loss** [37.885 §6.2.1]: loss = max(0 dB, X), with X Gaussian in dB:
- Case 1, min antenna height > blocker height: no loss.
- Case 2, max antenna height < blocker height: mean 9 + max(0, 15·log10(d) − 41), std 4.5.
- Case 3, otherwise: mean 5 + max(0, 15·log10(d) − 41), std 4.

**Link budget:**
SNR = P_tx + G_tx + G_rx − PL − shadowing − blockage − (−174 + 10·log10(B) + NF)

**SNR → throughput** [36.942 Annex A.1, attenuated and truncated Shannon bound]:
SE = 0 if SNR < SNR_min, otherwise min(α·log2(1 + SNR), Thr_max), and capacity = B·SE.
With α = 0.6, SNR_min = −10 dB, Thr_max = 4.4 bps/Hz (Table A.1, E-UTRA DL baseline), this
reproduces Table A.2 exactly (verified in tests).

**Probabilistic outputs** (used later for feasibility):
- Median SNR per state: shadowing median 0, blockage median max(0, mean).
- Outage P(SNR < γ): closed-form normal CDF for LOS and NLOS; NLOSv integrates the normal CDF
  over the blockage variable.
- Expected capacity: dense ±8σ trapezoidal grids. Gauss–Hermite was rejected because the
  integrands are discontinuous.
- Both are checked against an independent Monte Carlo sampler written from the TR text.

## 4. Parameters and assumptions

| Parameter | Value | Status |
|---|---|---|
| Carrier frequency (UE–UE) | 6 GHz | [37.885] Table 6.1.1-1 |
| Sidelink bandwidth | 20 MHz | [37.885] gives 10 **and** 20 MHz as baseline. **Assumption:** 20 MHz chosen |
| UE Tx power | 23 dBm | [37.885] Table 6.1.1-1 |
| UE noise figure | 9 dB | [37.885] Table 6.1.1-1 |
| Thermal noise | −174 dBm/Hz | standard |
| Antenna gain (Tx, Rx) | 3 dBi each | [37.885] Table 6.1.4-8 Option 1 max element gain. **Assumption:** pattern not modelled (optimistic) |
| Antenna height / blocker height | 1.6 m / 1.6 m | [37.885] §6.1.2 vehicle Type 2. **Assumption:** all vehicles Type 2, so d_3D = d_2D and blockage Case 3 |
| Scenario | urban | **Assumption** (SUMO network is an OSM city map); highway profile also provided |
| Throughput mapping | α 0.6, −10 dB, 4.4 bps/Hz | [36.942] E-UTRA DL baseline. **Assumption:** applied to NR sidelink (TR 37.885 defines none) |
| Minimum distance | 1 m | **Implementation guard**, not from the TR |

## 5. Calculated values (`nr_sidelink_tr37885_v1`, urban, same street unless noted)

| d | P_LOS | Median SNR LOS / NLOSv / NLOS | Capacity at median LOS / NLOSv / NLOS | Expected capacity (same street) |
|---|---|---|---|---|
| 50 m | 0.594 | 39.7 / 34.7 / 18.5 dB | 88.0 / 88.0 / 73.9 Mbps | 88.0 Mbps |
| 100 m | 0.336 | 34.7 / 29.7 / 9.4 dB | 88.0 / 88.0 / 39.5 Mbps | 87.6 Mbps |
| 200 m | 0.107 | 29.6 / 24.6 / 0.4 dB | 88.0 / 88.0 / 12.8 Mbps | 84.6 Mbps |
| 500 m | 0.004 | 23.0 / 18.0 / −11.5 dB | 88.0 / 72.0 / 0.0 Mbps | 69.7 Mbps |
| 1000 m | 0.000 | 18.0 / 9.0 / −20.6 dB | 71.9 / 37.8 / 0.0 Mbps | 38.9 Mbps |

The **same-street vs different-street** distinction dominates. Same-street links are usable at
1000 m; NLOS (building-blocked) links have zero capacity beyond about 300 m.

## 6. Building-aware channel-state classification

Modules: `src/network/buildings.py` (geometry) and `src/network/channel_state.py` (classifiers).
Building data come from `scripts/build_building_geometry.py` (config `config/network/geometry.json`).

### 6.1 Building data
- Source: `map (1).osm`, the OSM file `osm.net.xml` was built from. It is converted with SUMO
  `polyconvert` 1.28 into the **exact network coordinate system** (UTM 43N + netOffset).
- Output: `data/derived/geometry/buildings.poly.xml` (3,951 building polygons, 0 rejected by
  validation), `coverage.poly.xml` (the OSM download box projected the same way: the area where
  building data exist), and `geometry_manifest.json` (input SHA-256 hashes, tool version, counts).
- Map scenario: **urban** (the network has no motorway/trunk edges). The highway profile is never
  selected for this map, and classifiers refuse a highway LOS model.
- **Incompleteness:** 28.7 % of driven 250 m cells contain no mapped building, and the median
  footprint coverage is 2.3 %. OSM buildings are under-mapped, so **scenario G overstates line of
  sight: it is an optimistic estimate, not ground truth.**
- **No heights** (6 of 3,922 ways carry `building:levels`). Every mapped footprint is assumed to
  block the 1.6 m antenna path (2-D test). Some low structures may therefore be wrongly classified
  as blocking.

### 6.2 Algorithm (scenario G, per candidate pair per second)
1. Either endpoint outside the building-data coverage → `fallback_reason = outside_building_coverage`.
2. Either endpoint inside or on a footprint → `fallback_reason = endpoint_inside_building`.
   Such rows are never counted as LOS.
3. Otherwise, if the direct 2-D segment passes through a building **interior** → **NLOS**
   (P_NLOS = 1). Grazing a corner or running along a wall does not count.
4. Building-clear → urban P(LOS) = min(1, 1.05·exp(−0.0114·d)), P(NLOSv) = 1 − P(LOS).
   **This is a geometry-based extension of TR 37.885**, whose LOS/NLOSv model is defined for
   vehicles in the same street. Here it is applied to every building-clear path, including paths
   between different road edges.

Rows from steps 1–2 receive the probabilities of a labelled fallback scenario (default F2) and have
`fallback = True`, `primary_eligible = False`, and `classification_source = fallback:<label>`. They
are excluded from primary geometry-based results.

**Building vs vehicle blockage are separate mechanisms.** Building blockage is deterministic from
geometry and selects the NLOS pathloss. Vehicle blockage exists only within the probabilistic
NLOSv state (TR 37.885 blockage loss). It is **never** detected from simulated vehicles, because
simulated traffic (46–136 vehicles per run) is far sparser than real traffic. A building-blocked
link never also receives vehicle-blockage loss.

### 6.3 Labelled scenarios (never mixed)
| Label | Classifier | Rule | Role |
|---|---|---|---|
| G | `BuildingGeometryClassifier` | §6.2 | primary (optimistic, incomplete OSM) |
| F1 | `RoadTopologyClassifier` | same OSM way id → urban LOS/NLOSv; different way or internal junction edge → NLOS | documented approximation |
| F2 | `AllBuildingBlockedClassifier` | all NLOS | conservative bound |
| F3 | `AllBuildingClearClassifier` | all urban LOS/NLOSv | optimistic bound |

Every output row records `scenario_label`, `classifier`, `classification_source`, `fallback`,
`fallback_reason`, `primary_eligible`, `building_blocked`, `n_buildings_crossed`, and the state
probabilities.

### 6.4 State over time (`assign_episode_states`)
- Building obstruction is recalculated deterministically from the current positions every second.
- An episode is a maximal run of consecutive seconds in which a pair is a candidate. Each episode
  gets one uniform U, keyed by (seed, run, PUE, CUE, episode start) and identical across
  G/F1/F2/F3.
- Realised state: LOS if U < P_LOS; NLOSv if U < P_LOS + P_NLOSv; otherwise NLOS.
- The state is **not redrawn every second**. On building-clear seconds it changes only when
  P(LOS)(d) crosses the fixed U (as distance grows, LOS → NLOSv at most once). A new episode draws
  a new U.
- This is **not** spatially correlated shadowing, and no shadowing realisations are produced.

### 6.5 Validation-run comparison (runs 000/001/006/013, 38,641 candidate rows; descriptive only)
- G: 119 rows (0.3 %) use the fallback (99 outside coverage, 20 endpoint inside a building).
- Building-blocked share (G, primary rows), by distance:

  | Distance | ≤ 100 m | 100–200 | 200–300 | 300–500 | 500–750 | 750–1000 m |
  |---|---|---|---|---|---|---|
  | Blocked | 1.2 % | 9.8 % | 22.5 % | 30.7 % | 45.9 % | 59.2 % |

- Results by scenario:

  | Scenario | Mean P(NLOS) | Median E[capacity] | Rows with E[cap] ≥ 5 Mbps |
  |---|---|---|---|
  | G | 0.46 | 42.5 Mbps | 56 % |
  | F1 | 0.98 | 0.2 Mbps | 13 % |
  | F2 | 1.00 | 0.2 Mbps | 13 % |
  | F3 | 0.00 | 55.4 Mbps | 100 % |

## 7. Limitations (the model is NOT a complete NR physical-layer simulation)
1. Incomplete OSM buildings and no heights (§6.1). Vegetation and terrain are not modelled.
2. The extension of the same-street LOS model to building-clear paths (§6.2) is not part of TR 37.885.
3. **Spatially correlated shadowing** (TR 36.885, referenced by TR 37.885) is not implemented. Only
   per-second marginal probabilities/expectations and per-episode state draws exist.
4. **No interference:** SNR, not SINR. No resource-pool congestion or half-duplex effects.
5. **Antenna:** the maximum element gain (3 dBi) is used and the radiation pattern is ignored
   (optimistic). All vehicles are assumed Type 2 (1.6 m antennas).
6. **Throughput mapping:** the attenuated Shannon bound with E-UTRA DL parameters from TR 36.942 is
   applied to NR sidelink. NR-SL link-level curves would replace it.
7. Fast fading is not modelled separately.
8. Relay-protocol aspects (L2/L3 relay, PDU sessions, path switching, sidelink resource
   scheduling) are abstracted.
