# Project Architecture — MEC-Assisted Internet Sharing Marketplace

> Companion documents: [COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md) (full technical
> reference), [PROFESSOR_PROJECT_GUIDE.md](PROFESSOR_PROJECT_GUIDE.md) (readable overview),
> [PROJECT_VIVA_QUESTIONS.md](PROJECT_VIVA_QUESTIONS.md) (questions and answers).
>
> Everything here was checked against the repository on 2026-11-25. Where a component is planned but not
> implemented, it is marked **PENDING**.

---

## 1. Entities and their roles

| Entity | What it is in this project | Where it lives in the repository |
|---|---|---|
| **PUE** (Provider UE) | A vehicle that has its own cellular Internet connection (the *backhaul*) and offers part of it to nearby vehicles for a posted price per GB. Modelled as an NR sidelink **U2N relay UE**. | Mobility: SUMO traces (`data/raw/sumo/<run>/mobility.csv`). Resources and price: Phase 2 `pue_resources.csv.gz`. |
| **CUE** (Consumer UE) | A vehicle that needs Internet service. It sends a request with a data rate, duration and the maximum price it accepts. Modelled as a **U2N remote UE**. | Phase 2 `cue_requests.csv.gz`. |
| **gNodeB** | A cellular base station. It serves any demand the PUEs cannot or should not serve, at a fixed price of 1.0 price unit/GB. Six fixed sites. | `infrastructure.add.xml` (6 `gNodeB` POIs), Phase 2 `gnodeb_resources.csv.gz`, `src/network/coverage.py`. |
| **MEC server** | The marketplace coordinator at the network edge. It sees vehicle positions, the LSTM's predictions and the radio estimates, and decides who serves whom. | Phases 3–4 implement its *decision logic* as offline code. Phase 5 (`src/mec/`) adds session management — **only milestone M1 is implemented**. A `MEC-1` POI exists in `infrastructure.add.xml`, but its location is not used in any calculation. |
| **SUMO** | The open-source traffic simulator that produces realistic vehicle movement on an OpenStreetMap city map. | `scripts/generate_sumo_runs.py`, `data/raw/sumo/` (17 runs). |
| **LSTM reliability predictor** | A neural network that reads the last 10 s of a PUE–CUE pair's movement and outputs the probability that the pair stays within 1000 m for the next 30 s. | `src/models/lstm.py`, `src/models/predictor.py`, `src/reliability/`, checkpoint in `outputs/large_dataset_experiment/checkpoints/`. |
| **Wireless channel model** | Decides whether the PUE–CUE radio link actually works and how many Mbps it carries. Uses building outlines (LOS/NLOS) and 3GPP TR 37.885 equations. | `src/network/` (`buildings.py`, `channel_state.py`, `tr37885.py`), `src/simulation/`. |
| **Marketplace allocation engine** | Every second, decides which requests to admit, which PUE (at most one) serves each request, at what rate, and how much the gNodeB serves. Greedy heuristic and an exact two-stage MILP. | `src/marketplace/`, `scripts/run_phase4_allocation.py`. |

---

## 2. Overall system architecture

```mermaid
flowchart LR
    subgraph Vehicles["Vehicles on the road (SUMO)"]
        PUE1["PUE<br/>provider vehicle<br/>(cellular backhaul)"]
        PUE2["PUE"]
        CUE1["CUE<br/>consumer vehicle"]
        CUE2["CUE"]
    end

    subgraph Edge["MEC server (marketplace coordinator)"]
        DISC["Provider discovery<br/>distance ≤ 1000 m"]
        LSTM["LSTM reliability<br/>P(link survives 30 s)"]
        CHAN["Wireless feasibility<br/>buildings + TR 37.885"]
        ALLOC["Allocation engine<br/>Greedy / two-stage MILP"]
        SESS["Session manager<br/>(Phase 5 — M1 only)"]
    end

    GNB["gNodeB<br/>6 sites, price 1.0/GB"]
    NET(("Internet"))

    PUE1 -- "PC5 sidelink" --> CUE1
    PUE2 -- "PC5 sidelink" --> CUE2
    PUE1 -- "Uu backhaul" --> GNB
    PUE2 -- "Uu backhaul" --> GNB
    GNB --> NET
    CUE1 -. "fallback / residual demand" .-> GNB

    Vehicles -- "positions, speeds,<br/>requests, offers" --> DISC
    DISC --> LSTM --> CHAN --> ALLOC --> SESS
    ALLOC -- "assignments, rates,<br/>payments" --> Vehicles
```

**How to read it.** Vehicles report positions and offers/requests to the MEC server. The MEC server filters
possible provider–consumer pairs in three separate stages (distance, LSTM reliability, radio feasibility),
then allocates. A served consumer receives data from one PUE over the sidelink, from the gNodeB, or from
both (one PUE plus a gNodeB top-up).

---

## 3. End-to-end data flow (files actually produced)

```mermaid
flowchart TD
    OSM["OpenStreetMap city map<br/>osm.net.xml + map (1).osm"] --> SUMO
    SUMO["SUMO runs (17)<br/>data/raw/sumo/run_XXX/mobility.csv"] --> SEQ
    SEQ["Sequence builder<br/>(N, 10 s, 7 features) + 30 s labels<br/>data/processed/large_dataset/"] --> SPLIT
    SPLIT["Run-disjoint split<br/>train 9 runs / val 4 / test 4<br/>data/splits/large_dataset/"] --> TRAIN
    TRAIN["Phase 1: LSTM training + θ selection<br/>outputs/large_dataset_experiment/"]

    SUMO --> CONN["Phase 2: connectivity (≤ 1000 m pairs)<br/>data/derived/connectivity/"]
    CONN --> MKT["Phase 2: synthetic marketplace<br/>156 scenarios × {val, test}<br/>data/synthetic/marketplace/"]

    OSM --> GEO["Building geometry (polyconvert)<br/>data/derived/geometry/"]
    GEO --> CACHE
    SUMO --> CACHE["Phase 3 per-run cache<br/>channel, reliability, episodes<br/>data/derived/phase3_cache/"]
    TRAIN --> CACHE
    CACHE --> FEAS["Phase 3 per-scenario feasibility<br/>request_seconds + candidate_links<br/>data/simulation/phase3/"]
    MKT --> FEAS

    FEAS --> P4["Phase 4 allocation (per second)<br/>greedy / MILP / gNodeB-only<br/>data/allocation/phase4/"]
    MKT --> P4

    FEAS --> P5["Phase 5 MEC sessions<br/>M1 inputs + state (implemented)<br/>M2–M8 PENDING"]
    MKT --> P5
    P4 -. "frozen interfaces (hash-pinned)" .-> P5
```

**Important separation rule.** Each phase consumes only the *files* written by earlier phases. Phase 4
(`src/marketplace/`) imports no LSTM, radio or simulation code, and Phase 5 (`src/mec/`) may import only the
frozen Phase 4 interfaces plus a few utilities. Tests check both rules (`tests/test_marketplace.py`,
`tests/test_mec_state.py`).

---

## 4. Phase-wise pipeline

```mermaid
flowchart LR
    P1["Phase 1<br/>LSTM link-survival<br/>prediction<br/>✅ complete"]
    P2["Phase 2<br/>Synthetic marketplace<br/>data + scenarios<br/>✅ complete"]
    P3["Phase 3<br/>Building-aware channel<br/>+ link feasibility<br/>✅ complete"]
    P4["Phase 4<br/>Greedy / MILP<br/>allocation<br/>✅ val + test complete"]
    P5["Phase 5<br/>Dynamic MEC sessions<br/>🟡 M0–M1 only"]
    P1 --> P3
    P2 --> P3 --> P4 --> P5
    P2 --> P4
```

| Phase | Input | Output | Status (repository evidence) |
|---|---|---|---|
| 1 | SUMO traces | Trained LSTM, θ = 0.62 (primary), θ = 0.65 (sensitivity) | Complete. Checkpoint protected by SHA-256. |
| 2 | SUMO traces + assumed parameters | 156 scenarios per split (val, test) | Complete. 6,216 checks per split passed. Reproducible byte-for-byte. |
| 3 | SUMO, buildings, LSTM, Phase 2 | Per-second candidate links with stage flags | Complete for val and test. |
| 4 | Phase 2 + Phase 3 files | Per-second allocations, metrics, comparisons | Complete for val (frozen 2026-11-25 01:10) and test (written 01:51). |
| 5 | Phase 2 + 3 files, frozen Phase 4 | Sessions, reallocation, continuity metrics | **M0/M1 implemented (config, input adapter, state manager; 35 tests). M2–M8 not implemented.** |

---

## 5. LSTM prediction workflow

```mermaid
flowchart TD
    A["SUMO positions + speeds<br/>(1 s steps)"] --> B["For each PUE–CUE pair at time t<br/>with distance ≤ 1000 m"]
    B --> C{"Both vehicles observed<br/>for ≥ 10 s?"}
    C -- "no" --> X["insufficient_history<br/>→ pair not eligible"]
    C -- "yes" --> D["Build window [t−9, t]:<br/>distance, relative speed,<br/>2 speeds, Δx, Δy, closing speed"]
    D --> E["Scale with training-set scaler"]
    E --> F["2-layer LSTM (64 units)<br/>→ Dense 32 → 1 logit"]
    F --> G["Sigmoid → p̂ = P(pair stays<br/>within 1000 m for next 30 s)"]
    G --> H{"p̂ ≥ θ ?<br/>θ = 0.62 primary<br/>θ = 0.65 sensitivity"}
    H -- "yes" --> I["Reliable → go to wireless check"]
    H -- "no" --> Y["reliability_below_threshold"]
```

---

## 6. Wireless channel evaluation (Phase 3, scenario G)

```mermaid
flowchart TD
    S["Candidate pair at second t<br/>(PUE, requesting CUE, distance d)"] --> A{"Both endpoints inside<br/>building-data area?"}
    A -- "no" --> FB["Fallback (F2 = all NLOS)<br/>flagged, excluded from primary rows"]
    A -- "yes" --> B{"Endpoint inside a<br/>building footprint?"}
    B -- "yes" --> FB
    B -- "no" --> C{"Straight 2-D line crosses<br/>a building interior?"}
    C -- "yes" --> NLOS["NLOS (building-blocked)<br/>P_NLOS = 1"]
    C -- "no" --> D["Building-clear:<br/>P_LOS = min(1, 1.05·e^(−0.0114 d))<br/>P_NLOSv = 1 − P_LOS"]
    D --> E["Per-episode uniform U<br/>U < P_LOS → LOS, else NLOSv"]
    NLOS --> F
    E --> F["Realized SNR = P_tx + G − PL(state, d)<br/>− shadowing·z − NLOSv blockage − noise"]
    F --> G["Realized capacity (TR 36.942 mapping)<br/>0 if SNR < −10 dB; max 88 Mbps at 20 MHz"]
    G --> H["C3: B_ij = min(link_peak_i, realized capacity)"]
    H --> I{"B_ij > 0 and min(C3, C4, C5) > 0?"}
    I -- "yes" --> OK["Wireless + resource feasible"]
    I -- "no" --> OUT["wireless_outage_<cause> or resource_zero"]
```

Scenarios **F1** (same OSM road → LOS/NLOSv, otherwise NLOS), **F2** (everything NLOS, conservative) and
**F3** (everything building-clear, optimistic) replace the geometry test with simpler rules for sensitivity
analysis.

---

## 7. Greedy / MILP allocation workflow (Phase 4, one epoch = one second)

```mermaid
flowchart TD
    IN["Epoch t inputs (files only):<br/>active requests d_j, budgets π_j^max, coverage<br/>eligible links + realized B_l (Phase 3)<br/>A_i, C_i, π_i (Phase 2), cell capacity G_k"] --> M{"Method"}
    M -- "greedy" --> G1["Sort requests: weight ↓, arrival ↑, id ↑"]
    G1 --> G2["For each request: cheapest budget-feasible option<br/>(gNodeB only, or one PUE + gNodeB remainder)<br/>using remaining A_i, C_i, G_k"]
    G2 --> G3["Reduce remaining resources"]
    M -- "milp" --> S1["Stage 1: maximise Σ w_j y_j<br/>(number of admitted requests) → W*"]
    S1 --> S2["Stage 2: minimise Σ π_i r_l + π_g Σ g_j<br/>subject to Σ w_j y_j ≥ W*"]
    M -- "gnb_only" --> B1["Same MILP with all PUE links removed"]
    G3 --> OUT
    S2 --> OUT
    B1 --> OUT
    OUT["Allocation: admitted y_j, PUE (from rates, never from x),<br/>r_l, g_j, payments, rejection reasons"] --> AUD["Independent audit of C1–C9,<br/>phantom PUE, non-negativity"]
    AUD --> DOM["Dominance checks:<br/>MILP admits ≥ others; equal admission → cost ≤"]
```

---

## 8. MEC dynamic session lifecycle (Phase 5 — design intent)

> **Status.** Only the *observation layer* (M1) is implemented: `StateManager` can classify, for any second,
> whether a provider is present, departed, withdrawn, temporarily unavailable or out of range, and whether
> a link is *operational* or *policy-eligible*. The session state machine below (admission, interruption,
> reallocation, accounting) is the **planned** M2–M5 behaviour and is **not implemented**. Its detailed design
> document ("Design Revision 2") is referenced in code but is not stored in the repository.

```mermaid
stateDiagram-v2
    [*] --> Requested: CUE request arrives
    Requested --> Active: admitted (PUE and/or gNodeB)
    Requested --> Blocked: no feasible option (arrival_policy = block)
    Active --> Active: link operational (service continues,\nvolume and payment accrue)
    Active --> Interrupted: provider departed / withdrawn /\nunavailable / out of range / wireless outage
    Interrupted --> Active: reallocated within tolerance\n(new PUE or gNodeB)
    Interrupted --> Dropped: not reallocated within tolerance\n(tolerance value PENDING)
    Active --> Completed: service window ends (WINDOW_END)
    Active --> Ended: CUE departed (CUE_DEPARTED)
    Active --> Censored: trace ends (TRACE_END)
    Completed --> [*]
    Ended --> [*]
    Censored --> [*]
    Dropped --> [*]
    Blocked --> [*]
```

What the implemented M1 code already distinguishes (from `src/mec/state.py`):

| Provider condition at second t | Meaning | Rule |
|---|---|---|
| `PUE_NOT_YET_PRESENT` | Vehicle has not entered the simulation | t < first second in trace |
| `PUE_DEPARTED` | Vehicle left the road network | t > its last second (only possible if that second is before trace end) |
| `PUE_WITHDRAWN` | Provider permanently stopped selling | Phase 2 withdrawal time ≤ t |
| `PUE_UNAVAILABLE` | Temporarily off (on/off availability) or not registered | Phase 2 status |
| `PUE_OUT_OF_RANGE` | Present and available but > 1000 m from the CUE | no Phase 3 candidate row |
| `LINK_OBSERVED` | A candidate link exists; carries *operational* and *policy-eligible* flags | Phase 3 row |

| Request end cause | Rule |
|---|---|
| `WINDOW_END` | Nominal end (start + duration − 1) reached while the CUE is still present |
| `CUE_DEPARTED` | CUE left before the nominal end, and before the trace ended |
| `TRACE_END` | CUE still present when the SUMO run ended — **censored**, not counted as a departure |

---

## 9. Code packages and dependency rules

| Package | Responsibility |
|---|---|
| `src/mobility` | Load SUMO traces; vehicle roles and lifetimes |
| `src/features` | Kinematic features, LSTM labels, sequence windows, scaler |
| `src/models`, `src/training` | LSTM network, predictor wrapper, training loop, checkpoints |
| `src/reliability` | `ReliabilityModel` interface, LSTM adapter, kinematic baseline, `ThresholdPolicy`, online feature builder |
| `src/network` | Link-model interface, TR 37.885 NR sidelink model, buildings, LOS/NLOS classifiers, gNodeB coverage |
| `src/datagen` | Phase 2 synthetic PUE/CUE/gNodeB generation, connectivity, validation |
| `src/simulation` | Phase 3 channel/reliability caches, feasibility stages, validation |
| `src/marketplace` | Phase 4 problem builder, pricing, admission, greedy, MILP, audit, evaluation |
| `src/mec` | Phase 5: `Phase5Config`, `Phase5ScenarioInputs`, `StateManager` (M1 only) |
| `src/common`, `src/utils` | Units (1 GB = 8000 Mb), strict config loader, seeding, SHA-256 integrity |

Rules enforced by tests: no import cycles; `network` does not import `models`/`reliability`/`datagen`;
`marketplace` imports no reliability/radio code; `mec` imports only frozen Phase 4 interfaces, `common`,
`mobility.trace` and `utils`.
