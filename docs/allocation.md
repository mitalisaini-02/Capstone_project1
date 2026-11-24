# Phase 4 — Marketplace Allocation and Optimization

Per-second **snapshot** allocation: every active second is solved independently. Session
continuity and reallocation are Phase 5. Package: `src/marketplace/`. It consumes Phase 2/3
**files only** and imports no reliability, radio, feature, model or simulation code (enforced by a test).

## Inputs (nothing recomputed)
| Quantity | Source |
|---|---|
| Eligible links | Phase 3 `candidate_links`: `channel_scenario == C` and `feasible_<θ> == True` |
| B_l (C3) | `realized_link_capacity_mbps` (**realized**; `expected_*` columns are never read) |
| A_i (C4), C_i (C5) | `access_capacity_mbps`, `backhaul_capacity_mbps` (asserted equal to Phase 2) |
| G_k (C7), cov_j, serving cell | Phase 3 `request_seconds` |
| π_i, π_g, π_j^max, w_j, d_j | Phase 2 `pue_resources`, `gnodeb_resources`, `cue_requests` |

## Formulation (per second; x_l, y_j binary; r_l, g_j ≥ 0)
```
(C1) Σ_{l∈L(j)} x_l ≤ y_j        (C3) r_l ≤ B_l x_l        (C4) Σ_{l∈L(i)} r_l ≤ A_i
(C5) Σ_{l∈L(i)} r_l ≤ C_i         (C6) Σ_{l∈L(j)} r_l + g_j = d_j y_j
(C7) Σ_{j∈J(k)} g_j ≤ G_k         (C8) g_j ≤ d_j cov_j      (C9) Σ π_i r_l + π_g g_j ≤ π_j^max d_j y_j
Stage 1: W* = max Σ w_j y_j          Stage 2: min Σ π_i r_l + π_g Σ g_j  s.t. Σ w_j y_j ≥ W*
```
- Solver: HiGHS (`scipy.optimize.milp`), `mip_rel_gap = 0`. The objective is scaled to (price/GB)·Mbps.
  Payments = Δt/8000 × the scaled cost (`pricing.py`).
- **Phantom-PUE correction:** the recorded `pue_id` is derived from the **rates**: only r_l > 1e-6 Mbps
  counts, never x. A request served entirely by the gNodeB has `pue_id = null`. HiGHS returns
  x = 1 with r = 0 in some epochs ("phantom x"). These are counted (`phantom_x`) and never recorded
  as assignments.
- **Independent audit** (`allocation.audit`) re-checks every solution from the allocation alone,
  per family: C1, C2 (link belongs to the request/PUE), C3, C4, C5, C6, C7, C8, C9, phantom PUE
  (recorded PUE without traffic, or traffic without a recorded PUE), non-negativity.

## Methods (identical EpochProblem objects, i.e. the same eligible-link sets)
- **milp:** the two-stage lexicographic optimum above.
- **greedy:** requests in order (weight desc, arrival asc, request_id asc). Each takes the
  cheapest budget-feasible option given the remaining C4/C5/C7 resources: gNodeB only, or one
  PUE at its cost-minimal rate plus the gNodeB remainder.
- **gnb_only:** the same MILP with no PUE links (reference). It is independent of channel scenario and θ.

## Rejection reasons (fixed order, shared by all methods)
`no_service_option` (uncovered, no eligible PUE; the detail lists the Phase 3 first-failure counts) →
`insufficient_capacity_standalone` → `budget_exceeded` → `capacity_contention`.

## Sanity properties checked on every epoch
- MILP admits ≥ greedy and ≥ gNodeB-only.
- At equal admission, the MILP cost is ≤ theirs.
- Expected violations: 0.

## Reproducibility and freeze
- `config/marketplace/phase4.json` (no tunable parameters).
- `run --split val`, then `freeze` (SHA-256 of all Phase 4 code and config). `run --split test`
  refuses to start without the freeze file or if any hash changed.
- `repro-check` re-runs scenarios in a scratch directory and compares output hashes.

## Limitations
- Snapshot allocation: admission can change from second to second (no sessions, Phase 5).
- Posted prices only (no auctions or dynamic pricing). Equal priority weights in the Phase 2 data.
- Inherits all Phase 3 channel limitations (G optimistic, no SINR, LTE-derived mapping, …).
