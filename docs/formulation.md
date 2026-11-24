# Mathematical Formulation — MEC-Assisted Internet Sharing Marketplace

Status: **agreed specification (Phase 1)**. Implementation is planned for Phase 7.
Any later change to this formulation must be recorded in `docs/CHANGELOG.md`.

## 1. Link model

PUE–CUE connectivity is modelled as an **abstracted vehicular sidelink / access link**.
It is *not* modelled as Wi-Fi Direct or any other specific radio technology.

- A pair (i, j) is **in range** at time t if the Euclidean distance between the vehicles
  is at most `R_com = 1000 m`.
- This range is the same one used to define the LSTM's training labels
  (`config/stage3_large_dataset.json → link.comm_range_m`), so the trained model is reused
  without retraining. Changing `R_com` invalidates the LSTM labels and requires retraining.
- Two different PUE resources are kept separate:
  - **Internet backhaul throughput** — what the PUE's own cellular data link can supply.
  - **Sidelink access capacity** — what the PUE's local radio can deliver to nearby CUEs,
    both in aggregate and per individual PUE–CUE link.

## 2. Sets and indices

| Symbol | Meaning |
|---|---|
| `t` | Decision epoch; epochs are `Δt` seconds apart (default `Δt = 1 s`) |
| `P_t` | Registered, available PUEs at `t` (index `i`) |
| `J_t` | Active CUE requests at `t` (index `j`) |
| `K` | gNodeB cells (index `k`); `k(j)` is the serving cell of request `j` |

## 3. Parameters (units)

| Symbol | Meaning | Unit |
|---|---|---|
| `C_i^bh` | PUE `i` Internet backhaul throughput available for sharing | Mbps |
| `A_i` | PUE `i` aggregate sidelink access capacity | Mbps |
| `B_ij` | Individual PUE–CUE sidelink capacity of pair `(i, j)` | Mbps |
| `d_j` | Required data rate of request `j` | Mbps |
| `G_k` | Available capacity of gNodeB cell `k` | Mbps |
| `π_i` | PUE `i` asking price | price units / GB |
| `π_g` | gNodeB price | price units / GB |
| `cov_j ∈ {0,1}` | 1 if request `j` is inside gNodeB coverage | – |
| `p̂_ij ∈ [0,1]` | LSTM link-survival probability (next 30 s) | – |
| `θ` | Reliability threshold (experimental parameter; Phase 1 selection `θ* = 0.62`) | – |
| `w_j > 0` | Admission weight (1 = count requests; otherwise priority) | – |
| `budget_j` | Optional maximum payment per epoch for request `j` | price units |

**Unit conversion.** A rate of `r` Mbps for `Δt` seconds carries `r·Δt/8000` GB
(1 GB = 8000 Mb, decimal units). Cost per epoch is therefore
`(Δt / 8000) · price_per_GB · rate_Mbps`.

## 4. Eligibility (reliability-aware)

```
e_ij = 1  iff  i is registered and available
           and dist_ij(t) ≤ R_com
           and both vehicles have ≥ 10 s of observed history
           and p̂_ij ≥ θ
e_ij = 0  otherwise
```

Pairs with less than 10 s of history have no LSTM prediction and are **ineligible**
(the initial policy; to be revisited in Phase 5).

## 5. Decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `x_ij` | {0, 1} | CUE request `j` is served by PUE `i` |
| `y_j` | {0, 1} | Request `j` is admitted |
| `r_ij` | ≥ 0 (Mbps) | Rate delivered from PUE `i` to request `j` |
| `g_j` | ≥ 0 (Mbps) | Rate delivered by the gNodeB to request `j` |

## 6. Constraints

```
(C1)  Σ_i x_ij ≤ y_j                     ∀ j        at most one PUE, and only if admitted
(C2)  x_ij ≤ e_ij                        ∀ i, j     reliability-aware eligibility
(C3)  r_ij ≤ B_ij · x_ij                 ∀ i, j     individual PUE–CUE sidelink capacity
(C4)  Σ_j r_ij ≤ A_i                     ∀ i        PUE aggregate sidelink access capacity
(C5)  Σ_j r_ij ≤ C_i^bh                  ∀ i        PUE Internet backhaul capacity
(C6)  Σ_i r_ij + g_j = d_j · y_j         ∀ j        admitted requests are fully served (QoS)
(C7)  Σ_{j : k(j)=k} g_j ≤ G_k           ∀ k        gNodeB cell capacity
(C8)  g_j ≤ d_j · cov_j                  ∀ j        gNodeB coverage
(C9)  cost_j ≤ budget_j · y_j            ∀ j        optional consumer budget
```

with `cost_j = (Δt / 8000) · (Σ_i π_i r_ij + π_g g_j)`.

**C3, C4 and C5 are three distinct constraints and are all kept.** C3 bounds each individual
link, C4 bounds the PUE's radio across all its CUEs, and C5 bounds the PUE's Internet supply.

The gNodeB supplies the residual demand `g_j = d_j y_j − Σ_i r_ij` of every admitted request (C6).

## 7. Objective — admission control, then cost (lexicographic)

```
Stage 1:   W* = max  Σ_j w_j y_j                      subject to C1–C9
Stage 2:        min  Σ_j cost_j                       subject to C1–C9 and Σ_j w_j y_j ≥ W*
```

- Stage 1 ensures the optimiser cannot reach zero cost by rejecting requests that can be served.
- Stage 2 is the primary economic objective: **minimise total consumer payment** for the
  admitted set.
- The secondary throughput objective is **removed**. With `d_j` fixed for admitted requests,
  total delivered throughput equals `Σ_j d_j y_j`, which Stage 1 already determines, so a
  throughput stage would have no effect.
- **Future extension (not implemented):** elastic service `d_j^min ≤ Σ_i r_ij + g_j ≤ d_j^max`.
  A throughput objective becomes meaningful only in that variant.

Problem class: mixed-integer linear program (binary `x_ij, y_j`). Planned solver:
`scipy.optimize.milp` (HiGHS).

## 8. Rejection reasons (reported per rejected request)

| Code | Condition |
|---|---|
| `no_coverage_and_no_eligible_pue` | `cov_j = 0` and no `i` with `e_ij = 1` |
| `all_pue_filtered_by_reliability` | in-range PUEs exist but all have `p̂_ij < θ` |
| `insufficient_capacity` | eligible resources exist but cannot carry `d_j` alongside the admitted set |
| `budget_exceeded` | cheapest feasible service for `j` exceeds `budget_j` |
| `displaced_by_higher_priority` | serving `j` would reduce `Σ w_j y_j` |

## 9. Known modelling limitations

- The LSTM's survival labels drop samples where a vehicle leaves within 30 s. In the marketplace,
  a departing PUE is a session failure, so departures are counted separately in monitoring (Phase 9).
- The held-out test runs (`run_004, run_005, run_010, run_012`) contain no 136-vehicle
  (very heavy) traffic, so high-density reliability is not evaluated.
- Phase 1 found that validation precision at `θ* = 0.62` (0.900) dropped to 0.880 on the
  held-out test runs; the intended "≤ 10 % approved-link break rate" is not guaranteed out of sample.
