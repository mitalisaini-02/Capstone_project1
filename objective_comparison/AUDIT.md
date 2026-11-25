# Repository and formulation audit

The workspace is `capstone_sim`. The comparison uses the existing Phase 2 marketplace inputs, Phase 3 geometry-channel eligibility and SUMO mobility. Phase 4 and Phase 5 outputs use different objectives or session semantics. The comparison operates on independent one-second decisions.

| Variable | Source | Status | Units and interpretation |
|---|---|---|---|
| t, Δt, positions | SUMO mobility.csv, Phase 3 request_seconds | Existing | Seconds and map metres |
| D_j | required_rate_mbps × Δt / 8000 | Derived | Decimal GB due in the current epoch |
| P_i, P_g | pue_resources, gnodeb_resources | Existing | Price units/GB |
| d_ij | Delivered link rate × Δt / 8000 | Decision | Decimal GB |
| x_ij | PUE selection | Decision | Binary, at most one per request, with positive volume required |
| s_j | Sum of selected PUE indicators | Assumed definition | MEC acquisition charged for one PUE selection in an epoch |
| η | Experiment configuration | Assumed | Fixed price units per acquisition |
| r_ij | Phase 3 realized_link_capacity_mbps | Existing | Mbps, legacy per-link limit retained |
| B_ij, B_j^g | Scheduled radio bandwidth | Decision | MHz resource shares within configured carrier sizes |
| γ_ij, γ_j^g | On-the-fly radio calculations | Derived under assumptions | Linear SINR |
| C4/C5/C7 | PUE and cell resource files | Existing | Mbps resource limits |
| C8 | Phase 3 serving site and coverage | Existing | Coverage flag |
| C9 | CUE max_price_per_gb × epoch volume | Derived | Price units, includes the fee |
| y_j | Existing Phase 4 admission variable | Fixed at 1 | The new residual term charges full D_j; rejecting demand would require changing the formula or adding an admission objective |
| p̂_ij, θ | Existing LSTM and corrected checkpoint metadata | Derived / existing | Probability, threshold 0.62 |

The cost expression bills all residual demand to gNodeB. Accordingly, a feasible epoch serves all active demand, and infeasible epochs are recorded explicitly. Heuristic partial allocations are diagnostics only. The rate objective is the achievable scheduled radio rate in Mbps. Delivered traffic is tracked independently and is capped by demand. Bandwidth cannot be inferred from throughput in the resource files.

The inference path uses the existing 7-input, 64-hidden-unit, two-layer unidirectional LSTM (dropout 0.3), training-fitted per-feature scaling, and online 10-s windows sampled at 1 Hz. In feature order it uses distance, relative speed, provider speed, consumer speed, delta x, delta y, and closing speed. The model predicts 30-s survival within 1 km. Candidate flags retain the existing availability and legacy radio gates and require fresh inference at the validated threshold. The project has no gNodeB propagation/transmit profile or interference schedule. These are explicit simulation assumptions in the new configuration.
