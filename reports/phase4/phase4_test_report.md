# Phase 4 — Held-out Test Evaluation Report

*Marketplace allocation with LSTM-based link reliability (per-second snapshot). Test split run **once** with the frozen
implementation; this report only reads recorded outputs. Generated from `data/allocation/phase4/test/` and
`data/allocation/phase4/val/`. Data: real SUMO mobility; PUE/CUE/gNodeB marketplace data are SYNTHETIC (Phase 2).*

## 1. Summary
* **Validity:** test executed with code/configuration hashes identical to the Phase 4 freeze manifest; 156/156 test scenarios
  (runs run_004, run_005, run_010, run_012, disjoint from validation runs); **0 constraint-audit violations, 0/4 dominance violations**,
  20,152 phantom-x solver artefacts correctly not recorded as PUE assignments.
* **Reliability-aware MILP vs gNodeB-only (channel G, θ = 0.62), test:** admitted request-seconds
  48.33% → 50.20% (pooled); per-scenario paired difference
  +1.65 pp
  (95% CI +1.30 to +2.02);
  cost per GB 1.0000 → 0.9631; PUE traffic share 9.80%.
  MILP admits more in 95 of 156 scenarios and never fewer.
* **MILP vs Greedy:** practically identical (identical admission and payment in 127/156 test scenarios; MILP admits more in
  6, never fewer; +0.017 pp) at ~50× the solver time.
* **Validation → test:** every direction and ranking found on validation holds on test. Absolute admission is 3.6–4.4 pp lower on
  test for **all** methods including gNodeB-only (a property of the test mobility runs, not of the methods); the gain over
  gNodeB-only is slightly smaller (+1.65 vs +1.92 pp) and the cost saving slightly larger (−0.029 vs −0.025 per GB).
* **Channel scenarios:** F3 (all clear, optimistic bound) > G (building geometry, primary) > F1 (road topology) ≈ F2 (all blocked, conservative bound).
  F1 ≈ F2 because the road-topology classifier labels ~99% of in-range pairs NLOS in this urban map (Phase 3 data: 926
  feasible links in both on a 20-scenario sample).
* **θ = 0.65 vs 0.62:** slightly fewer eligible links → slightly lower admission (−0.09 pp) and PUE share; same conclusions.

## 2. Methods compared and availability
| Method | In Phase 4 test outputs | Notes |
|---|---|---|
| gNodeB-only baseline | yes | same MILP with no PUE links; independent of channel and θ (one row) |
| Reliability-aware Greedy | yes | cheapest feasible option per request (priority, arrival, id order) |
| Reliability-aware MILP | yes | exact two-stage lexicographic MILP (max weighted admission, then min cost), HiGHS, gap 0 |
| **Reliability-unaware policy** | **no — not part of Phase 4** | introduced only in Phase 5 (session layer); **no Phase 4 test metric exists for it** |

Channel scenarios G, F1, F2, F3 × thresholds θ = 0.62 (primary) and 0.65 (sensitivity) for both aware methods.
Units: rates Mbps, volume GB (1 GB = 8000 Mb), price units per GB (gNodeB price = 1.0). Shares are pooled over all
request-seconds / delivered volume of the 156 scenarios unless stated.

## 3. Admission, cost per GB, traffic shares (validation vs test)
| Method | Channel | θ | Admitted (val) | Admitted (test) | Δ test−val (pp) | Cost/GB (val) | Cost/GB (test) | PUE share (val) | PUE share (test) | gNodeB share (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| gNodeB-only | – | – | 51.94% | 48.33% | -3.61 | 1.0000 | 1.0000 | 0.00% | 0.00% | 100.00% |
| Greedy (aware) | G | 0.62 | 54.32% | 50.17% | -4.15 | 0.9728 | 0.9632 | 9.31% | 9.75% | 90.25% |
| Greedy (aware) | G | 0.65 | 54.15% | 50.09% | -4.06 | 0.9740 | 0.9646 | 8.83% | 9.32% | 90.68% |
| Greedy (aware) | F1 | 0.62 | 53.09% | 49.18% | -3.91 | 0.9849 | 0.9799 | 4.77% | 4.97% | 95.03% |
| Greedy (aware) | F1 | 0.65 | 53.03% | 49.15% | -3.88 | 0.9854 | 0.9805 | 4.58% | 4.83% | 95.17% |
| Greedy (aware) | F2 | 0.62 | 53.09% | 49.18% | -3.91 | 0.9849 | 0.9799 | 4.77% | 4.97% | 95.03% |
| Greedy (aware) | F2 | 0.65 | 53.03% | 49.15% | -3.88 | 0.9854 | 0.9805 | 4.58% | 4.83% | 95.17% |
| Greedy (aware) | F3 | 0.62 | 55.40% | 50.98% | -4.42 | 0.9599 | 0.9485 | 13.83% | 13.84% | 86.16% |
| Greedy (aware) | F3 | 0.65 | 55.15% | 50.82% | -4.33 | 0.9617 | 0.9511 | 13.02% | 13.14% | 86.86% |
| MILP (aware) | G | 0.62 | 54.33% | 50.20% | -4.13 | 0.9728 | 0.9631 | 9.34% | 9.80% | 90.20% |
| MILP (aware) | G | 0.65 | 54.16% | 50.11% | -4.04 | 0.9740 | 0.9645 | 8.85% | 9.37% | 90.63% |
| MILP (aware) | F1 | 0.62 | 53.09% | 49.19% | -3.90 | 0.9849 | 0.9798 | 4.78% | 4.99% | 95.01% |
| MILP (aware) | F1 | 0.65 | 53.04% | 49.17% | -3.87 | 0.9854 | 0.9804 | 4.59% | 4.85% | 95.15% |
| MILP (aware) | F2 | 0.62 | 53.09% | 49.19% | -3.90 | 0.9849 | 0.9798 | 4.78% | 4.99% | 95.01% |
| MILP (aware) | F2 | 0.65 | 53.04% | 49.17% | -3.87 | 0.9854 | 0.9804 | 4.59% | 4.85% | 95.15% |
| MILP (aware) | F3 | 0.62 | 55.41% | 51.02% | -4.39 | 0.9599 | 0.9484 | 13.86% | 13.91% | 86.09% |
| MILP (aware) | F3 | 0.65 | 55.16% | 50.86% | -4.30 | 0.9617 | 0.9510 | 13.05% | 13.21% | 86.79% |

## 4. Service mix and rejection reasons (test)
"Served" columns are mean shares over scenarios; rejection columns are request-second counts (first applicable reason in
the fixed order: no service option → insufficient capacity → budget exceeded → capacity contention).

| Method | Channel | θ | Request-seconds | PUE only | PUE+gNodeB | gNodeB only | Rejected | No service option | Insufficient capacity | Budget exceeded | Capacity contention |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gNodeB-only | – | – | 221530 | 0.00% | 0.00% | 46.48% | 53.52% | 12511 | 15619 | 84987 | 1352 |
| Greedy (aware) | G | 0.62 | 221530 | 3.07% | 0.35% | 44.69% | 51.89% | 11681 | 14627 | 82761 | 1312 |
| Greedy (aware) | G | 0.65 | 221530 | 2.94% | 0.32% | 44.78% | 51.96% | 11735 | 14694 | 82833 | 1313 |
| Greedy (aware) | F1 | 0.62 | 221530 | 1.39% | 0.57% | 45.29% | 52.74% | 11985 | 15268 | 83991 | 1346 |
| Greedy (aware) | F1 | 0.65 | 221530 | 1.34% | 0.55% | 45.34% | 52.77% | 12001 | 15284 | 84013 | 1346 |
| Greedy (aware) | F2 | 0.62 | 221530 | 1.39% | 0.57% | 45.29% | 52.74% | 11985 | 15268 | 83991 | 1346 |
| Greedy (aware) | F2 | 0.65 | 221530 | 1.34% | 0.55% | 45.34% | 52.77% | 12001 | 15284 | 84013 | 1346 |
| Greedy (aware) | F3 | 0.62 | 221530 | 4.70% | 0.32% | 43.81% | 51.18% | 11303 | 14110 | 81919 | 1261 |
| Greedy (aware) | F3 | 0.65 | 221530 | 4.43% | 0.30% | 43.97% | 51.31% | 11409 | 14213 | 82051 | 1273 |
| MILP (aware) | G | 0.62 | 221530 | 3.08% | 0.36% | 44.70% | 51.87% | 11681 | 14627 | 82761 | 1245 |
| MILP (aware) | G | 0.65 | 221530 | 2.94% | 0.33% | 44.79% | 51.94% | 11735 | 14694 | 82833 | 1249 |
| MILP (aware) | F1 | 0.62 | 221530 | 1.39% | 0.57% | 45.30% | 52.73% | 11985 | 15268 | 83991 | 1313 |
| MILP (aware) | F1 | 0.65 | 221530 | 1.34% | 0.55% | 45.35% | 52.76% | 12001 | 15284 | 84013 | 1314 |
| MILP (aware) | F2 | 0.62 | 221530 | 1.39% | 0.57% | 45.30% | 52.73% | 11985 | 15268 | 83991 | 1313 |
| MILP (aware) | F2 | 0.65 | 221530 | 1.34% | 0.55% | 45.35% | 52.76% | 12001 | 15284 | 84013 | 1314 |
| MILP (aware) | F3 | 0.62 | 221530 | 4.70% | 0.33% | 43.82% | 51.15% | 11303 | 14110 | 81919 | 1175 |
| MILP (aware) | F3 | 0.65 | 221530 | 4.43% | 0.31% | 43.97% | 51.29% | 11409 | 14213 | 82051 | 1189 |

Budget limits dominate rejections (82,761 of the MILP/G/0.62 rejections): many
requests' maximum price is below the cheapest feasible option, which no allocation policy can fix.

## 5. Resource utilisation (mean over scenarios)
PUE utilisation = Σ PUE traffic / Σ min(C4, C5) of the providers with at least one eligible link in that second;
gNodeB utilisation = Σ gNodeB traffic / Σ available capacity of the cells serving requesting CUEs.

| Method | Channel | θ | PUE util. (val) | PUE util. (test) | gNodeB util. (val) | gNodeB util. (test) |
|---|---|---|---|---|---|---|
| gNodeB-only | – | – | n/a | n/a | 16.02% | 13.23% |
| Greedy (aware) | G | 0.62 | 13.10% | 13.39% | 15.43% | 12.77% |
| Greedy (aware) | G | 0.65 | 13.06% | 13.59% | 15.45% | 12.79% |
| Greedy (aware) | F1 | 0.62 | 11.52% | 12.35% | 15.70% | 12.97% |
| Greedy (aware) | F1 | 0.65 | 11.44% | 12.49% | 15.71% | 12.98% |
| Greedy (aware) | F2 | 0.62 | 11.52% | 12.34% | 15.70% | 12.97% |
| Greedy (aware) | F2 | 0.65 | 11.44% | 12.49% | 15.71% | 12.99% |
| Greedy (aware) | F3 | 0.62 | 15.16% | 12.85% | 15.04% | 12.53% |
| Greedy (aware) | F3 | 0.65 | 15.05% | 13.10% | 15.08% | 12.57% |
| MILP (aware) | G | 0.62 | 13.11% | 13.39% | 15.39% | 12.72% |
| MILP (aware) | G | 0.65 | 13.07% | 13.60% | 15.41% | 12.74% |
| MILP (aware) | F1 | 0.62 | 11.53% | 12.35% | 15.65% | 12.92% |
| MILP (aware) | F1 | 0.65 | 11.45% | 12.49% | 15.66% | 12.93% |
| MILP (aware) | F2 | 0.62 | 11.53% | 12.34% | 15.65% | 12.92% |
| MILP (aware) | F2 | 0.65 | 11.45% | 12.49% | 15.66% | 12.93% |
| MILP (aware) | F3 | 0.62 | 15.17% | 12.86% | 15.00% | 12.49% |
| MILP (aware) | F3 | 0.65 | 15.06% | 13.11% | 15.04% | 12.53% |

## 6. Solver runtime and constraint checks (test)
| Method | Channel | θ | Solver time total (s, test) | p95 per epoch (ms, test) | Audit violations (test) | Phantom x handled (test) |
|---|---|---|---|---|---|---|
| gNodeB-only | – | – | 438.8 | 4.89 | 0 | 0 |
| Greedy (aware) | G | 0.62 | 10.0 | 0.13 | 0 | 0 |
| Greedy (aware) | G | 0.65 | 10.1 | 0.13 | 0 | 0 |
| Greedy (aware) | F1 | 0.62 | 10.0 | 0.13 | 0 | 0 |
| Greedy (aware) | F1 | 0.65 | 9.8 | 0.13 | 0 | 0 |
| Greedy (aware) | F2 | 0.62 | 9.9 | 0.13 | 0 | 0 |
| Greedy (aware) | F2 | 0.65 | 10.0 | 0.13 | 0 | 0 |
| Greedy (aware) | F3 | 0.62 | 10.3 | 0.14 | 0 | 0 |
| Greedy (aware) | F3 | 0.65 | 10.0 | 0.13 | 0 | 0 |
| MILP (aware) | G | 0.62 | 520.6 | 7.58 | 0 | 3801 |
| MILP (aware) | G | 0.65 | 528.5 | 7.65 | 0 | 3523 |
| MILP (aware) | F1 | 0.62 | 510.9 | 7.16 | 0 | 1169 |
| MILP (aware) | F1 | 0.65 | 505.5 | 6.70 | 0 | 1113 |
| MILP (aware) | F2 | 0.62 | 517.0 | 7.03 | 0 | 1169 |
| MILP (aware) | F2 | 0.65 | 524.3 | 6.96 | 0 | 1113 |
| MILP (aware) | F3 | 0.62 | 584.7 | 9.60 | 0 | 4326 |
| MILP (aware) | F3 | 0.65 | 574.7 | 9.48 | 0 | 3938 |

Runtime is wall-clock and nondeterministic (excluded from reproducibility checks by the freeze definition).
Dominance checks (MILP must admit ≥ any other feasible method and cost ≤ at equal admission), test totals:
milp_admits_less_than_greedy = 0, milp_costs_more_than_greedy_at_equal_admission = 0, milp_admits_less_than_gnb_only = 0, milp_costs_more_than_gnb_only_at_equal_admission = 0. Validation: milp_admits_less_than_greedy = 0, milp_costs_more_than_greedy_at_equal_admission = 0, milp_admits_less_than_gnb_only = 0, milp_costs_more_than_gnb_only_at_equal_admission = 0.

## 7. Paired comparisons (per-scenario differences, mean and bootstrap 95% CI over scenarios)
"A admits more / less / identical" counts scenarios (of 156) where method A admits more, fewer, or has identical
admission and payment. Cost/GB difference in price units per GB (negative = cheaper for consumers).

| Comparison | Channel | θ | Split | Δ admitted (pp) | Δ cost/GB | Δ PUE share (pp) | A admits more / less / identical |
|---|---|---|---|---|---|---|---|
| MILP vs Greedy | F1 | 0.62 | val | +0.005 [+0.000, +0.011] | -0.000 [-0.000, -0.000] | +0.013 [+0.004, +0.022] | 3 / 0 / 142 |
| MILP vs Greedy | F1 | 0.62 | test | +0.010 [+0.000, +0.028] | -0.000 [-0.000, -0.000] | +0.014 [+0.006, +0.024] | 4 / 0 / 128 |
| MILP vs Greedy | F1 | 0.65 | val | +0.005 [+0.000, +0.011] | -0.000 [-0.000, -0.000] | +0.012 [+0.004, +0.021] | 3 / 0 / 142 |
| MILP vs Greedy | F1 | 0.65 | test | +0.010 [+0.000, +0.026] | -0.000 [-0.000, -0.000] | +0.013 [+0.006, +0.023] | 4 / 0 / 128 |
| MILP vs Greedy | F2 | 0.62 | val | +0.005 [+0.000, +0.011] | -0.000 [-0.000, -0.000] | +0.013 [+0.005, +0.023] | 3 / 0 / 142 |
| MILP vs Greedy | F2 | 0.62 | test | +0.010 [+0.000, +0.027] | -0.000 [-0.000, -0.000] | +0.014 [+0.006, +0.023] | 4 / 0 / 128 |
| MILP vs Greedy | F2 | 0.65 | val | +0.005 [+0.000, +0.011] | -0.000 [-0.000, -0.000] | +0.012 [+0.004, +0.021] | 3 / 0 / 142 |
| MILP vs Greedy | F2 | 0.65 | test | +0.010 [+0.000, +0.028] | -0.000 [-0.000, -0.000] | +0.013 [+0.006, +0.023] | 4 / 0 / 128 |
| MILP vs Greedy | F3 | 0.62 | val | +0.008 [+0.002, +0.018] | -0.000 [-0.000, -0.000] | +0.032 [+0.012, +0.058] | 4 / 0 / 141 |
| MILP vs Greedy | F3 | 0.62 | test | +0.023 [+0.004, +0.053] | -0.000 [-0.000, +0.000] | +0.037 [+0.017, +0.060] | 7 / 0 / 127 |
| MILP vs Greedy | F3 | 0.65 | val | +0.008 [+0.002, +0.017] | -0.000 [-0.000, -0.000] | +0.033 [+0.012, +0.060] | 4 / 0 / 141 |
| MILP vs Greedy | F3 | 0.65 | test | +0.023 [+0.004, +0.053] | -0.000 [-0.000, +0.000] | +0.036 [+0.016, +0.060] | 7 / 0 / 127 |
| MILP vs Greedy | G | 0.62 | val | +0.008 [+0.002, +0.017] | -0.000 [-0.000, -0.000] | +0.031 [+0.013, +0.055] | 4 / 0 / 140 |
| MILP vs Greedy | G | 0.62 | test | +0.017 [+0.003, +0.039] | -0.000 [-0.000, +0.000] | +0.026 [+0.012, +0.042] | 6 / 0 / 127 |
| MILP vs Greedy | G | 0.65 | val | +0.008 [+0.002, +0.017] | -0.000 [-0.000, -0.000] | +0.030 [+0.011, +0.053] | 4 / 0 / 140 |
| MILP vs Greedy | G | 0.65 | test | +0.017 [+0.002, +0.036] | -0.000 [-0.000, +0.000] | +0.025 [+0.012, +0.041] | 6 / 0 / 127 |
| MILP vs gNodeB-only | F1 | 0.62 | val | +0.926 [+0.697, +1.188] | -0.014 [-0.018, -0.011] | +4.587 [+3.636, +5.677] | 60 / 0 / 46 |
| MILP vs gNodeB-only | F1 | 0.62 | test | +0.784 [+0.574, +1.023] | -0.017 [-0.021, -0.012] | +4.236 [+3.378, +5.165] | 64 / 0 / 53 |
| MILP vs gNodeB-only | F1 | 0.65 | val | +0.885 [+0.661, +1.125] | -0.014 [-0.018, -0.011] | +4.437 [+3.476, +5.478] | 60 / 0 / 46 |
| MILP vs gNodeB-only | F1 | 0.65 | test | +0.760 [+0.555, +0.996] | -0.016 [-0.020, -0.012] | +4.098 [+3.268, +4.998] | 64 / 0 / 53 |
| MILP vs gNodeB-only | F2 | 0.62 | val | +0.926 [+0.683, +1.185] | -0.014 [-0.018, -0.011] | +4.587 [+3.564, +5.661] | 60 / 0 / 46 |
| MILP vs gNodeB-only | F2 | 0.62 | test | +0.784 [+0.570, +1.030] | -0.017 [-0.021, -0.013] | +4.236 [+3.335, +5.180] | 64 / 0 / 53 |
| MILP vs gNodeB-only | F2 | 0.65 | val | +0.885 [+0.656, +1.138] | -0.014 [-0.018, -0.011] | +4.437 [+3.408, +5.581] | 60 / 0 / 46 |
| MILP vs gNodeB-only | F2 | 0.65 | test | +0.760 [+0.552, +0.978] | -0.016 [-0.020, -0.012] | +4.098 [+3.253, +4.974] | 64 / 0 / 53 |
| MILP vs gNodeB-only | F3 | 0.62 | val | +2.837 [+2.287, +3.432] | -0.038 [-0.045, -0.031] | +13.178 [+11.129, +15.331] | 109 / 0 / 26 |
| MILP vs gNodeB-only | F3 | 0.62 | test | +2.364 [+1.933, +2.826] | -0.041 [-0.049, -0.034] | +10.993 [+9.276, +12.800] | 101 / 0 / 31 |
| MILP vs gNodeB-only | F3 | 0.65 | val | +2.651 [+2.123, +3.233] | -0.036 [-0.043, -0.030] | +12.473 [+10.562, +14.524] | 109 / 0 / 26 |
| MILP vs gNodeB-only | F3 | 0.65 | test | +2.230 [+1.772, +2.678] | -0.039 [-0.046, -0.032] | +10.456 [+8.746, +12.299] | 101 / 0 / 31 |
| MILP vs gNodeB-only | G | 0.62 | val | +1.921 [+1.546, +2.318] | -0.025 [-0.031, -0.021] | +8.811 [+7.318, +10.325] | 103 / 0 / 30 |
| MILP vs gNodeB-only | G | 0.62 | test | +1.647 [+1.303, +2.017] | -0.029 [-0.035, -0.023] | +7.816 [+6.407, +9.382] | 95 / 0 / 38 |
| MILP vs gNodeB-only | G | 0.65 | val | +1.799 [+1.449, +2.173] | -0.024 [-0.030, -0.020] | +8.396 [+7.068, +9.888] | 103 / 0 / 30 |
| MILP vs gNodeB-only | G | 0.65 | test | +1.573 [+1.237, +1.938] | -0.028 [-0.034, -0.023] | +7.506 [+6.209, +8.938] | 94 / 0 / 38 |
| Greedy vs gNodeB-only | G | 0.62 | val | +1.912 [+1.531, +2.309] | -0.025 [-0.031, -0.021] | +8.780 [+7.303, +10.291] | 102 / 0 / 30 |
| Greedy vs gNodeB-only | G | 0.62 | test | +1.629 [+1.293, +1.993] | -0.029 [-0.035, -0.023] | +7.791 [+6.391, +9.353] | 95 / 0 / 37 |
| Greedy vs gNodeB-only | G | 0.65 | val | +1.790 [+1.436, +2.167] | -0.024 [-0.030, -0.020] | +8.366 [+7.043, +9.862] | 102 / 0 / 30 |
| Greedy vs gNodeB-only | G | 0.65 | test | +1.556 [+1.225, +1.918] | -0.028 [-0.034, -0.023] | +7.481 [+6.191, +8.898] | 94 / 0 / 37 |

## 8. Per-configuration results (channel G, θ = 0.62)
| Configuration | gNodeB-only adm. val/test | MILP adm. val/test | Greedy adm. test | MILP cost/GB val/test | MILP PUE share val/test |
|---|---|---|---|---|---|
| C00_baseline | 60.3% / 55.2% | 62.6% / 56.6% | 56.6% | 0.9760 / 0.9627 | 9.2% / 8.9% |
| C01_avail_low | 60.3% / 55.2% | 60.5% / 55.8% | 55.8% | 0.9929 / 0.9881 | 3.3% / 3.7% |
| C02_avail_high | 60.3% / 55.2% | 64.3% / 57.8% | 57.8% | 0.9522 / 0.9455 | 15.7% / 13.7% |
| C03_demand_low | 60.4% / 55.5% | 62.0% / 56.5% | 56.5% | 0.9843 / 0.9667 | 6.7% / 7.2% |
| C04_demand_high | 55.8% / 57.1% | 58.0% / 58.7% | 58.7% | 0.9709 / 0.9659 | 9.1% / 8.4% |
| C05_price_cheap | 60.3% / 55.2% | 64.6% / 57.7% | 57.7% | 0.9325 / 0.9281 | 14.2% / 14.2% |
| C06_price_expensive | 60.3% / 55.2% | 60.3% / 55.4% | 55.4% | 0.9991 / 0.9932 | 1.5% / 4.8% |
| C07_gnb_insufficient | 39.3% / 36.4% | 43.4% / 39.5% | 39.4% | 0.9538 / 0.9364 | 19.1% / 16.9% |
| C08_coverage_gaps | 40.8% / 34.2% | 44.0% / 37.5% | 37.5% | 0.9695 / 0.9481 | 12.7% / 13.9% |
| C09_withdrawals_some | 60.3% / 55.2% | 62.4% / 56.5% | 56.5% | 0.9812 / 0.9683 | 7.6% / 7.3% |
| S01_highdemand_gnbinsufficient | 31.9% / 31.7% | 36.3% / 35.4% | 35.2% | 0.9416 / 0.9294 | 21.3% / 20.8% |
| S02_lowavail_withdrawals | 60.3% / 55.2% | 60.5% / 55.8% | 55.8% | 0.9971 / 0.9884 | 2.1% / 3.5% |
| S03_expensive_coveragegaps | 40.8% / 34.2% | 41.2% / 35.8% | 35.8% | 1.0003 / 0.9913 | 2.7% / 8.3% |

## 9. Missing or unavailable metrics
* **Reliability-unaware policy:** not evaluated in Phase 4 (exists only in Phase 5); no numbers reported.
* **Link failures / service continuity / reallocation:** Phase 4 is a per-second snapshot without sessions; these are Phase 5 metrics.
* **Consumer-level QoS, delay, energy, battery:** not modelled in Phase 4.
* **Provider revenue split by provider / fairness:** per-request revenue is recorded in `allocation_results`, but no
  aggregate fairness metric was defined before the test run, so none is computed here (no post-hoc metrics on test).
* **PUE utilisation** is undefined (n/a) where no provider has an eligible link (gNodeB-only).

## 10. Limitations
* Marketplace entities (PUE resources, CUE requests, prices, budgets, gNodeB load) are synthetic; mobility is real SUMO.
* Per-second snapshot: no session continuity, switching cost or signalling delay (addressed in Phase 5).
* Reliability is the Phase 1 LSTM (1000 m labels, 30 s horizon); thresholds were chosen on validation only and were not
  changed; channel states use TR 37.885/36.942 abstractions; F1/F3 are approximations / bounds, G is primary.
* Eligible PUE links are sparse (most traffic stays on the gNodeB), so absolute gains are modest (≈ +1.6 pp admission,
  ≈ −3% cost per GB) though statistically clear and consistent across validation and test.
* Bootstrap CIs treat scenarios as independent; scenarios sharing a mobility run are correlated, so CIs may be optimistic.

## 11. Provenance and integrity
* Test outputs: 156 scenarios, 156 manifest rows, every file hash matches `phase4_manifest.csv`;
  `experiment_config.json` code hashes = freeze manifest (`config/marketplace/phase4_frozen.json`, SHA-256 `f5308b352ac84484…`).
* Not rerun: outputs were complete and uncorrupted. No Phase 1–4 code, configuration, artefact or result was modified.
* Machine-readable companions in this folder: `phase4_test_report.json`, `phase4_pooled_val_vs_test.csv`,
  `phase4_per_config_G_primary.csv`.
