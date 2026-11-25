# MEC marketplace experiment

## Findings

Four algorithms ran on 156 existing held-out scenarios from 4 SUMO runs, using the Phase 3 geometry channel `G`. Each method evaluated 110,287 active one-second decisions and 221,530 request-seconds. The demand per active snapshot had a median of 7.164 Mbps and a range of 1.026–58.451 Mbps. All four methods returned fully feasible allocations on 32,413 common decisions.

Greedy had the lowest summed cost among the returned paired allocations. Exact's Stage 1 separately establishes the minimum cost before the allowed Stage 2 increase. Exact had the highest mean achievable radio rate in the paired comparison. Best-SINR had the lowest mean runtime. The largest number of fully feasible decisions was 32,417, achieved by Random, Exact. The rates in the table are scheduled radio capacities. The demand actually delivered can be reconstructed from the allocated GB or read in the raw results. An epoch with failed full service has no valid objective cost.

## Paired cost and rate

| algorithm | eta | total_cost | mean_cost_per_consumer | cost_per_gb | mean_snapshot_rate_mbps | mean_rate_per_consumer | pue_pct | stage1_min_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 5e-05 | 23.6915 | 0.000517169 | 0.966102 | 29.2118 | 20.6689 | 9.56544 | — |
| Random | 5e-05 | 23.79 | 0.000519318 | 0.970117 | 30.49 | 21.5733 | 8.63882 | — |
| Best-SINR | 5e-05 | 23.8811 | 0.000521308 | 0.973834 | 29.9902 | 21.2197 | 12.7037 | — |
| Exact | 5e-05 | 23.6938 | 0.000517219 | 0.966196 | 30.5382 | 21.6074 | 9.54503 | 23.6915 |

### Paired cost, capacity and volume components

| algorithm | pue_cost | gnb_cost | mec_fee | pue_rate_sum_mbps | gnb_rate_sum_mbps | sum_snapshot_rate_mbps | pue_gb | gnb_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 1.34034 | 22.1771 | 0.1741 | 109,281 | 837,561 | 946,843 | 2.34571 | 22.1771 |
| Random | 1.21472 | 22.4043 | 0.17095 | 107,765 | 880,508 | 988,272 | 2.11848 | 22.4043 |
| Best-SINR | 2.21239 | 21.4075 | 0.26125 | 151,759 | 820,314 | 972,073 | 3.11531 | 21.4075 |
| Exact | 1.33763 | 22.1821 | 0.1741 | 109,286 | 880,549 | 989,835 | 2.34071 | 22.1821 |

The capacity-sum columns add the instantaneous Mbps totals across one-second snapshots. Use the mean rate for a time-average comparison.

The stage-one cost column is the sum of the per-epoch proven minima. Stage two may increase cost by up to epsilon in each epoch. A heuristic outside an epoch's cost band can achieve a higher rate at a higher price. The paired solver audit found 0 costs below the Stage 1 bound and 0 rates above the Stage 2 bound within the band, using numerical comparison tolerances of 10⁻⁹ price units and 10⁻⁶ Mbps.

## All-epoch service and feasibility

| algorithm | feasible_decisions | feasibility_pct | served | partial | unserved | violations | mean_runtime_ms | total_runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 32413 | 29.3897 | 108004 | 0 | 113526 | 113526 | 0.395768 | 43.6481 |
| Random | 32417 | 29.3933 | 108086 | 0 | 113444 | 113444 | 5.14534 | 567.464 |
| Best-SINR | 32416 | 29.3924 | 108051 | 0 | 113479 | 113479 | 0.294177 | 32.4439 |
| Exact | 32417 | 29.3933 | 45818 | 0 | 175712 | 175712 | 4.42528 | 488.051 |

Counts are request-seconds and include repeated service to the same physical consumer. The solver records an empty diagnostic output for a proven-infeasible epoch. Its unmet-demand count reflects the absence of a full-service solution. Heuristics can report partial progress from fully served individual requests in the same epoch. See the per-family violations in the CSV. The number of violations does not measure the relative quality of these infeasible outputs. 

## Model and protocol

See [repository and variable audit](../AUDIT.md) for the complete mapping. Costs include PUE GB at the posted price, remaining GB at the gNodeB price and the MEC fee for a PUE acquisition. The fixed, assumed fee was **η = 5e-05 price units per PUE acquisition** in every epoch and method. Epsilon was fixed at **1e-06 price units per epoch** (2% of one fee) to allow a small absolute cost tolerance. The method set uses the same budget, coverage, provider access and backhaul resources, cell capacity, legacy link peak, one-PUE rule, and radio resource limits. The positivity floor for an active allocation in the solver was 1e-05 Mbps.

- Greedy processes requests by descending priority, ascending arrival and request ID. It examines the gNodeB-only option and the cost-minimizing endpoint of each feasible relay split interval. Within a local epsilon tie it prefers higher PUE spectral efficiency. It then assigns spare bandwidth in descending efficiency order to selected links. The procedure has no global cost guarantee.
- Random generates 16 seeded attempts per epoch by shuffling request order, uniformly sampling a feasible split interval per option and sampling an option. Incomplete attempts remain in the log. Among feasible attempts it finds the minimum cost and then picks the highest rate within the epsilon band. When all attempts fail it reports the attempt serving the most requests, with delivered traffic as a tiebreaker. The master seed was 42 and each scenario and timestamp derived a stable seed from SHA-256.
- Best-SINR uses the same sequential order as Greedy and gives first preference to the feasible relay with highest instantaneous SINR. The splitter uses the largest feasible relay share. It uses gNodeB alone only when no feasible relay split exists. The heuristic's cost and rate are evaluated independently.
- Exact uses two HiGHS MILPs. Stage 1 minimizes price and fees. Stage 2 maximizes radio rate with cost bounded by Stage 1 plus epsilon. Fixed SINR makes each log coefficient constant. Positive traffic and bandwidth are enabled by binary selection, which removes the selection–volume product. Bandwidth can be capped at the point at which a PUE's link limit saturates, so the minimum operator becomes linear without loss under this model. Solver time limit was 3.0 s per stage with requested relative gap 0.0. The raw results store both stages and are independently checked against the evaluator.

## Radio and inference

SINR is dynamically derived from the existing scenario and mobility data using the configured propagation and interference model. It is a simulation-derived quantity, not a directly measured SINR value from the original dataset.

The PC5 link uses the repository TR 37.885 urban V2V path loss at 6 GHz, 23 dBm, 20 MHz, 3 dBi at both ends, and 9 dB receiver noise figure. Candidate path states come from Phase 3 geometry realizations. The calculation uses median NLOSv blockage and zero additional shadow-fading draws. Out-of-range interfering paths without a state use LOS. Available providers split by vehicle ID modulo reuse channels. Each is assumed active on its reuse channel during the snapshot. Reuse channels are modeled as orthogonal 20 MHz carriers. Activity is held fixed during allocation. All configured sites transmit simultaneously on one reused downlink carrier, orthogonal to PC5. Serving cell shares bandwidth among CUEs. The gNodeB surrogate uses the same LOS propagation equations with the configuration's 3.5 GHz, 46 dBm and antenna overrides. The surrogate is an explicit assumption because the repository provides site positions and Mbps capacities without gNodeB RF measurements. Thermal noise is -174 dBm/Hz plus full-carrier bandwidth and receiver noise figure. Desired and interfering powers are converted into mW before forming the ratio. Integer channel choice and changing mobility affect the derived ratios. The per-scenario link tables include the intermediate radio quantities and eligibility.

The checkpoint metadata records a mobility-run-disjoint split. Every experiment trace belonged to the selected test split and was absent from training. The existing checkpoint, model configuration and scaler matched their recorded hashes before and after the run. The existing online features and fitted scaler produced fresh inference probabilities in [0, 1]. Inference used evaluation mode and the model's parameters were read from disk. The applied threshold for all methods was **0.62** from corrected metadata. Its range and horizon describe link survival independently of the radio SINR. Fresh probabilities matched the existing Phase 3 cache to the validation tolerance with the same eligibility decisions.

| SUMO run | n_candidates | n_predictions | min_probability | max_probability | max_cached_delta | n_pues | n_cues |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run_004 | 4,205 | 3,915 | 0.00121748 | 0.994791 | 2.98023e-08 | 28 | 18 |
| run_005 | 7,970 | 7,415 | 0.00112948 | 0.995754 | 5.96046e-08 | 42 | 28 |
| run_010 | 20,677 | 19,310 | 0.00109867 | 0.996239 | 5.96046e-08 | 60 | 40 |
| run_012 | 3,206 | 3,013 | 0.0012118 | 0.995527 | 5.96046e-08 | 28 | 18 |

## Random attempts and solver validation

Random generated 1,764,592 attempts, of which 518,428 passed the full evaluation. The statistics pool feasible trials over all epochs. Per-epoch statistics are also available. Lower cost and higher rate define the favorable extremes of their respective metrics.

| metric | count | min | mean | median | max | std | mean_within_epoch_std | selected_mean | selected_best | selected_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cost | 518,428 | 0.000127981 | 0.000747902 | 0.000648 | 0.00375948 | 0.000541812 | 8.45871e-06 | 0.000734072 | 0.000127981 | 0.00361875 |
| rate | 518,428 | 7.50782 | 30.1542 | 20.3569 | 285.215 | 29.5607 | 1.61672 | 30.4937 | 285.215 | 7.50782 |

| stage1_status | stage2_status | proven_optimal | decisions |
| --- | --- | --- | --- |
| 0 | 0 | False | 1 |
| 0 | 0 | True | 32416 |
| 2 | — | False | 77870 |

HiGHS statuses are 0 for optimal, 1 for a resource or iteration limit, 2 for infeasible, 3 for unbounded and 4 for solver error. An empty Stage 2 status means the stage was skipped. The strict proof flag also requires both reported gaps to equal exactly zero. A solver status of 0 with a tiny floating-point gap can therefore appear with the flag unset. The maximum reported Stage 1 and Stage 2 gaps were 1.38252e-16 and 0. Runtime includes the objective evaluator. Shared snapshot time is logged separately. Full-run wall time was 742.2 seconds.

Small-case validation checks distance, path loss, received power, thermal noise, multiple interferers, isolated transmitters, movement, reproducibility and invalid coordinates. A hand-computed 8 Mbps example uses 6 Mbps PUE plus 2 Mbps gNodeB, costs 0.00130 price units including the fee and provides 14 Mbps achievable radio capacity. All four methods were exercised. The audit tests also inject invalid demands, excessive resource use and a reliability failure.

## Limitations

The full-demand equation is infeasible when the fixed budgets, radio links and network capacity cannot support all requests simultaneously. The feasible subset defines the paired objective comparison and can favor easier epochs. The results describe an independent snapshot experiment using held-out SUMO mobility and synthetic demand, budgets and prices. Channel reuse was held fixed throughout each decision. A scheduler with allocation-dependent interference could change the ranking and would require a different optimization model. A link carrying a small positive volume can contribute spare scheduled capacity to the rate objective. The delivered traffic metric remains capped by demand, backhaul and cell throughput. Only the geometry channel and the primary threshold were evaluated.

## Reproduction and figures

Run from the repository root with the project's existing Python environment.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest objective_comparison.test_objectives
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m objective_comparison.run --output objective_comparison/reproduction
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m objective_comparison.report objective_comparison/reproduction
```

- [Cost, rate, service, feasibility and runtime](figures/comparison.png)
- [Traffic contributions, tradeoff and Random variability](figures/traffic_tradeoff_random.png)
- [Solver gaps](figures/solver_gaps.png)
- [Epoch results](raw_results.csv.gz), [paired summary](paired_summary.csv), [all-epoch summary](all_epoch_summary.csv) and [scenario results](scenario_results.csv) and [scenario configuration results](scenario_config_results.csv)
- [Random trials](random_trials.csv.gz), [per-epoch Random statistics](random_trial_statistics.csv), [solver status](solver_status.csv), [validation](validation.json) and [experiment metadata](metadata.json)
